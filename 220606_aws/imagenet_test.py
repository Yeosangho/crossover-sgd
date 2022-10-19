import argparse
import copy
import os
import socket
import time
import threading
import random
import sys
import numpy as np
import csv
from itertools import cycle 
from functools import reduce

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

#from apex import amp
#from apex.parallel import DistributedDataParallel as ApexDDP
#from apex.fp16_utils import network_to_half, FP16_Optimizer
from torchvision.models.resnet import Bottleneck
from torch.nn.parameter import Parameter


from lars import LARS
#from resnet import resnet50
#from scheduler import GradualWarmupScheduler, PolynomialLRDecay
#from optimizer import SGD_without_lars, SGD_with_lars, SGD_with_lars_ver2
import math
from gossip_module.utils import flatten_tensors, flatten_tensors_grad, unflatten_tensors, unflatten_tensors_grad
from copy import copy, deepcopy
#from LARC import LARC
import apex.amp as amp
import torch.nn.functional as F




parser = argparse.ArgumentParser(description='Playground')

parser.add_argument('--batch_size', default=64, type=int,
					help='per-agent batch size')

parser.add_argument('--world_size', default=4, type=int)
parser.add_argument('--rank', default=4, type=int)

parser.add_argument('--gpu_per_node', default=2, type=int)
parser.add_argument('--proc_per_gpu', default=1, type=int)
parser.add_argument('--local_itr', default=1, type=int)
parser.add_argument('--warmup_epoch', default=12, type=int)

parser.add_argument('--crossover', default='False', type=str)
parser.add_argument('--chromosome', default='coarse', type=str)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--lars', default='False', type=str)
parser.add_argument('--allreduce', default='False', type=str)
parser.add_argument('--sync_grad', default='False', type=str)
parser.add_argument('--amp', default='False', type=str)
parser.add_argument('--clip_grad', default='False', type=str)
parser.add_argument('--lrdecay', default='step', type=str)
parser.add_argument('--lars_coef', default=0.01, type=float)
parser.add_argument('--baselr', default=40, type=float)
parser.add_argument('--maxepoch', default=160, type=int)
parser.add_argument('--wd', default=5*10**-5, type=float)
parser.add_argument('--sync_lars_start_epoch', default=20, type=int)
parser.add_argument('--sync_lars_group_size', default=2, type=int)
parser.add_argument('--manual_seed', default=1, type=int)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--val_iter', default=1.0, type=int)
parser.add_argument('--proc_per_node', default=2, type=int)
parser.add_argument('--groupnum', default=8, type=int)
parser.add_argument('--test_num', default=20, type=int)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data, self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def send(tensor, send_to):
	dist.send(tensor=tensor, dst=send_to)
def find_block(model_name, name):	
	return [model_name.index(n) for n in model_name if(n in name)]

def main():
	args = parser.parse_args()
	rank = args.rank

	world_size = args.world_size
	group_num = args.groupnum
	batch_size = args.batch_size
	lars_coef = args.lars_coef
	baselr = args.baselr
	maxepoch = args.maxepoch
	lrdecay = args.lrdecay
	local_itr = args.local_itr
	gpu_per_node = args.gpu_per_node
	warmup_epoch = args.warmup_epoch
	sync_lars_start_epoch = args.sync_lars_start_epoch
	sync_lars_group_size = args.sync_lars_group_size
	manual_seed = args.manual_seed
	amp_flag = False
	weight_decay = args.wd
	val_iter = args.val_iter
	proc_per_node =args.proc_per_node
	proc_per_gpu = int(proc_per_node / gpu_per_node)
	if(args.amp == 'True'):
		amp_flag = True
	clip_grad = False
	if(args.clip_grad == 'True'):
		clip_grad = True			
	sync_grad = False
	if(args.sync_grad == 'True'):
		sync_grad = True	
	lars = False
	if(args.lars=='True'):
		lars = True	
	allreduce = False
	if(args.allreduce == 'True'):
		allreduce = True
	crossover_flag = False
	if(args.crossover == 'True'):
		crossover_flag = True

	file_prefix = ""

	if(crossover_flag == True):
		file_prefix = "crossover"
	elif(allreduce == True):
		file_prefix = "allreduce"
	else:
		file_prefix = "sgp"
	if(sync_grad == True):
		file_prefix = file_prefix + "_sync_grad"
	else:
		file_prefix = file_prefix + "_sync_param"
	rseed_per_rank = []
	#last seed is global random seed 
	for i in range(world_size +1):
		rseed_per_rank.append(np.random.RandomState(i+6))


	#if((rank % proc_per_node)< (proc_per_node/2)):	
	#	GPU_NUM = 0
	#else :
	#	GPU_NUM = 1
	GPU_NUM =  rank % gpu_per_node

	#print(f"GPU NUM {GPU_NUM}")
	device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
	torch.backends.cudnn.benchmark = True
	print('__Number CUDA Devices:', torch.cuda.device_count())
	print ('Current cuda device ', torch.cuda.current_device()) # check
	print('__Python VERSION:', sys.version)
	print('__pyTorch VERSION:', torch.__version__)
	print('__CUDA VERSION')
	# call(["nvcc", "--version"]) does not work
	print('__Number CUDA Devices:', torch.cuda.device_count())
	print('__Devices')
	print('Active CUDA Device: GPU', torch.cuda.current_device())
	
	print ('Available devices ', torch.cuda.device_count())
	print ('Current cuda device ', torch.cuda.current_device())

	torch.cuda.set_device(device) # change allocation of current GPU

	# Additional Infos
	if device.type == 'cuda':
	    print(torch.cuda.get_device_name(GPU_NUM))
	    print('Memory Usage:')
	    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
	    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

	dist.init_process_group(backend="nccl", world_size =world_size, rank=rank)

	model = init_model()


	torch.manual_seed(manual_seed)
	def init_weights(m):
		if type(m) == nn.Linear or type(m) == nn.Conv2d:
			torch.nn.init.kaiming_uniform_(m.weight)
	model.apply(init_weights)
	crossover_params = []

	if(args.chromosome == 'coarse'):
		model_name = ['-1','layer1',  'layer2', 'layer3',  'layer4', 'fc']
	elif(args.chromosome == 'fine'):
		#model_name = ['-1','layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1','fc']
		model_name = ['-1','layer1.0', 'layer1.1',  'layer1.2','layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2','fc']

	layer_idx = []
	layer_numel = []
	numel = 0
	for idx, (n, param) in enumerate(model.named_parameters()):
		print(f"parameter_names {n}")
		numel += param.numel()
		if(n == 'conv1.weight' or n == 'bn1.weight' or n == 'bn1.bias'):
			old_block = 0
		else:
			is_changed =  False if (old_block == find_block(model_name, n)[0]) else True
			if is_changed :
				layer_idx.append(idx)
				layer_numel.append(numel)
				print(n)
			old_block = find_block(model_name, n)[0] if(is_changed) else old_block
	layer_idx.append(len(list(model.parameters())))		
	layer_numel.append(numel)
	log_softmax = nn.LogSoftmax(dim=1)
	

	#criterion = nn.CrossEntropyLoss()
	criterion = LabelSmoothingLoss(1000)
	
	#recursive_batch_norm_momentum(model, momentum=1.0)
	
	#optimizer = torch.optim.SGD(model.parameters() , lr=baselr, momentum=0.9, nesterov=True)
	if(lars == True):
		optimizer = LARS(model, model.parameters(), lr=baselr, momentum=0.96, weight_decay=weight_decay, eta=lars_coef, max_epoch=maxepoch, dist=dist, world_size=world_size, amp=amp, rank=rank)
	if(amp_flag == True):
		model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
	optimizer.zero_grad()

	


	iter_num = 0
	global_itr = 0



	itr_time = {}
	co_itr = []
	co_g2_itr = []
	co_g4_itr = []
	co_g8_itr = []
	ar_itr = []
	sgp_itr = []
	itr_time["co"] = co_itr
	itr_time["co_g2"] = co_g2_itr

	itr_time["co_g4"] = co_g4_itr
	itr_time["co_g8"] = co_g8_itr
	itr_time["ar"] = ar_itr
	itr_time["sgp"] = sgp_itr
	peer_groups = {}
	for i in range(world_size):
		for j in range(i+1, world_size):
			group = [i, j]
			group.sort()
			new_group = dist.new_group(ranks=group)
			key = f"{i}:{j}"
			#sys.exit()
			peer_groups[key] = new_group

	group_num = world_size
	groups = [] 
	group_roots = []
	group_size = int(world_size / group_num)
	for x in range(int(world_size / group_size)):
		group = []
		group_roots.append(x * group_size)
		for y in range(group_size):
			group.append(y+ x*group_size)
		new_group = dist.new_group(ranks=group)
		groups.append(new_group)
	
	mygroup = int(rank/group_size)
	roulettes = [] * world_size

	tensor_each_rank = []
	for i in range(world_size):
		tensor_each_rank.append(list(model.parameters())[rank::world_size])
	norm_list = [[0.0, 0.0]] * len(list(model.parameters()))
	norm_tensor = torch.Tensor(norm_list).float().cuda()
	norm_each_rank = []
	for i in range(world_size):
		norm_each_rank.append(torch.Tensor(norm_list[i::world_size]).float().cuda())

	for i in range(0, group_num):
		roulette_except_rank = [1./(group_num -1) for i in range(0, group_num)]
		roulette_except_rank[i] = 0
		roulettes.append(roulette_except_rank)


	#sgp
	sgp_send = []
	sgp_recv = []
	for i in range(int(math.log(world_size,2))):
		send_to = int((rank+(2**(i%(math.log(world_size,2)))))%world_size)

		receive_from = int((rank-(2**(i%(math.log(world_size,2)))))%world_size)
		sgp_send.append(send_to)
		sgp_recv.append(receive_from)


	test_num = args.test_num
	itr_num = 0
	for itr in range(0, test_num*6+1):
		key_name = ''
		select_ranks = []
		if(itr>=1 and itr<=test_num):
			allreduce = True
			crossover_flag = False
			key_name = 'ar'
		elif(itr >test_num and itr<=test_num*2):
			itr_num = itr - test_num -1
##
			allreduce = False
			crossover_flag = False
			key_name = 'sgp'
		elif(itr>test_num*2 and itr <= test_num*3):
			itr_num = itr - test_num*2 -1
##
			key_name = 'co'
			allreduce = False
			crossover_flag = True
			for j in range(len(layer_idx)):
				select_rank = select_layer(rank, rseed_per_rank, group_num, roulettes)
				select_ranks.append(select_rank)
			dist.barrier()
		elif(itr>=test_num*3+1 and itr<=test_num*4):
			itr_num = itr - test_num*3 -1
			key_name = 'co_g2'
			allreduce = False
			crossover_flag = True
			if(itr == test_num*3 +1):	
				group_num = 2		
				groups = [] 
				group_roots = []
				group_size = int(world_size / group_num)
				for x in range(int(world_size / group_size)):
					group = []
					group_roots.append(x * group_size)
					for y in range(group_size):
						group.append(y+ x*group_size)
					new_group = dist.new_group(ranks=group)
					groups.append(new_group)
				
				mygroup = int(rank/group_size)
				roulettes = [] * world_size
				
				tensor_each_rank = []
				for i in range(world_size):
					tensor_each_rank.append(list(model.parameters())[rank::world_size])
				norm_list = [[0.0, 0.0]] * len(list(model.parameters()))
				norm_tensor = torch.Tensor(norm_list).float().cuda()
				norm_each_rank = []
				for i in range(world_size):
					norm_each_rank.append(torch.Tensor(norm_list[i::world_size]).float().cuda())
				
				for i in range(0, group_num):
					roulette_except_rank = [1./(group_num -1) for i in range(0, group_num)]
					roulette_except_rank[i] = 0
					roulettes.append(roulette_except_rank)
			for j in range(len(layer_idx)):
				select_rank = select_layer(rank, rseed_per_rank, group_num, roulettes)
				select_ranks.append(select_rank)
			dist.barrier()						
		elif(itr>test_num*4 and itr<=test_num*5 ):
			itr_num = itr - test_num*4 -1
#
			key_name = 'co_g4'
			allreduce = False
			crossover_flag = True
			if(itr == test_num*4 +1):	
				group_num = 4		
				groups = [] 
				group_roots = []
				group_size = int(world_size / group_num)
				for x in range(int(world_size / group_size)):
					group = []
					group_roots.append(x * group_size)
					for y in range(group_size):
						group.append(y+ x*group_size)
					new_group = dist.new_group(ranks=group)
					groups.append(new_group)
				
				mygroup = int(rank/group_size)
				roulettes = [] * world_size
			
				tensor_each_rank = []
				for i in range(world_size):
					tensor_each_rank.append(list(model.parameters())[rank::world_size])
				norm_list = [[0.0, 0.0]] * len(list(model.parameters()))
				norm_tensor = torch.Tensor(norm_list).float().cuda()
				norm_each_rank = []
				for i in range(world_size):
					norm_each_rank.append(torch.Tensor(norm_list[i::world_size]).float().cuda())
			
				for i in range(0, group_num):
					roulette_except_rank = [1./(group_num -1) for i in range(0, group_num)]
					roulette_except_rank[i] = 0
					roulettes.append(roulette_except_rank)
			for j in range(len(layer_idx)):
				select_rank = select_layer(rank, rseed_per_rank, group_num, roulettes)
				select_ranks.append(select_rank)
			dist.barrier()
		elif(itr>=test_num*5+1 and itr<=test_num*6 and world_size >= 8):
			itr_num = itr - test_num*5 -1
			key_name = 'co_g8'
			allreduce = False
			crossover_flag = True	
			if(itr == test_num*5 +1):	
				group_num = 8		
				groups = [] 
				group_roots = []
				group_size = int(world_size / group_num)
				for x in range(int(world_size / group_size)):
					group = []
					group_roots.append(x * group_size)
					for y in range(group_size):
						group.append(y+ x*group_size)
					new_group = dist.new_group(ranks=group)
					groups.append(new_group)
				
				mygroup = int(rank/group_size)
				roulettes = [] * world_size
				
				tensor_each_rank = []
				for i in range(world_size):
					tensor_each_rank.append(list(model.parameters())[rank::world_size])
				norm_list = [[0.0, 0.0]] * len(list(model.parameters()))
				norm_tensor = torch.Tensor(norm_list).float().cuda()
				norm_each_rank = []
				for i in range(world_size):
					norm_each_rank.append(torch.Tensor(norm_list[i::world_size]).float().cuda())
				
				for i in range(0, group_num):
					roulette_except_rank = [1./(group_num -1) for i in range(0, group_num)]
					roulette_except_rank[i] = 0
					roulettes.append(roulette_except_rank)
			for j in range(len(layer_idx)):
				select_rank = select_layer(rank, rseed_per_rank, group_num, roulettes)
				select_ranks.append(select_rank)
			dist.barrier()
		elif(itr != 0):
			break
		print(key_name)
		batch_time = time.time()

		#final polishing
		#if(epoch == 85):	
		#	loader, sampler = make_dataloader_no_data_aug(rank, batch_size, world_size)
		print('1')

		#update_learning_rate(lrdecay, baselr, optimizer, maxepoch, epoch, 0, len(loader), world_size, batch_size, warmup_epoch)
		batch = torch.rand(batch_size, 3, 224,224, dtype=torch.float32).cuda()

		target = torch.zeros(batch_size, 1000, dtype=torch.int64).cuda()
		print('2')

		#print(rank)

		global_itr = global_itr + 1
		minibatch_time = time.time()
		model.train()

		target = target.cuda()

		batch = batch.cuda()

		r = np.random.rand(1)
		loss = 0
		print('3')


		output = model(batch)
		print(output.shape)
		loss = criterion(output, target)


		loss = loss /local_itr

		with amp.scale_loss(loss, optimizer, delay_overflow_check=True) as scaled_loss:
			scaled_loss.backward()
	
		
		#if(crossover_flag == True and group_size > 1):
		#	tensor_flatten = flatten_tensors_grad(list(amp.master_params(optimizer)))
		#	dist.reduce(tensor_flatten, dst=group_roots[mygroup], op=dist.ReduceOp.SUM, group=groups[mygroup])
		#	if(rank in group_roots):
		#		tensor_flatten = tensor_flatten / group_size
		#		tensor_unflatten = unflatten_tensors_grad(tensor_flatten, list(amp.master_params(optimizer)))	
		#		for param_model, unflat in zip(amp.master_params(optimizer), tensor_unflatten):
		#			param_model.grad.data = unflat							
		#	crossover(model,  rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, elitism=False, elite_ratio=elite_ratio, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)		
		#if((crossover_flag == False) and (allreduce == False) and (sync_grad == True)):
		#	sgp(model, model_cp, rank, world_size, epoch, itr, len_loader, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)
		torch.nn.utils.clip_grad_norm(amp.master_params(optimizer), 1000)					
		print('4')
		

		norms = []
		for tensor in amp.master_params(optimizer) :
			weight_norm = torch.norm(tensor.data).float().item()
			if tensor.grad is None or torch.isnan(tensor.grad).any() or torch.isinf(tensor.grad).any():
				grad_norm = 0.0
			else :
				grad_norm = torch.norm(tensor.grad.data).float().item()
			norm = [weight_norm, grad_norm]
			norms.append(norm)
		norms = torch.Tensor(norms).float().cuda()	
		optimizer.set_norms(norms) 						

##		
	##		
		#update_learning_rate(lrdecay, baselr, optimizer, maxepoch, epoch, itr, len(loader), world_size, batch_size, warmup_epoch)
		if(rank in group_roots):
			optimizer.step()
		optimizer.zero_grad()
		print('5')
		if(allreduce == True ):
			for i in range(len(layer_idx)):
				part_model = None	
				if(i==0):
					part_model = list(amp.master_params(optimizer))[0:layer_idx[i]]
				else :
					part_model = list(amp.master_params(optimizer))[layer_idx[i-1]:layer_idx[i]]
				tensor_flatten = flatten_tensors(part_model)
				dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
				tensor_flatten = tensor_flatten / world_size
				tensor_unflatten = unflatten_tensors(tensor_flatten, part_model)
				for param_model, unflat in zip(part_model, tensor_unflatten):
					param_model.data = unflat
			#tensor_flatten = flatten_tensors(list(amp.master_params(optimizer)))
			#dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
			#tensor_flatten = tensor_flatten / world_size
			#tensor_unflatten = unflatten_tensors(tensor_flatten, list(amp.master_params(optimizer)))	
			#for param_model, unflat in zip(amp.master_params(optimizer), tensor_unflatten):
			#	param_model.data = unflat
	##
		if(crossover_flag == True ):
			crossover(model, rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag, layer_idx=layer_idx, groups=groups, group_roots=group_roots, group_size=group_size, group_num=group_num, peer_groups=peer_groups, select_ranks=select_ranks, layer_numel=layer_numel)		
			#if(group_size > 1):
			#	mygroup = int(rank/(group_size))
			#	tensor_flatten = flatten_tensors_grad(list(amp.master_params(optimizer)))
			#	dist.broadcast(tensor=tensor_flatten.data, src=group_roots[mygroup], group=groups[mygroup])
			#	if(rank not in group_roots):	
			#		tensor_unflatten = unflatten_tensors(tensor_flatten, list(amp.master_params(optimizer)))	
			#		for param_model, unflat in zip(amp.master_params(optimizer), tensor_unflatten):
			#			param_model.data = unflat
	##
		if((crossover_flag == False) and (allreduce == False)):
			sgp(model, rank, world_size, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag, iter_num=itr_num, layer_idx=layer_idx, peer_groups=peer_groups, sgp_send=sgp_send, sgp_recv=sgp_recv)
		dist.barrier()
		torch.cuda.synchronize()
		elapsed_time = time.time()-batch_time
		print(elapsed_time)
		if(itr>0):
			itr_time[key_name].append(elapsed_time)
			#print(f'{key_name} {itr_time[key_name]}')	
		#dist.barrier()
		if(key_name != '' and len(itr_time[key_name]) == test_num):
			with open(f'itr_time_{world_size}_{rank}_{key_name}.csv', 'w', newline='') as f:
				writer = csv.writer(f)
				for i in range(test_num):
					writer.writerow([itr_time[key_name][i]])
		

def update_learning_rate(lrdecay, target_lr, optimizer, maxepoch, epoch, itr, itr_per_epoch, world_size, batch_size,
						warmup_epoch, scale=1, end_learning_rate=0.0025):
	target_lr = target_lr
	target_wd = optimizer.wd
	lr = 0
	wd = 0
	print(f"itr per epoch {itr_per_epoch}, itr {itr}, epoch {epoch}")
	if(epoch < warmup_epoch):
		count = epoch * itr_per_epoch + itr + 1
		incr = ((count / (warmup_epoch * itr_per_epoch)))
		#print(count / (5 * itr_per_epoch))
		lr = incr * target_lr
		wd = incr * target_wd
	elif(lrdecay == 'step'):
		if(epoch >= 5):
			lr = target_lr
		if(epoch >= 81):
			lr = target_lr * 0.1
		if(epoch >= 122 ):
			lr = target_lr * 0.1 * 0.1
	elif(lrdecay=='poly'):

		count =  float(epoch-warmup_epoch) * itr_per_epoch + itr +1
		if(epoch >= maxepoch -2):
			count = (maxepoch-warmup_epoch-2) * itr_per_epoch
		total = (maxepoch-warmup_epoch-2) * itr_per_epoch
		decay = (1.0-end_learning_rate) *((1 - (count / total)) ** 2.2) + end_learning_rate

		lr = target_lr * decay 
		wd = target_wd * decay 

	#print(itr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		param_group['weight_decay'] = wd



def init_model():
	model = models.resnet50()
	model.cuda()
	return model

def validate(model, val_loader, criterion):
	val_loss = 0
	correct = 0
	total = 0

	model.eval()
	with torch.no_grad():
		for i, (features, target) in enumerate(val_loader):
			target = target.cuda()
			#kl_target = torch.zeros(target.shape[0], 1000, device='cuda').scatter_(1, target.view(-1,1),1)
			features = features.cuda()
			output = model(features)
			del features
			loss = criterion(output, target)
			
			val_loss += loss.item()
			_, predicted = output.max(1)
			total += target.size(0)
			correct += predicted.eq(target).sum().item()
			del target
			del output

	return ((float)(val_loss))/((float)(len(val_loader))), 100.*correct/total		
def select_layer(rank, rseed_per_rank, world_size, roulettes):
	select_rank = []
	copy_roulettes = deepcopy(roulettes)
	for i in range(0,world_size):
		for j in select_rank :
				copy_roulettes[i][j] = 0
		roulette_sum = sum(copy_roulettes[i])
		copy_roulettes[i] = [ r/roulette_sum for r in copy_roulettes[i]]
		if((world_size-2 == i) and ((world_size-1) not in select_rank)):
			select_rank.append(world_size -1)
		else:
			select_rank.append(rseed_per_rank[0].choice(world_size, p=copy_roulettes[i]))
	return select_rank

def sgp( model, rank, world_size, amp=None, optimizer=None, sync_grad=False, amp_flag=False, iter_num=0, layer_idx=None, peer_groups=None, sgp_send=None, sgp_recv=None):
	worlds = dist.new_group()

	part_model = None	
	#if(i==0):
	#	part_model = list(amp.master_params(optimizer))[0:layer_idx[i]]
	#else :
	#	part_model = list(amp.master_params(optimizer))[layer_idx[i-1]:layer_idx[i]]
	#part_model = list(amp.master_params(optimizer))
	#iter_num = itr / local_itr

	#iter_num = epoch * max_iter + itr *len(layer_idx) + i
	#iter_num = epoch * max_iter + itr * 1 + i
	send_to = sgp_send[iter_num % len(sgp_send)]
	receive_from = sgp_recv[iter_num % len(sgp_recv)]		
	for i in range(len(layer_idx)):
		part_model = None	
		if(i==0):
			part_model = list(amp.master_params(optimizer))[0:layer_idx[i]]
		else :
			part_model = list(amp.master_params(optimizer))[layer_idx[i-1]:layer_idx[i]]
		tensor_flatten = flatten_tensors(part_model)

		#print(f"send {send_to} receive {receive_from}")
		if(send_to == receive_from):
			key = ''
			if(send_to < rank):
				key = f"{send_to}:{rank}"
			else:
				key = f"{rank}:{send_to}"
			dist.all_reduce(tensor=tensor_flatten, group=peer_groups[key])
			tensor_flatten = tensor_flatten / 2.0				
		else:	
			t = threading.Thread(target=send, args=(tensor_flatten.data, send_to))
			t.start()	
			tensor_flatten.data = torch.zeros_like(tensor_flatten.data).cuda()
			dist.recv(tensor=tensor_flatten.data, src=receive_from)
			t.join()
		torch.cuda.synchronize()
		tensor_unflatten = unflatten_tensors(tensor_flatten, part_model)
		
		for p, cp in zip(part_model, tensor_unflatten):
			p.data.copy_(0.5*(cp.data+p.data))



def crossover_comm(param, receive_from, send_to, rank, world_size, groups, group_roots, group_size, group_num, part_model, peer_groups):
	#dist.broadcast(tensor=param.data, src=select_model, async_op=False)
		
		#print("!!!!")
		if(send_to == receive_from):
			if(send_to < rank):
				key = f"{group_roots[send_to]}:{rank}"
			else:
				key = f"{rank}:{group_roots[send_to]}" 			
			dist.all_reduce(tensor=param.data, group=peer_groups[key])
			param.data = param.data / 2.0			
		else:
			t = threading.Thread(target=send, args=(param.data, send_to))
			t.start()			
			#completed_recv = dist.irecv(tensor=param.data, src=receive_from)
			param.data = torch.zeros_like(param.data).cuda()
			dist.recv(tensor=param.data, src=receive_from)
			t.join()
		#completed_send.wait()
		torch.cuda.synchronize()	
		tensor_unflatten = unflatten_tensors(param, part_model)
		for p, cp in zip(part_model, tensor_unflatten):
			p.data.copy_(0.5*(cp.data+p.data))



def crossover(model, rank, world_size, model_name, rseed_per_rank, roulettes=None, amp=None, optimizer=None, sync_grad=False, amp_flag=False, layer_idx=None, groups=None, group_roots=None, group_size=None, group_num=None, peer_groups=None, select_ranks=None, layer_numel=None):
	tensors = [] 
	send_flag = []
	recv_flag = []

	if(group_size > 1):
		#print(layer_numel)
		mygroup = int(rank/(group_size))
		dist.barrier(groups[mygroup])

		tensor_flatten = flatten_tensors(list(amp.master_params(optimizer)))
		dist.reduce(tensor=tensor_flatten.data, dst=group_roots[mygroup], group=groups[mygroup])
		tensor_flatten = tensor_flatten / group_size
		#param_reduced = param
		if(rank in group_roots):
			#whole_param_unflatten = unflatten_tensors(tensor_flatten, list(amp.master_params(optimizer)))
			seg_list = []
			for i in range(len(layer_idx)):
				part_model = None	
				if(i==0):
					part_model = tensor_flatten[0:layer_numel[i]]
				else :
					part_model = tensor_flatten[layer_numel[i-1]:layer_numel[i]]
				
				#param_flatten = flatten_tensors(part_model)
				param_flatten = part_model
				param_flatten_detach = param_flatten.clone().detach()
				
				receive_from = select_ranks[i][int(rank/group_size)]		 
				send_to = select_ranks[i].index(int(rank/group_size))

				if(group_roots[send_to] == group_roots[receive_from]):
					key = ''
					if(group_roots[send_to] < rank):
						key = f"{group_roots[send_to]}:{rank}"
					else:
						key = f"{rank}:{group_roots[send_to]}" 
					dist.all_reduce(tensor=param_flatten.data, group=peer_groups[key])
					param_flatten.data = param_flatten.data / 2.0				
				else:				
					t = threading.Thread(target=send, args=(param_flatten_detach.data, group_roots[send_to]))
					t.start()			
					#completed_recv = dist.irecv(tensor=param.data, src=receive_from)
					param_flatten.data = torch.zeros_like(param_flatten.data).cuda()
					dist.recv(tensor=param_flatten.data, src=group_roots[receive_from])	
					t.join()	
				#completed_send.wait()
				torch.cuda.synchronize()
				param_flatten.data.copy_(0.5*(param_flatten_detach.data+param_flatten.data))			
				seg_list.append(param_flatten)
			whole_param_flatten = flatten_tensors(seg_list)
		whole_param_flatten = tensor_flatten
		dist.barrier(groups[mygroup])

		dist.broadcast(tensor=whole_param_flatten.data,src=group_roots[mygroup], group=groups[mygroup])
		tensor_unflatten = unflatten_tensors(whole_param_flatten, list(amp.master_params(optimizer)))
				
		for p, cp in zip(list(amp.master_params(optimizer)), tensor_unflatten):
			p.data = cp
	else :
		tensor_flatten = flatten_tensors(list(amp.master_params(optimizer)))
		seg_list = []
		for i in range(len(layer_idx)):
			part_model = None	
			if(i==0):
				part_model = tensor_flatten[0:layer_numel[i]]
			else :
				part_model = tensor_flatten[layer_numel[i-1]:layer_numel[i]]
			#tensor_flatten = flatten_tensors(part_model)
			param_flatten = part_model

			receive_from = select_ranks[i][int(rank/group_size)]		 
			send_to = select_ranks[i].index(int(rank/group_size))
		
			if(send_to == receive_from):
				if(send_to < rank):
					key = f"{send_to}:{rank}"
				else:
					key = f"{rank}:{send_to}" 			
				dist.all_reduce(tensor=param_flatten.data, group=peer_groups[key])
				param_flatten.data = param_flatten.data / 2.0			
			else:
				t = threading.Thread(target=send, args=(param_flatten.data, send_to))
				t.start()			
				#completed_recv = dist.irecv(tensor=param.data, src=receive_from)
				param_flatten.data = torch.zeros_like(param_flatten.data).cuda()
				dist.recv(tensor=param_flatten.data, src=receive_from)
				t.join()
			#completed_send.wait()
			torch.cuda.synchronize()	
			seg_list.append(param_flatten)
		whole_param_flatten = flatten_tensors(seg_list)		
		whole_param_unflatten = unflatten_tensors(whole_param_flatten,  list(amp.master_params(optimizer)))
		for p, cp in zip(list(amp.master_params(optimizer)), whole_param_unflatten):
			p.data.copy_(0.5*(cp.data+p.data))

if __name__ == '__main__':
	os.environ['MASTER_ADDR'] = '172.31.21.183'
	os.environ['MASTER_PORT'] = '29500'
	main()	