import argparse
import copy
import os
import socket
import time
import random
import sys
import numpy as np
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
from experiment_utils import make_logger
from experiment_utils import Meter
from experiment_utils import get_tcp_interface_name
from gossip_module import GossipDataParallel
from gossip_module import DynamicBipartiteExponentialGraph as DBEGraph
from gossip_module import DynamicBipartiteLinearGraph as DBLGraph
from gossip_module import DynamicDirectedExponentialGraph as DDEGraph
from gossip_module import DynamicDirectedLinearGraph as DDLGraph
from gossip_module import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from gossip_module import RingGraph
from gossip_module import UniformMixing

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
from autoaugment import ImageNetPolicy
import utils
import folder2lmdb
GRAPH_TOPOLOGIES = {
    0: DDEGraph,    # Dynamic Directed Exponential
    1: DBEGraph,    # Dynamic Bipartite Exponential
    2: DDLGraph,    # Dynamic Directed Linear
    3: DBLGraph,    # Dynamic Bipartite Linear
    4: RingGraph,   # Ring
    5: NPDDEGraph,  # N-Peer Dynamic Directed Exponential
    -1: None,
}

MIXING_STRATEGIES = {
    0: UniformMixing,  # assign weights uniformly
    -1: None,
}
parser = argparse.ArgumentParser(description='Playground')

parser.add_argument('--batch_size', default=64, type=int,
					help='per-agent batch size')

parser.add_argument('--world_size', default=4, type=int)
parser.add_argument('--gpu_per_node', default=2, type=int)
parser.add_argument('--proc_per_gpu', default=1, type=int)
parser.add_argument('--local_itr', default=1, type=int)
parser.add_argument('--warmup_epoch', default=12, type=int)

parser.add_argument('--crossover', default='True', type=str)
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

# Lighting data augmentation take from here - https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))
class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

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
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def find_block(model_name, name):	
	return [model_name.index(n) for n in model_name if(n in name)]

def main():
	args = parser.parse_args()
	world_size = args.world_size
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

	rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
	if(rank == 0 ):
		f = open("/scratch/x2026a02/imagenet_test.py", mode='r')
		for line in f:
			print(line, end ="")
		f.close()
		f = open("/scratch/x2026a02/lars.py", mode='r')
		for line in f:
			print(line, end ="")
		f.close()	
	GPU_NUM = rank  % gpu_per_node
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

	dist.init_process_group(backend="mpi", world_size =world_size, rank=rank)

	model = init_model()

	#make sync_lars_group
	groups = [] 
	for x in range(int(world_size / sync_lars_group_size)):
		group = []
		for y in range(sync_lars_group_size):
			group.append(y+ x*sync_lars_group_size)
		new_group = dist.new_group(ranks=group)
		groups.append(new_group)

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
	for idx, (n, param) in enumerate(model.named_parameters()):
		print(f"parameter_names {n}")
		if(n == 'conv1.weight' or n == 'bn1.weight' or n == 'bn1.bias'):
			old_block = 0
		else:
			is_changed =  False if (old_block == find_block(model_name, n)[0]) else True
			if is_changed :
				layer_idx.append(idx)
				print(n)
			old_block = find_block(model_name, n)[0] if(is_changed) else old_block
	layer_idx.append(len(list(model.parameters())))		
	print(layer_idx)		

	log_softmax = nn.LogSoftmax(dim=1)
	
	loader, sampler = make_dataloader(rank, batch_size, world_size)
	val_loader = make_validation_dataloader( batch_size)

	#criterion = nn.CrossEntropyLoss()
	criterion = LabelSmoothingLoss(1000)
	
	#recursive_batch_norm_momentum(model, momentum=1.0)
	
	optimizer = torch.optim.SGD(model.parameters() , lr=baselr, momentum=0.9, nesterov=True)
	if(lars == True):
		optimizer = LARS(model, model.parameters(), lr=baselr, momentum=0.96, weight_decay=weight_decay, eta=lars_coef, max_epoch=maxepoch, dist=dist, world_size=world_size, amp=amp, rank=rank)
	if(amp_flag == True):
		model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
	optimizer.zero_grad()
	_train_loader = loader.__iter__()

	
	roulettes = [] * world_size

	tensor_each_rank = []
	for i in range(world_size):
		tensor_each_rank.append(list(model.parameters())[rank::world_size])
	norm_list = [[0.0, 0.0]] * len(list(model.parameters()))
	norm_tensor = torch.Tensor(norm_list).float().cuda()
	norm_each_rank = []
	for i in range(world_size):
		norm_each_rank.append(torch.Tensor(norm_list[i::world_size]).float().cuda())

	for i in range(0, world_size):
		roulette_except_rank = [1./(world_size -1) for i in range(0, world_size)]
		roulette_except_rank[i] = 0
		roulettes.append(roulette_except_rank)
	iter_num = 0
	global_itr = 0
	for epoch in range(0, maxepoch):
		batch_time = time.time()
		#final polishing
		#if(epoch == 85):	
		#	loader, sampler = make_dataloader_no_data_aug(rank, batch_size, world_size)
		sampler.set_epoch(epoch + 1 * 90)

		_train_loader = loader.__iter__()

		#update_learning_rate(lrdecay, baselr, optimizer, maxepoch, epoch, 0, len(loader), world_size, batch_size, warmup_epoch)
		len_loader = len(_train_loader)
		for itr,(batch, target) in enumerate(_train_loader, start=0):
			if(itr == 10):
				break
			global_itr = global_itr + 1
			minibatch_time = time.time()
			model.train()

			target = target.cuda(non_blocking=True)

			batch = batch.cuda(non_blocking=True)

			r = np.random.rand(1)
			loss = 0
			#if(args.beta > 0 and r< args.cutmix_prob and epoch < 85):
			#	# generate mixed sample
			#	lam = np.random.beta(args.beta, args.beta)
			#	rand_index = torch.randperm(batch.size()[0]).cuda()
			#	target_a = target
			#	target_b = target[rand_index]
			#	bbx1, bby1, bbx2, bby2 = rand_bbox(batch.size(), lam)
			#	batch[:, :, bbx1:bbx2, bby1:bby2] = batch[rand_index, :, bbx1:bbx2, bby1:bby2]
			#	# adjust lambda to exactly match pixel ratio
			#	lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
			#	# compute output
			#	output = model(batch)
			#	loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)			
			#else:
			#	output = model(batch)
			#	loss = criterion(output, target)

			output = model(batch)
			loss = criterion(output, target)
			
			#No-BN weigth decay version 2
			#reg_loss = 0
			#for name,param in model.named_parameters():
			#	if 'bn' not in name and 'bias' not in name:
			#		reg_loss += torch.norm(param)
			#loss = cls_loss + (2.56*10**-4)*reg_loss #Manual add weight decay
			
			#if(itr > len(loader)-((len(loader))% local_itr)):
			#	loss = loss / ((len(loader))% local_itr)
			#else :	
			#	loss = loss /local_itr

			loss = loss /local_itr

			#recursive_batch_norm_sync(model, world_size, dist)
			
			if(amp_flag==True):
				with amp.scale_loss(loss, optimizer, delay_overflow_check=True) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()
	
			#if((itr %  local_itr == local_itr -1) or len(loader)-1 == itr):
			if(global_itr % local_itr == 0):
				#apply_weight_decay(model, weight_decay_factor=5e-4, wo_bn=True)
				if(amp_flag==True):
					if(allreduce == True and sync_grad == True):
						tensor_flatten = flatten_tensors_grad(list(amp.master_params(optimizer)))
						dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
						tensor_flatten = tensor_flatten / world_size
						tensor_unflatten = unflatten_tensors_grad(tensor_flatten, list(amp.master_params(optimizer)))	
						for param_model, unflat in zip(amp.master_params(optimizer), tensor_unflatten):
							param_model.grad.data = unflat					
					if(crossover_flag == True and sync_grad == True):
						crossover(model,  rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, elitism=False, elite_ratio=elite_ratio, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)		
					#if((crossover_flag == False) and (allreduce == False) and (sync_grad == True)):
					#	sgp(model, model_cp, rank, world_size, epoch, itr, len_loader, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)
					if(clip_grad == True):
						torch.nn.utils.clip_grad_norm(amp.master_params(optimizer), 1000)					
					
	##
					if(lars == True and sync_lars_start_epoch <= epoch):
						placeholder = None
						placeholder_grad = None
						tensor_mine = None
						grad_mine = None
						#print(list(model.parameters())[0].grad.data)
						wait_list = []
						#for i in range(world_size):
						#	tensor_flatten = flatten_tensors(list(amp.master_params(optimizer))[i::world_size])
						#	grad_flatten = flatten_tensors_grad(list(amp.master_params(optimizer))[i::world_size])
						#
						#	if(rank == i):
						#		placeholder = tensor_flatten.clone()
						#		placeholder_grad = grad_flatten.clone()
						#		wait_reduce_tensor = dist.reduce(placeholder, dst=i, op=dist.ReduceOp.SUM, async_op=True)
						#		wait_reduce_grad = dist.reduce(placeholder_grad, dst=i, op=dist.ReduceOp.SUM, async_op=True)
						#		print(f"rank {rank}, dst {i} tensor_flatten {tensor_flatten}")
						#	else :
						#		wait_reduce_tensor = dist.reduce(tensor_flatten, dst=i, op=dist.ReduceOp.SUM, async_op=True)
						#		wait_reduce_grad = dist.reduce(grad_flatten, dst=i, op=dist.ReduceOp.SUM, async_op=True)
						#		print(f"rank {rank}, dst {i} tensor_flatten {tensor_flatten}")
						#	wait_list.append([wait_reduce_tensor, wait_reduce_grad])
#	##
						#for i, wait_reduce in enumerate(wait_list) :
						#	wait_reduce[0].wait()
						#	wait_reduce[1].wait()
						#	if(i==rank):
						#		tensor_mine = placeholder
						#		grad_mine = placeholder_grad
						#print(f"rank {rank},  placeholder {placeholder}")
						##print(placeholder)
						#tensor_unflatten = unflatten_tensors(tensor_mine, list(amp.master_params(optimizer))[rank::world_size])
						#grad_unflatten = unflatten_tensors_grad(grad_mine, list(amp.master_params(optimizer))[rank::world_size])
						#
						#norms = []
						#for tensor, grad in zip(tensor_unflatten, grad_unflatten) :
						#	weight_norm = torch.norm(tensor.data).float().item()
						#	grad_norm = torch.norm(grad.data).float().item()
						#	norm = [weight_norm, grad_norm]
						#	norms.append(norm)
						#print(f"rank : {rank}, norms {norms}")
						#wait_list = []
						#for i in range(world_size):
						#	if(i == rank):
						#		norms = torch.Tensor(norms).float().cuda()
						#		norm_each_rank[i] = torch.div(norms, world_size)
						#		print(f"src {rank},  div {norm_each_rank[i]}")
#	##
						#	wait_broadcast = dist.broadcast(norm_each_rank[i], src=i, async_op=True)	
						#	wait_list.append(wait_broadcast)
						#	
						#for i, wait_bcast in enumerate(wait_list) :
						#	wait_bcast.wait()
						#	norm_tensor[i::world_size] = norm_each_rank[i]
						#print(f"rank {rank},  div {norm_each_rank[5]}")
						#optimizer.set_norms(norm_tensor) 	
	##
						norms = []
						for param in amp.master_params(optimizer) :
							param_clone = param.clone()
							grad_clone = param.grad.clone()
	##
							if grad_clone is None or torch.isnan(grad_clone).any() or torch.isinf(grad_clone).any():
								grad_clone.fill_(0)
	##
							dist.all_reduce(param_clone.data, op=dist.ReduceOp.SUM, group=groups[int(rank/sync_lars_group_size)])
							dist.all_reduce(grad_clone.data, op=dist.ReduceOp.SUM, group=groups[int(rank/sync_lars_group_size)])
	##
							weight_norm = torch.norm(param_clone.data).float().item()
							grad_norm = torch.norm(grad_clone.data).float().item()
							norms.append([weight_norm, grad_norm])
	##
						norms = torch.Tensor(norms).float().cuda()
						optimizer.set_norms(norms) 	
	##
					if(lars == True and sync_lars_start_epoch > epoch):						 					
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
				else:
					if(allreduce == True and sync_grad == True):
						tensor_flatten = flatten_tensors_grad(list(model.parameters()))
						dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
						tensor_flatten = tensor_flatten / world_size
						tensor_unflatten = unflatten_tensors(tensor_flatten, list(model.parameters()))	
						for param_model, unflat in zip(model.parameters(), tensor_unflatten):
							param_model.grad.data = unflat
	##
					if(crossover_flag == True and sync_grad == True):
						crossover(model, rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)		
					
			
					if((crossover_flag == False) and (allreduce == False) and (sync_grad == True)):
						sgp(local_itr, model, rank, world_size, epoch, itr, len_loader, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)
	##
					if(clip_grad == True):
						torch.nn.utils.clip_grad_norm(model.parameters(), 1000)
	##
					if(lars == True and sync_lars_start_epoch <= epoch):				
						tensor_mine = None
						grad_mine = None
						#print(list(model.parameters())[0].grad.data)
						for i in range(world_size):
							tensor_flatten = flatten_tensors(list(model.parameters())[i::world_size])
							grad_flatten = flatten_tensors_grad(list(model.parameters())[i::world_size])
			
							dist.reduce(tensor_flatten, dst=i, op=dist.ReduceOp.SUM, async_op=False)
							dist.reduce(grad_flatten, dst=i, op=dist.ReduceOp.SUM, async_op=False)
			
							if(i==rank):
								tensor_mine = tensor_flatten
								grad_mine = grad_flatten
			
			
						tensor_unflatten = unflatten_tensors(tensor_mine, list(model.parameters())[rank::world_size])
						grad_unflatten = unflatten_tensors(grad_mine, list(model.parameters())[rank::world_size])
			
						norms = []
						for tensor, grad in zip(tensor_unflatten, grad_unflatten) :
							weight_norm = torch.norm(tensor.data).float().item()
							grad_norm = torch.norm(grad.data).float().item()
							if torch.isnan(grad).any() or torch.isinf(grad).any():
								grad_norm = 0.0
							norm = [weight_norm, grad_norm]
							norms.append(norm)
			
						for i in range(world_size):
							if(i == rank):
								norms = torch.Tensor(norms).float().cuda()
								norm_each_rank[i] = norms
							dist.broadcast(norm_each_rank[i], src=i, async_op=False)	
			
							norm_tensor[i::world_size] = norm_each_rank[i]
						optimizer.set_norms(norm_tensor)
					if(lars == True and sync_lars_start_epoch > epoch):						 					
						norms = []
						for tensor in model.parameters() :
							weight_norm = torch.norm(tensor.data).float().item()
							grad_norm = torch.norm(tensor.grad.data).float().item()
							norm = [weight_norm, grad_norm]
							norms.append(norm)
						norms = torch.Tensor(norms).float().cuda()	
						optimizer.set_norms(norms)
##
	##		
				update_learning_rate(lrdecay, baselr, optimizer, maxepoch, epoch, itr, len(loader), world_size, batch_size, warmup_epoch)
				optimizer.step()
				optimizer.zero_grad()
	##
	##
	##
				#if((crossover_flag == False) and (allreduce == False)):
				#	model.transfer_params()
				#	model.gossip_flag.wait(timeout=300)
				if(allreduce == True and sync_grad == False):
					#for param in model.parameters():
					#	dist.all_reduce(param.data, op=dist.ReduceOp.SUM, async_op=False)
					#	param.data = param.data *(1.0/world_size)
					tensor_flatten = flatten_tensors(list(amp.master_params(optimizer)))
					dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
					tensor_flatten = tensor_flatten / world_size
					tensor_unflatten = unflatten_tensors(tensor_flatten, list(amp.master_params(optimizer)))	
					for param_model, unflat in zip(amp.master_params(optimizer), tensor_unflatten):
						param_model.data = unflat
	##
				if(crossover_flag == True and sync_grad == False):
					crossover(model, rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag, layer_idx=layer_idx)		
	##
				if((crossover_flag == False) and (allreduce == False) and (sync_grad == False)):
					sgp(local_itr, model, rank, world_size, epoch, itr, len_loader, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag, iter_num=iter_num)
				iter_num = iter_num +1

			#optimizer.step()
			#optimizer.zero_grad()
				print(time.time() - minibatch_time)	
				minibatch_data_load_itme = time.time()
		elapsed_time = time.time()-batch_time
		if(rank == 0):
			for param_group in optimizer.param_groups:
				print(param_group['lr'])
		if((epoch+1)%val_iter == 0):
			losses,top1 = validate(model, val_loader, criterion)
	
			with open(f'/scratch/x2026a02/{args.tag}_{file_prefix}_{world_size}_{batch_size}_{local_itr}_{rank}_sync_lars_start_at_{sync_lars_start_epoch}_sync_lars_group_size_{sync_lars_group_size}_amp_{amp_flag}_clip_grad_{clip_grad}_baselr_{baselr}_maxepoch_{maxepoch}_lrdecay_{lrdecay}_lars_{lars}_lars_coef_{lars_coef}_chromo_{args.chromosome}_'+'val.csv', '+a') as f:
				print('{ep}, {rank}, '
					'{loss:.4f},'
					'{top1:.3f},'
					'{val}, {elapsed_time}'
					.format(ep=epoch,rank=rank, loss=losses, top1=top1, val=top1, elapsed_time=elapsed_time),
					file=f
					)

	losses,top1 = validate(model, val_loader, criterion)
	
	with open(f'/scratch/x2026a02/{args.tag}_{file_prefix}_{world_size}_{batch_size}_{local_itr}_{rank}_sync_lars_start_at_{sync_lars_start_epoch}_sync_lars_group_size_{sync_lars_group_size}_amp_{amp_flag}_clip_grad_{clip_grad}_baselr_{baselr}_maxepoch_{maxepoch}_lrdecay_{lrdecay}_lars_{lars}_lars_coef_{lars_coef}_chromo_{args.chromosome}_'+'val.csv', '+a') as f:
		print('{ep}, {rank}, '
			'{loss:.4f},'
			'{top1:.3f},'
			'{val}, {elapsed_time}'
			.format(ep=epoch,rank=rank, loss=losses, top1=top1, val=top1, elapsed_time=elapsed_time),
			file=f
			)

def rand_bbox(size, lam):
	W = size[2]
	H = size[3]
	cut_rat = np.sqrt(1. - lam)
	cut_w = np.int(W * cut_rat)
	cut_h = np.int(H * cut_rat)

	# uniform
	cx = np.random.randint(W)
	cy = np.random.randint(H)

	bbx1 = np.clip(cx - cut_w // 2, 0, W)
	bby1 = np.clip(cy - cut_h // 2, 0, H)
	bbx2 = np.clip(cx + cut_w // 2, 0, W)
	bby2 = np.clip(cy + cut_h // 2, 0, H)

	return bbx1, bby1, bbx2, bby2
#label smoothing *reference  https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py#L38
def cal_loss(pred, gold, trg_pad_idx, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
    	#reference https://pytorch.org/docs/stable/nn.functional.html
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def recursive_batch_norm_momentum(child, momentum=None):
    
    if type(child) == torch.nn.BatchNorm2d:
        print(child.momentum)
        child.momentum = momentum
        return
    
    for children in child.children():
        lowest_child = recursive_batch_norm_momentum(children, momentum=momentum)
    
    return

def recursive_batch_norm_sync(child, world_size, dist):
	
	if type(child) == torch.nn.BatchNorm2d:
		#print(child.momentum)
		#print(child.running_mean)
		dist.all_reduce(child.running_mean, op=dist.ReduceOp.SUM)
		dist.all_reduce(child.running_var, op=dist.ReduceOp.SUM)
		child.running_mean = child.running_mean / world_size
		child.running_var = child.running_var / world_size
		#print(child.running_var)
		return
	
	for children in child.children():
		lowest_child = recursive_batch_norm_sync(children, world_size, dist)


def add_weight_decay(net, l2_value, skip_list=()):
	decay, no_decay = [], []
	for name, param in net.named_parameters():
		if not param.requires_grad: continue # frozen weights		            
		if len(param.shape) == 1 or name.endswith(".bias") or ('bn' in name) or name in skip_list: 
			no_decay.append(param)
		else: decay.append(param)
	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]
def apply_weight_decay(*modules, weight_decay_factor=0., wo_bn=True):
    '''
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/5
    Apply weight decay to pytorch model without BN;
    In pytorch:
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
    p is the param;
    :param modules:
    :param weight_decay_factor:
    :return:
    '''
    for module in modules:
        for m in module.modules():
        	if hasattr(m, 'weight'):
        		if wo_bn and isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        			#print("batch_norm")
        			continue
        		m.weight.grad += m.weight * weight_decay_factor
        		
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

def make_dataloader(rank, batch_size, world_size):
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	jittering = utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
	lighting = utils.Lighting(alphastd=0.1,
								eigval=[0.2175, 0.0188, 0.0045],
								eigvec=[[-0.5675, 0.7192, 0.4009],
										[-0.5808, -0.0045, -0.8140],
										[-0.5836, -0.6948, 0.4203]])

	#train_dataset = torchvision.datasets.ImageNet(root='/scratch/x2026a02/', split='train', download=False, transform=transforms.Compose([
	#		transforms.RandomResizedCrop(224, scale=(0.05, 0.9), ratio=(0.666667, 1.5)),
	#		transforms.RandomHorizontalFlip(),
	#		#ImageNetPolicy(),
	#		transforms.ToTensor(),
	#		#jittering,
	#		#lighting,
	#		normalize,
	#	]))

	train_dataset = folder2lmdb.ImageFolderLMDB('/scratch/x2026a02/train_lmdb',transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.05, 0.9), ratio=(0.666667, 1.5)),
			transforms.RandomHorizontalFlip(),
			#ImageNetPolicy(),
			transforms.ToTensor(),
			#jittering,
			#lighting,
			normalize,
		]))
	train_sampler = torch.utils.data.distributed.DistributedSampler(
						shuffle=True,
						dataset=train_dataset,
						num_replicas=world_size,
						rank=rank)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler)
	return train_loader, train_sampler


def make_dataloader_no_data_aug(rank, batch_size, world_size):
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_dataset = torchvision.datasets.ImageNet(root='/scratch/x2026a02/', split='train', download=False, transform=transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.999, 1.0), ratio=(0.999, 1.001)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))
	train_sampler = torch.utils.data.distributed.DistributedSampler(
						shuffle=True,
						dataset=train_dataset,
						num_replicas=world_size,
						rank=rank)
	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, sampler=train_sampler)
	return train_loader, train_sampler


def make_validation_dataloader(batch_size):
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	torch.cuda.manual_seed(r)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	#val_set = torchvision.datasets.ImageNet(root='/scratch/x2026a02/', split='val', download=False, transform=transforms.Compose([
	#		transforms.Resize(256),
	#		transforms.CenterCrop(224),
	#		transforms.ToTensor(),
	#		normalize,
	#	]))
	val_set = folder2lmdb.ImageFolderLMDB('/scratch/x2026a02/val_lmdb',transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]))

	val_loader = torch.utils.data.DataLoader(
		val_set,
		batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)
	
	return val_loader


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
			target = target.cuda(non_blocking=True)
			#kl_target = torch.zeros(target.shape[0], 1000, device='cuda').scatter_(1, target.view(-1,1),1)
			features = features.cuda(non_blocking=True)
			output = model(features)
			loss = criterion(output, target)
			val_loss += loss.item()
			_, predicted = output.max(1)
			total += target.size(0)
			correct += predicted.eq(target).sum().item()

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

def sgp(local_itr, model, rank, world_size, epoch, itr, max_iter, amp=None, optimizer=None, sync_grad=False, amp_flag=False, iter_num=0):
	#params = model.parameters()
	#if(amp_flag == True):
	#	params = amp.master_params(optimizer)

	part_model = None	
	#if(i==0):
	#	part_model = list(amp.master_params(optimizer))[0:layer_idx[i]]
	#else :
	#	part_model = list(amp.master_params(optimizer))[layer_idx[i-1]:layer_idx[i]]
	part_model = list(amp.master_params(optimizer))
	#iter_num = itr / local_itr

	#iter_num = epoch * max_iter + itr *len(layer_idx) + i
	#iter_num = epoch * max_iter + itr * 1 + i
	send_to = int((rank+(2**(iter_num%(math.log(world_size,2)))))%world_size)
	receive_from = int((rank-(2**(iter_num%(math.log(world_size,2)))))%world_size)
	
	tensor_flatten = flatten_tensors(part_model)

	
	completed = dist.isend(tensor=tensor_flatten.data, dst=send_to)
	dist.recv(tensor=tensor_flatten.data, src=receive_from)
	completed.wait()

	tensor_unflatten = unflatten_tensors(tensor_flatten, part_model)
	
	for p, cp in zip(part_model, tensor_unflatten):
		p.data.copy_(0.5*(cp.data+p.data))
	





def crossover_comm(param, receive_from, send_to, select_rank, rank, world_size):
	#dist.broadcast(tensor=param.data, src=select_model, async_op=False)
	completed = dist.isend(tensor=param.data, dst=send_to)
	dist.recv(tensor=param.data, src=receive_from)
	completed.wait()



def crossover(model, rank, world_size, model_name, rseed_per_rank, roulettes=None, amp=None, optimizer=None, sync_grad=False, amp_flag=False, layer_idx=None):

	for i in range(len(layer_idx)):
		part_model = None	
		if(i==0):
			part_model = list(amp.master_params(optimizer))[0:layer_idx[i]]
		else :
			part_model = list(amp.master_params(optimizer))[layer_idx[i-1]:layer_idx[i]]

		tensor_flatten = flatten_tensors(part_model)
		select_rank = select_layer(rank, rseed_per_rank, world_size, roulettes)
		receive_from = select_rank[rank]		 
		send_to = select_rank.index(rank)
		crossover_comm(tensor_flatten, receive_from, send_to, select_rank, rank, world_size)
		tensor_unflatten = unflatten_tensors(tensor_flatten, part_model)
		
		for p, cp in zip(part_model, tensor_unflatten):
			p.data.copy_(0.5*(cp.data+p.data))



if __name__ == '__main__':

	main()	