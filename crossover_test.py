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
#from scheduler import GradualWarmupScheduler, PolynomialLRDecay
#from optimizer import SGD_without_lars, SGD_with_lars, SGD_with_lars_ver2
import math
from models import *
from gossip_module.utils import flatten_tensors, flatten_tensors_grad, unflatten_tensors, unflatten_tensors_grad
from copy import copy, deepcopy
#from LARC import LARC
import apex.amp as amp

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
#parser.add_argument('--all_reduce', default='False', type=str,
#                    help='whether to use all-reduce or gossip')
parser.add_argument('--batch_size', default=64, type=int,
					help='per-agent batch size')
#parser.add_argument('--lr', default=0.1, type=float,
#                    help='reference learning rate (for 256 sample batch-size)')
#parser.add_argument('--num_epochs', default=160, type=int,
#                    help='number of epochs to train')
#
#parser.add_argument('--push_sum', default='True', type=str,
#                    help='whether to use push-sum or push-pull gossip')
#parser.add_argument('--graph_type', default=5, type=int,
#                    choices=GRAPH_TOPOLOGIES,
#                    help='the graph topology to use for gossip'
#                         'cf. the gossip_module graph_manager for available'
#                         'graph topologies and their corresponding int-id')
#parser.add_argument('--mixing_strategy', default=0, type=int,
#                    choices=MIXING_STRATEGIES,
#                    help='the mixing strategy to use for gossip'
#                         'cf. the gossip_module mixing_manager for available'
#                         'mixing strategies and their corresponding int-id.')
#parser.add_argument('--schedule', nargs='+', default='30 0.1 60 0.1 80 0.1',
#                    type=float, help='learning rate schedule')
#parser.add_argument('--momentum', default=0.9, type=float,
#                    help='optimization momentum')
#parser.add_argument('--weight_decay', default=5e-4, type=float,
#                    help='regularization applied to non batch-norm weights')
#parser.add_argument('--nesterov', default='False', type=str,
#                    help='whether to use nesterov style momentum'
#                         'otherwise will use regular Polyak momentum')
#parser.add_argument('--elitism', default='False', type=str)
#parser.add_argument('--elite_ratio', default=0.25, type=float)
#parser.add_argument('--parent_num', default=4, type=int)
parser.add_argument('--world_size', default=4, type=int)
#parser.add_argument('--selection_method', default=None, type=str)
#parser.add_argument('--fitness_roulette', default='True', type=str)
#parser.add_argument('--evaluate_batches', default=10000, type=int)
parser.add_argument('--parent_num', default=4, type=int)
parser.add_argument('--elitism', default='False', type=str)
parser.add_argument('--elite_ratio', default=0.25, type=float)

parser.add_argument('--crossover', default='True', type=str)
parser.add_argument('--ga', default='True', type=str)
parser.add_argument('--chromosome', default='coarse', type=str)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--lars', default='False', type=str)
parser.add_argument('--allreduce', default='False', type=str)
parser.add_argument('--sync_grad', default='False', type=str)
parser.add_argument('--sync_lars', default='False', type=str)
parser.add_argument('--amp', default='False', type=str)
parser.add_argument('--clip_grad', default='False', type=str)
parser.add_argument('--lrdecay', default='step', type=str)
parser.add_argument('--lars_coef', default=0.001, type=float)
parser.add_argument('--baselr', default=1.6, type=float)
parser.add_argument('--maxepoch', default=160, type=int)

def find_block(model_name, name):	
	return [model_name.index(n) for n in model_name if(n in name)]

def main():
	args = parser.parse_args()
	world_size = args.world_size
	batch_size = args.batch_size
	parent_num = args.parent_num
	lars_coef = args.lars_coef
	baselr = args.baselr
	maxepoch = args.maxepoch
	lrdecay = args.lrdecay
	sync_lars = False
	if(args.sync_lars == 'True'):
		sync_lars = True		
	amp_flag = False
	if(args.amp == 'True'):
		amp_flag = True
	clip_grad = False
	if(args.clip_grad == 'True'):
		clip_grad = True			
	sync_grad = False
	if(args.sync_grad == 'True'):
		sync_grad = True	

	if(args.elitism == 'True'):
		elitism = True
	else:
		elitism = False
	ga_flag = True
	if(args.ga=='True'):
		ga_flag = True
	else:
		ga_flag = False
	lars = False
	if(args.lars=='True'):
		lars = True	
	allreduce = False
	if(args.allreduce == 'True'):
		allreduce = True
	elite_ratio = args.elite_ratio
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

	GPU_NUM = rank  % 2
	device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(device) # change allocation of current GPU
	print ('Current cuda device ', torch.cuda.current_device()) # check

	# Additional Infos
	if device.type == 'cuda':
	    print(torch.cuda.get_device_name(GPU_NUM))
	    print('Memory Usage:')
	    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
	    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

	dist.init_process_group(backend="mpi", world_size =world_size, rank=rank)

	model = init_model()
	ppi_schedule = {}
	peers_per_itr_schedule = [0, 1]
	i, epoch = 0, None
	for v in peers_per_itr_schedule:
		if i == 0:
			epoch = v
		elif i == 1:
			ppi_schedule[epoch] = v
		i = (i + 1) % 2

	graph =None
	mixing = None
	graph_class = GRAPH_TOPOLOGIES[0]
	if graph_class:
		# dist.barrier is done here to ensure the NCCL communicator is created
		# here. This prevents an error which may be caused if the NCCL
		# communicator is created at a time gap of more than 5 minutes in
		# different processes
		dist.barrier()
		graph = graph_class(
			rank, world_size, peers_per_itr=ppi_schedule[0])

	mixing_class = MIXING_STRATEGIES[0]
	if mixing_class and graph:
		mixing = mixing_class(graph, torch.device('cuda'))
	model_cp = models.resnet18().cuda()

	#if(allreduce == True):
	#	model = torch.nn.parallel.DistributedDataParallel(model)
	#else:
	#model = GossipDataParallel(model, graph=graph, mixing=mixing, push_sum=True,rank=rank, overlap=False, synch_freq=0, verbose=False, use_streams=False)
	torch.manual_seed(1)
	def init_weights(m):
		if type(m) == nn.Linear or type(m) == nn.Conv2d:
			torch.nn.init.kaiming_uniform_(m.weight)
	#model.module.apply(init_weights)
	model.apply(init_weights)
	model_cp.apply(init_weights)		
	crossover_params = []
	#for n, p in model.module.named_parameters():
	#	cp = p.clone().detach_()
	#	cp = cp.cuda()
	#	crossover_params.append([n, cp])
	#coarse
	if(args.chromosome == 'coarse'):
		model_name = ['-1','layer1',  'layer2', 'layer3',  'layer4', 'fc']
	elif(args.chromosome == 'fine'):
		#model_name = ['-1','layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1','fc']
		model_name = ['-1','layer1.0', 'layer1.1',  'layer1.2','layer2.0', 'layer2.1', 'layer2.2', 'layer2.3', 'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4.0', 'layer4.1', 'layer4.2','fc']

	#print(find_block(model_name, 'layer1.0.1'))
	#if(rank == 0):
	#	print("==========================================")
	#	i = 0
	#	for n, p in model_cp.named_parameters():
	#		if(p.requires_grad):
	#			print(n)
	#			i = i +1
	#	print(i)
	#	print(len(list(model.named_parameters())))
	#	print(len(list(model.parameters())))
	#core_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
	log_softmax = nn.LogSoftmax(dim=1)
	
	loader, sampler = make_dataloader(rank, batch_size, world_size)
	val_loader = make_validation_dataloader(rank, batch_size)
	fitness_loader = make_fitness_dataloader()

	criterion = nn.CrossEntropyLoss()

	#def criterion(input, kl_target):
	#	assert kl_target.dtype != torch.int64
	#	loss = core_criterion(log_softmax(input), kl_target)
	#	return loss
	optimizer = torch.optim.SGD(model.parameters(), lr=baselr, momentum=0.9,weight_decay=5e-4, nesterov=True)
	if(lars == True):
		#optimizer = LARS(optimizer=optimizer, eps=5e-3, trust_coef=lars_coef)
		#optimizer = SGD_with_lars_ver2(model.parameters(), lr=0.16, momentum=0.9, weight_decay=5e-4, trust_coef=lars_coef, adaptive_lars=adaptive_lars)
		#optimizer = LARC(optimizer, trust_coefficient=lars_coef, eps=5e-4, adaptive_lars=adaptive_lars)
		optimizer = LARS(model.parameters(), lr=baselr, momentum=0.9, weight_decay=5e-4, eta=lars_coef, max_epoch=maxepoch, dist=dist, world_size=world_size, amp=amp, rank=rank)
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

	for epoch in range(0, maxepoch):
		batch_time = time.time()	
		sampler.set_epoch(epoch + 1 * 90)
		#update_peers_per_itr(model, ppi_schedule, epoch)
		#model.block()
		_train_loader = loader.__iter__()
		#if(args.lrdecay == 'poly'):
		#	if epoch <= 5:
		#		warmup_scheduler.step()
		#	if epoch > 5:
		#		poly_decay_scheduler.base_lr = warmup_scheduler.get_lr()
		#if(epoch > 80):
		#	sync_grad = True
		#	sync_lars = False
		update_learning_rate(lrdecay, baselr, optimizer, maxepoch, epoch, 0, len(loader), world_size, batch_size)
		len_loader = len(_train_loader)
		for itr,(batch, target) in enumerate(_train_loader, start=0):
			#if(args.lrdecay == 'poly'):
			#	if epoch > 5:
			#		poly_decay_scheduler.step()
			#if(crossover_flag == True):
			model.train()
			#else:
			#	model.train()
			target = target.cuda(non_blocking=True)
			#kl_target = torch.zeros(target.shape[0], 1000, device='cuda').scatter_(
			#	1, target.view(-1, 1), 1)
			batch = batch.cuda(non_blocking=True)
			#if(crossover_flag == True or allreduce == True):
			output = model(batch)
			#else:
			#	output = model(batch)
			loss = criterion(output, target)
			if(amp_flag==True):
				with amp.scale_loss(loss, optimizer, delay_overflow_check=True) as scaled_loss:
					scaled_loss.backward()

	
				if(allreduce == True and sync_grad == True):
					#old
					#for param in amp.master_params(optimizer):
					#	grad = param.grad.clone()
					#	if param.grad is None or torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
					#		grad = grad.fill_(0)
					#		dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=False)
					#		#continue
					#	else:
					#		dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=False)
					#	#dist.all_reduce(grad.data, op=dist.ReduceOp.SUM, async_op=False)
					#	param.grad.data = grad.data *(1.0/world_size)

					#new
					tensor_flatten = flatten_tensors_grad(list(amp.master_params(optimizer)))
					dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
					tensor_flatten = tensor_flatten / world_size
					tensor_unflatten = unflatten_tensors_grad(tensor_flatten, list(amp.master_params(optimizer)))	
					for param_model, unflat in zip(amp.master_params(optimizer), tensor_unflatten):
						param_model.grad.data = unflat
					

											
				if(crossover_flag == True and sync_grad == True):
					crossover(model, model_cp, rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, elitism=False, elite_ratio=elite_ratio, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)		
					
		
				if((crossover_flag == False) and (allreduce == False) and (sync_grad == True)):
					sgp(model, model_cp, rank, world_size, epoch, i, len_loader, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)
				if(clip_grad == True):
					torch.nn.utils.clip_grad_norm(amp.master_params(optimizer), 1)					
				

				if(lars == True and sync_lars == True):
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
#
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
#
					#	wait_broadcast = dist.broadcast(norm_each_rank[i], src=i, async_op=True)	
					#	wait_list.append(wait_broadcast)
					#	
					#for i, wait_bcast in enumerate(wait_list) :
					#	wait_bcast.wait()
					#	norm_tensor[i::world_size] = norm_each_rank[i]
					#print(f"rank {rank},  div {norm_each_rank[5]}")
					#optimizer.set_norms(norm_tensor) 	

					norms = []
					for param in amp.master_params(optimizer) :
						param_clone = param.clone()
						grad_clone = param.grad.clone()

						if grad_clone is None or torch.isnan(grad_clone).any() or torch.isinf(grad_clone).any():
							grad_clone.fill_(0)

						dist.all_reduce(param_clone.data, op=dist.ReduceOp.SUM)
						dist.all_reduce(grad_clone.data, op=dist.ReduceOp.SUM)

						weight_norm = torch.norm(param_clone.data).float().item()
						grad_norm = torch.norm(grad_clone.data).float().item()
						norms.append([weight_norm, grad_norm])

					norms = torch.Tensor(norms).float().cuda()
					optimizer.set_norms(norms) 	

				if(lars == True and sync_lars == False):						 					
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


			else:
				loss.backward()
				if(allreduce == True and sync_grad == True):
					#for param in model.parameters():
					#	if p.grad is None or torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
					#		continue
					#	dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=False)
					#	param.grad.data = param.grad.data *(1.0/world_size)
					tensor_flatten = flatten_tensors_grad(list(model.parameters()))
					dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
					tensor_flatten = tensor_flatten / world_size
					tensor_unflatten = unflatten_tensors(tensor_flatten, list(model.parameters()))	
					for param_model, unflat in zip(model.parameters(), tensor_unflatten):
						param_model.grad.data = unflat

				if(crossover_flag == True and sync_grad == True):
					crossover(model, model_cp, rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, elitism=False, elite_ratio=elite_ratio, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)		
				
		
				if((crossover_flag == False) and (allreduce == False) and (sync_grad == True)):
					sgp(model, model_cp, rank, world_size, epoch, i, len_loader, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)

				if(clip_grad == True):
					torch.nn.utils.clip_grad_norm(model.parameters(), 1)

				if(lars == True and sync_lars == True):				
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
						norm = [weight_norm, grad_norm]
						norms.append(norm)
		
					for i in range(world_size):
						if(i == rank):
							norms = torch.Tensor(norms).float().cuda()
							norm_each_rank[i] = norms
						dist.broadcast(norm_each_rank[i], src=i, async_op=False)	
		
						norm_tensor[i::world_size] = norm_each_rank[i]
					optimizer.set_norms(norm_tensor)
				if(lars == True and sync_lars == False):						 					
					norms = []
					for tensor in model.parameters() :
						weight_norm = torch.norm(tensor.data).float().item()
						grad_norm = torch.norm(tensor.grad.data).float().item()
						norm = [weight_norm, grad_norm]
						norms.append(norm)
					norms = torch.Tensor(norms).float().cuda()	
					optimizer.set_norms(norms) 

			optimizer.step()
			optimizer.zero_grad()



			#if((crossover_flag == False) and (allreduce == False)):
			#	model.transfer_params()
			#	model.gossip_flag.wait(timeout=300)
			update_learning_rate(lrdecay, baselr, optimizer, maxepoch, epoch, itr, len(loader), world_size, batch_size)
			if(allreduce == True and sync_grad == False):
				#for param in model.parameters():
				#	dist.all_reduce(param.data, op=dist.ReduceOp.SUM, async_op=False)
				#	param.data = param.data *(1.0/world_size)
				tensor_flatten = flatten_tensors(list(model.parameters()))
				dist.all_reduce(tensor_flatten, op=dist.ReduceOp.SUM)
				tensor_flatten = tensor_flatten / world_size
				tensor_unflatten = unflatten_tensors(tensor_flatten, list(model.parameters()))	
				for param_model, unflat in zip(model.parameters(), tensor_unflatten):
					param_model.data = unflat

			if(crossover_flag == True and sync_grad == False):
				crossover(model, model_cp, rank, world_size, model_name, rseed_per_rank, roulettes=roulettes, elitism=False, elite_ratio=elite_ratio, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)		
			

			if((crossover_flag == False) and (allreduce == False) and (sync_grad == False)):
				sgp(model, model_cp, rank, world_size, epoch, i, len_loader, amp=amp, optimizer=optimizer, sync_grad=sync_grad, amp_flag=amp_flag)

			#if((crossover_flag == True ) and ((i % 10) == 0)):
				#generation = epoch
				#fitness_roulette = create_fintess_roulette(rank, model, criterion, fitness_loader, generation, rseed_per_rank, world_size=world_size, parent_num=parent_num, selection_method="roulette_select")
			#	crossover(model, model_cp, rank, world_size, model_name, rseed_per_rank, roulette=None, elitism=False, elite_ratio=elite_ratio)
		#model.gossip_flag.wait(timeout=300)
		elapsed_time = time.time()-batch_time
		if(rank == 0):
			for param_group in optimizer.param_groups:
				print(param_group['lr'])

		losses,top1 = validate(model, val_loader, criterion)

		with open(f'/scratch/a1158a01/{args.tag}_{file_prefix}_{world_size}_{batch_size}_{rank}_sync_lars_{sync_lars}_amp_{amp_flag}_clip_grad_{clip_grad}_baselr_{baselr}_maxepoch_{maxepoch}_lrdecay_{lrdecay}_lars_{lars}_lars_coef_{lars_coef}_chromo_{args.chromosome}_'+'val.csv', '+a') as f:
			print('{ep}, {rank}, '
				'{loss:.4f},'
				'{top1:.3f},'
				'{val}, {elapsed_time}'
				.format(ep=epoch,rank=rank, loss=losses, top1=top1, val=top1, elapsed_time=elapsed_time),
				file=f
				)
		#model.state_dict()
		#optimizer.state_dict()


	

def update_peers_per_itr(model, ppi_schedule, epoch):
	ppi = None
	e_max = -1
	for e in ppi_schedule:
		if e_max <= e and epoch >= e:
			e_max = e
			ppi = ppi_schedule[e]
	model.update_gossiper('peers_per_itr', ppi)


def update_learning_rate(lrdecay, target_lr, optimizer, maxepoch, epoch, itr, itr_per_epoch, world_size, batch_size,
						scale=1):
	target_lr = target_lr
	lr = 0
	print(f"itr per epoch {itr_per_epoch}, itr {itr}, epoch {epoch}")
	if(epoch < 5):
		count = epoch * itr_per_epoch + itr + 1
		incr = (target_lr - 0.1) * (count / (5 * itr_per_epoch))
		#print(count / (5 * itr_per_epoch))
		lr = 0.1 + incr
	elif(lrdecay == 'step'):
		if(epoch >= 5):
			lr = target_lr
		if(epoch >= 81):
			lr = target_lr * 0.1
		if(epoch >= 122 ):
			lr = target_lr * 0.1 * 0.1
	elif(lrdecay=='poly'):
		decay = (1 - float(epoch) / maxepoch) ** 2
		lr = target_lr * decay

	#print(itr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
def make_dataloader(rank, batch_size, world_size):
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	train_dataset = torchvision.datasets.CIFAR10(root='/scratch/a1158a01/data', train=True, download=True, transform=transform_train)
	# sampler produces indices used to assign data samples to each agent
	train_sampler = torch.utils.data.distributed.DistributedSampler(
						shuffle=True,
						dataset=train_dataset,
						num_replicas=world_size,
						rank=rank)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size,
		shuffle=False,
		num_workers=1,
		pin_memory=False, sampler=train_sampler)

	return train_loader, train_sampler

def make_validation_dataloader(rank, batch_size):
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	torch.cuda.manual_seed(r)
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	testset = torchvision.datasets.CIFAR10(root='/scratch/a1158a01/data', train=False, download=True, transform=transform_test)
	val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
	return val_loader


def make_fitness_dataloader():
	torch.manual_seed(777)
	torch.cuda.manual_seed(777)
	transform_train = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
										download=True, transform=transform_train)

	# sampler produces indices used to assign data samples to each agent

	evaluate_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=100,
		shuffle=False,
		num_workers=0,
		pin_memory=False)

	return evaluate_loader

def init_model():
	model = models.resnet18()
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

def sgp(model, model_cp, rank, world_size, epoch, i, max_iter, amp=None, optimizer=None, sync_grad=False, amp_flag=False ):
	#params = model.parameters()
	#if(amp_flag == True):
	#	params = amp.master_params(optimizer)	
	iter_num = epoch * max_iter + i
	send_to = int((rank+(2**(iter_num%(math.log(world_size,2)))))%world_size)
	receive_from = int((rank-(2**(iter_num%(math.log(world_size,2)))))%world_size)
	for p, cp in zip(amp.master_params(optimizer), model_cp.parameters()):
		if(sync_grad == True):
			cp.data.copy_(p.grad.data)
		else :
			cp.data.copy_(p.data)

	for param in model_cp.parameters():
		completed = dist.isend(tensor=param.data, dst=send_to)
		dist.recv(tensor=param.data, src=receive_from)
		completed.wait()

	for p, cp in zip(amp.master_params(optimizer), model_cp.parameters()):
		if(sync_grad == True):
			p.grad.data.copy_(0.5*(cp.data+p.grad.data))
		else:
			p.data.copy_(0.5*(cp.data+p.data))

def crossover_comm(param, receive_from, send_to, select_rank, rank, world_size, elitism=False, elite_ratio=0.25):
	#dist.broadcast(tensor=param.data, src=select_model, async_op=False)
	#랭크 별로 Seed를 고정하여 통신 없이 각 노드들이 다음에 받을것을 확인.

	##print(f"Rank {rank} Select_layer : {select_rank}")
	#print(f"Rank {rank} receive_from : {receive_from}")
	#print(f"Rank {rank} Send_to : {send_to}")	
	#print(f"send source: {rank}  destination: {i}" )		
	completed = dist.isend(tensor=param.data, dst=send_to)
#
	#print(f"recv source: {j}  destination: {rank}" )		
	#completed = dist.irecv(tensor=param.data, src=j)
	#req.append(completed)		
	dist.recv(tensor=param.data, src=receive_from)
	
	#completed = dist.isend(tensor=param.data, dst=(world_size + rank+1)%world_size)
	#dist.recv(tensor=param.data, src=(world_size + rank-1)%world_size)
	completed.wait()



def crossover(model, model_cp, rank, world_size, model_name, rseed_per_rank, roulettes=None, elitism=False, elite_ratio=0.25, amp=None, optimizer=None, sync_grad=False, amp_flag=False):
	#dist.barrier(async_op=False)
	#model_cp.load_state_dict(model.module.state_dict())
	#except my rank in selection
	
	#if(roulette != None):
	#	for i in range(0, world_size):
	#		fitness_sum = sum(roulette) - roulette[i]
	#		roulette_except_rank = [(float)(f)/ fitness_sum for f in roulette]
	#		roulette_except_rank[i] = 0
	#		roulettes.append(roulette_except_rank)
	#else:
	#params = model.parameters()
	#if(amp_flag == True):
	#	params = amp.master_params(optimizer)	

	for p, cp in zip(amp.master_params(optimizer), model_cp.parameters()):
		if(sync_grad == True):
			cp.data.copy_(p.grad.data)
		else:
			cp.data.copy_(p.data)
	old_block = -1
	select_rank = select_layer(rank, rseed_per_rank, world_size, roulettes)
	#select_rank = rseed_per_rank[0].choice(world_size, world_size, replace=False)
	receive_from = select_rank[rank]		 
	send_to = select_rank.index(rank)
	for n, param in model_cp.named_parameters():
		if(n == 'conv1.weight' or n == 'bn1.weight' or n == 'bn1.bias'):
			old_block = 0
		else:
			is_changed =  False if (old_block == find_block(model_name, n)[0]) else True
			if is_changed :
				select_rank = select_layer(rank, rseed_per_rank, world_size, roulettes)
				#select_rank = rseed_per_rank[0].choice(world_size, world_size, replace=False)
				receive_from = select_rank[rank]		 
				send_to = select_rank.index(rank)
			old_block = find_block(model_name, n)[0] if(is_changed) else old_block
			#if(is_changed and rank == 0):
			#	print('changed')
		#select_model = 1
		#if(rank == 0):
		#	print(f'select model {select_model} name :{n}')
		#select_model = 0
		#print(select_model)
		crossover_comm(param, receive_from, send_to, select_rank, rank, world_size, elitism=elitism, elite_ratio=elite_ratio)
		#dist.barrier()

	for p, cp in zip(amp.master_params(optimizer), model_cp.parameters()):
		if(sync_grad == True):
			p.grad.data.copy_(0.5*(cp.data+p.grad.data))
		else:
			p.data.copy_(0.5*(cp.data+p.data))
		#p.data.copy_(cp.data)
	#dist.barrier()

def create_fintess_roulette(rank, model, criterion, fitness_loader, generation, rseed_per_rank, world_size=4, parent_num=4, selection_method=None):
	avg_loss = evaluate_fitness(fitness_loader, model, criterion, generation)
	avg_loss_tensor = torch.FloatTensor(1).fill_(avg_loss)
	gather_loss= [torch.FloatTensor(1).fill_(0) for _ in range(world_size)]
	gather_loss_tensor = torch.stack(gather_loss).reshape(-1)
	if(rank == 0):
		dist.gather(avg_loss_tensor, gather_list=gather_loss, async_op=False)
		gather_loss_tensor = torch.stack(gather_loss).reshape(-1)
	else:
		dist.gather(avg_loss_tensor, 0)
	dist.broadcast(gather_loss_tensor, 0)

	gather_loss_list = gather_loss_tensor.tolist()
	loss_sum = sum(gather_loss_list)
	fitness_roulette = [(1-((float)(l)/ loss_sum))/(float)(world_size-1) for l in gather_loss_list ]
	#print(f"fitness_roulette {fitness_roulette}")
	#exit()
	if(parent_num < world_size):
		if(selection_method == "roulette_select"):
			parents = rseed_per_rank[world_size].choice(world_size, parent_num, replace=False, p=fitness_roulette)
			#make parents roulette
			#print(f"parents {parents}")

			gather_loss_list = [item if idx in parents else 0 for idx, item in enumerate(gather_loss_list) ]
			loss_sum = sum(gather_loss_list)
			fitness_roulette = [(1-((float)(l)/ loss_sum))/(float)(parent_num-1) if (l !=0) else 0 for l in gather_loss_list ]
			#print(f"fitness_roulette {fitness_roulette}")

	return fitness_roulette
def evaluate_fitness(fitness_loader, model, criterion, generation):
	val_loss = 0
	total = 0
	correct = 0

	model.eval()
	#model.train()
	_fitness_loader = fitness_loader.__iter__()
	list_len = len(_fitness_loader)
	start = (generation *1) % list_len
	with torch.no_grad():
		for i, (features, target) in enumerate(_fitness_loader, start=start):
			if(i -start == 100):
				break
			target = target.cuda(non_blocking=True)
			features = features.cuda(non_blocking=True)
			output = model(features)
			loss = criterion(output, target)
			val_loss += loss.item()
			_, predicted = output.max(1)
			total += target.size(0)
			correct += predicted.eq(target).sum().item()

	return ((float)(val_loss))/((float)(100.))


def is_NotElite(rank, roulette_wheel=None, elitism=False, elite_ratio=0.25):
	if(elitism == False):
		return True
	elif(roulette_wheel==None):
		return True
	else :
		world_size = len(roulette_wheel)
		elite_number = (int)(world_size * elite_ratio)
		sorted_wheel = sorted(roulette_wheel, reverse=True)
		index = [roulette_wheel.index(v) for v in sorted_wheel]
		not_elite_members = index[elite_number:]
		#print(roulette_wheel)
		#print(not_elite_members)
		#exit()
		if(rank in not_elite_members):
			return True
		else:
			return False

def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """

	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)

		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))
		#print(correct)

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

if __name__ == '__main__':
	main()	