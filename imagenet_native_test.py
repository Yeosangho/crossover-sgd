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



import math
from copy import copy, deepcopy
#from LARC import LARC



def find_block(model_name, name):	
	return [model_name.index(n) for n in model_name if(n in name)]

def main():
	batch_size = 256
	world_size = 4 
	baselr = 29
	maxepoch = 90
	local_itr = 1
	torch.backends.cudnn.benchmark = True


	model = init_model()

	torch.manual_seed(1)
	def init_weights(m):
		if type(m) == nn.Linear or type(m) == nn.Conv2d:
			torch.nn.init.kaiming_uniform_(m.weight)
	model.apply(init_weights)

	log_softmax = nn.LogSoftmax(dim=1)
	
	loader = make_dataloader(batch_size, world_size)
	val_loader = make_validation_dataloader( batch_size)

	criterion = nn.CrossEntropyLoss()
	
	optimizer = torch.optim.SGD(model.parameters() , lr=baselr, momentum=0.9, nesterov=True)
	optimizer.zero_grad()
	_train_loader = loader.__iter__()


	for epoch in range(0, maxepoch):
		batch_time = time.time()	

		_train_loader = loader.__iter__()

		len_loader = len(_train_loader)
		for itr,(batch, target) in enumerate(_train_loader, start=0):
			minibatch_time = time.time()
			model.train()

			target = target.cuda(non_blocking=True)

			batch = batch.cuda(non_blocking=True)
			output = model(batch)
			loss = criterion(output, target)
			loss = loss /local_itr

			loss.backward()
		
			optimizer.step()
			optimizer.zero_grad()
			print(time.time() - minibatch_time)	
			minibatch_data_load_itme = time.time()
		elapsed_time = time.time()-batch_time





def make_dataloader( batch_size, world_size):
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_dataset = torchvision.datasets.ImageFolder(root='/scratch/x1801a03/a1158a01_scratch/train/', transform=transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
	return train_loader



def make_validation_dataloader(batch_size):
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	torch.cuda.manual_seed(r)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	val_set = torchvision.datasets.ImageFolder(root='/scratch/x1801a03/a1158a01_scratch/val/',transform=transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]))

	val_loader = torch.utils.data.DataLoader(
		val_set,
		batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)
	
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


if __name__ == '__main__':
	main()	