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
def make_dataloader():
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
	]))





def make_validation_dataloader():
	ii64 = np.iinfo(np.int64)
	r = random.randint(0, ii64.max)
	torch.manual_seed(r)
	torch.cuda.manual_seed(r)
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	val_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=128, shuffle=False,
		num_workers=2, pin_memory=True)
	
	return val_loader

make_dataloader( )
val_loader = make_validation_dataloader( )



