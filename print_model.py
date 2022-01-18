import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
model = models.resnet50().cuda()
for name, param in model.named_parameters():
	if('bn' in name) or name.endswith(".bias"):
		print(len(param.shape))