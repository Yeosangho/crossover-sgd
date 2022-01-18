import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms

import random
def make_dataloader():
    ii64 = np.iinfo(np.int64)
    r = random.randint(0, ii64.max)
    torch.manual_seed(r)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageNet(root='/scratch/x2026a02/', split='train',
                                        download=False, transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


def make_validation_dataloader():
    ii64 = np.iinfo(np.int64)
    r = random.randint(0, ii64.max)
    torch.manual_seed(r)
    torch.cuda.manual_seed(r)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_set = torchvision.datasets.ImageNet(root='/scratch/x2026a02/', split='val', download=False, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
make_validation_dataloader()
make_dataloader()
