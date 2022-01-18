""" Layer-wise adaptive rate scaling for SGD in PyTorch! """
import torch
from torch.optim.optimizer import Optimizer, required
import math

class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, model, params, lr=required, momentum=.9,
                 weight_decay=0, eta=0.001, max_epoch=200, dist=None, world_size=None, amp=None, rank=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch)
        self.dist = dist
        self.world_size = world_size
        self.amp = amp
        self.rank = rank
        self.norms = []
        self.model = model
        self.wd = weight_decay 
        super(LARS, self).__init__(params, defaults)
    def set_norms(self, norms):
        self.norms = norms

    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            #print(1);
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']
            weight_norms = []
            grad_norms = []
            norms = []
            #for p in self.amp.master_params(self):
            #    if p.grad is None:
            #        weight_norms.append(0.0)
            #        grad_norms.append(0.0)
            #    else:
            #        param_state = self.state[p]
            #        d_p = p.grad.data
    #
            #        weight_norm = torch.norm(p.data).double().item()
            #        grad_norm = torch.norm(d_p).double().item()
            #        weight_norms.append(weight_norm)
            #        grad_norms.append(grad_norm)
            #norms = [weight_norms,grad_norms]
            #norms = torch.Tensor(norms).double().cuda()
            #torch.div(norms, self.world_size) 
            ##print(f"before {norms}")
            ##print("=================================")
            #self.dist.all_reduce(norms, op=self.dist.ReduceOp.SUM, async_op=False)
            #self.dist.barrier()
            #print(f"after {norms}")
            #print(norms)
            count = 0
            for n, p in self.model.named_parameters():
                # Global LR computed on polynomial decay schedule
                if len(p.shape) == 1 or n.endswith(".bias"): 
                #if n.endswith(".bias"):              
                    weight_decay = 0.0
                else :
                    weight_decay = group['weight_decay']

                grad = p.grad.clone()
                if p.grad is None :
                    count = count+1
                    continue
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    #grad[torch.isinf(grad)] = 0
                    #grad[torch.isnan(grad)] = 0
                    #grad = grad.fill_(0)
                    count = count+1
                    continue
                param_state = self.state[p]
                d_p = grad.data
                weight_norm = self.norms[count][0]
                grad_norm = self.norms[count][1]
                #if grad_norm == 0 :
                #    count = count + 1
                #    continue
                #weight_norm = torch.norm(p.data)
                #torch.div(weight_norm, self.world_size) 
                #grad_norm = torch.norm(d_p)
                #torch.div(grad_norm, self.world_size)
                #print(f"rank {self.rank} {weight_norm.type()}")
                #print(f"rank {self.rank} {grad_norm.type()}")
                #self.dist.barrier()
                #self.dist.all_reduce(grad_norm.cpu(), op=self.dist.ReduceOp.SUM, async_op=False)
                #self.dist.all_reduce(weight_norm.cpu(), op=self.dist.ReduceOp.SUM, async_op=False)
                #self.dist.barrier()
                global_lr = lr
                actual_lr = 0.0
                if len(p.shape) == 1 or n.endswith(".bias"): 
                    actual_lr = global_lr
                else :
                    if(grad_norm > 0 and weight_norm > 0):
                        local_lr = weight_norm / (grad_norm + weight_decay * weight_norm +1e-9)                
                        local_lr = local_lr * eta
                        # Update the momentum term
                        actual_lr = local_lr * global_lr
                    else :
                        actual_lr = global_lr
        

                #print(actual_lr)
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']

                buf.mul_(momentum).add_(torch.mul((d_p + weight_decay * p.data), actual_lr))
                if torch.isnan(buf).any() or torch.isinf(buf).any():
                    count = count + 1
                    continue
                p.data.add_(-buf)
                count = count+1
        return loss