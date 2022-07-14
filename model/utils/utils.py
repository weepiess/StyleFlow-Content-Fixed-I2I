import yaml
import os
import torch

def get_config(config):
    config = os.path.join('../',config)
    with open(config, 'r') as stream:
        return yaml.load(stream)

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
    
class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, last_iter=-1):
        #assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.last_iter = last_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.last_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr'] * self.lr_mults, self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr
