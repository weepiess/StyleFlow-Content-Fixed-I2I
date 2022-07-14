import random
import numpy as np
import os

import torch
import torch.nn as nn
import model.network.net as net

from torchvision.utils import save_image
from model.network.glow import Glow
from model.utils.utils import IterLRScheduler,remove_prefix
from tensorboardX import SummaryWriter
from model.layers.activation_norm import calc_mean_std

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    torch.save(state, filename+'.pth.tar')

class merge_model(nn.Module):
    def __init__(self,cfg):
        super(merge_model,self).__init__()
        self.glow = Glow(3, cfg['n_flow'], cfg['n_block'], affine=cfg['affine'], conv_lu=not cfg['no_lu'])

    def forward(self,content_images, domain_class):
        z_c = self.glow(content_images, forward=True)
        stylized = self.glow(z_c, forward=False, style=domain_class)

        return stylized

def get_smooth(I, direction):
        #smooth
        weights = torch.tensor([[0., 0.],
                                [-1., 1.]]
                                ).cuda()
        weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
        weights_y = torch.transpose(weights_x, 0, 1)
        if direction == 'x':
            weights = weights_x
        elif direction == 'y':
            weights = weights_y

        F = torch.nn.functional
        output = torch.abs(F.conv2d(I, weights, stride=1, padding=1))  # stride, padding
        return output

def avg(R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(R, direction))

def get_gradients_loss(I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    I_gray = torch.mean(I, dim=1, keepdim=True)
    gradients_I_x = get_smooth(I_gray,'x')
    gradients_I_y = get_smooth(I_gray,'y')

    return torch.mean(gradients_I_x * torch.exp(-10 * avg(R_gray, 'x')) + gradients_I_y * torch.exp(-10 * avg(R_gray, 'y')))
    
class Trainer():
    def __init__(self,cfg,seed=0):
        self.init = True
        set_random_seed(seed)
        self.cfg = cfg
        Mmodel = merge_model(cfg)
        
        self.model = Mmodel
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.lr_scheduler = IterLRScheduler(self.optimizer, cfg['lr_steps'], cfg['lr_mults'], last_iter=cfg['last_iter'])
        
        vgg = net.vgg
        vgg.load_state_dict(torch.load(cfg['vgg']))
        self.encoder = net.Net(vgg).cuda()

        self.logger = SummaryWriter(os.path.join(self.cfg['output'],self.cfg['task_name'],'runs'))

    def train(self,batch_id, content_iter, style_iter, source_iter, target_iter, code_iter, imgA_aug, imgB_aug, imgC_aug, imgD_aug):
        content_images = content_iter.cuda()
        style_images = style_iter.cuda()
        target_style = style_iter
        
        domain_weight = torch.tensor(1).cuda()

        if self.init:
            base_code = self.encoder.cat_tensor(style_images.cuda())
            self.model(content_images,domain_class=base_code.cuda())
            self.init = False
            return

        base_code = self.encoder.cat_tensor(target_style.cuda())
        stylized = self.model(content_images,domain_class=base_code.cuda())
        stylized = torch.clamp(stylized,0,1)

        smooth_loss = get_gradients_loss(stylized, target_style.cuda())
        loss_c, loss_s = self.encoder(content_images, style_images, stylized, domain_weight)
        loss_c = loss_c.mean().cuda()
        loss_s = loss_s.mean().cuda()

        Loss = self.cfg['content_weight']*loss_c + self.cfg['style_weight']*loss_s + smooth_loss#  + self.cfg['histo_weight']*hist_loss #+ self.cfg['mse_weight']*loss_mse

        Loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        reduce_loss = Loss.clone()
        loss_c_ = loss_c.clone()
        loss_s_ = loss_s.clone()
        smooth_loss_ = smooth_loss.clone()

        current_lr = self.lr_scheduler.get_lr()[0]
        self.logger.add_scalar("current_lr", current_lr, batch_id + 1)
        self.logger.add_scalar("loss_s", loss_s_.item(), batch_id + 1)
        self.logger.add_scalar("smooth_loss", smooth_loss_.item(), batch_id + 1)
        self.logger.add_scalar("Loss", reduce_loss.item(), batch_id + 1)

        if batch_id % 100 == 0:
            output_name = os.path.join(self.cfg['output'], self.cfg['task_name'],'img_save', 
                            str(batch_id)+'_'+str(code_iter[0].cpu().numpy()[0])+'.jpg')
            output_images = torch.cat((content_images.cpu(), style_images.cpu(), stylized.cpu(),target_style.cpu()), 
                                    0)
            save_image(output_images, output_name, nrow=1)


        if batch_id % 500 == 0:
            save_checkpoint({
                'step':batch_id,
                'state_dict':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict()
                },os.path.join(self.cfg['output'],self.cfg['task_name'],'model_save',str(batch_id)+ '.ckpt'))
