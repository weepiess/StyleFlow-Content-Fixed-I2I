import argparse
import os

import torch
from torch.utils.data import DataLoader

from model.trainers.Trainer_StyleFlow import Trainer,set_random_seed#,Trainer_PCA
from model.utils.dataset import get_data_loader_folder_pair
from model.utils.sampler import DistributedGivenIterationSampler
from model.utils.utils import get_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict.yaml')
    opts = parser.parse_args()
    args = get_config(opts.config)
    print('job name: ',args['job_name'])
    return args


def main():
    torch.backends.cudnn.benchmark = True

    set_random_seed(0)

    last_iter = -1
    args = parse_args()

    if not os.path.exists(args['output']):
        os.makedirs(args['output'])
        print('mkdir args.output')
    if not os.path.exists(os.path.join(args['output'],args['task_name'])):
        os.makedirs(os.path.join(args['output'],args['task_name']))
        print('mkdir task folder')
    if not os.path.exists(os.path.join(args['output'],args['task_name'],'img_save')):
        os.makedirs(os.path.join(args['output'],args['task_name'],'img_save'))
        print('mkdir img_save folder')
    if not os.path.exists(os.path.join(args['output'],args['task_name'],'model_save')):
        os.makedirs(os.path.join(args['output'],args['task_name'],'model_save'))
        print('mkdir model folder')

    trainer = Trainer(args)
    train_dataset = get_data_loader_folder_pair(args['rootA'],
                                                args['rootB'],
                                                args['infotxt'],
                                                args['batch_size'], 
                                                True, 
                                                args['keep_percent'],
                                                get_direct=args['get_direct'],
                                                used_domain=args['used_domain'],
                                                train_vr=args['train_vr'])
                                                
    train_sampler = DistributedGivenIterationSampler(train_dataset,
        args['max_iter'], args['batch_size'], world_size=1, rank=0, last_iter=last_iter)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['workers'],
        pin_memory=False,
        sampler=train_sampler)
    for batch_id, (content_iter, style_iter, source_iter, target_iter, code_iter, imgA_aug, imgB_aug, imgC_aug, imgD_aug) in enumerate(train_loader):
        trainer.train(batch_id, content_iter, style_iter, source_iter, target_iter, code_iter, imgA_aug, imgB_aug, imgC_aug, imgD_aug)


if __name__ == "__main__":
    main()
