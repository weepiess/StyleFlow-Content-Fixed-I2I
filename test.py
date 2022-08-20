import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.trainers.Trainer_StyleFlow import Trainer,set_random_seed,merge_model#,Trainer_PCA
from model.utils.dataset import get_data_loader_folder_pair
from model.utils.sampler import DistributedGivenIterationSampler
from model.utils.utils import get_config
import model.network.net as net

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict.yaml')
    parser.add_argument('--model_path', type=str, default='../scripts/output/wikiart/model_save/187500.ckpt.pth.tar')
    opts = parser.parse_args()
    args = get_config(opts.config)
    args['model_path'] = opts.model_path
    print('job name: ',args['job_name'])
    return args

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

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
    if not os.path.exists(os.path.join(args['output'],args['task_name'],'img_gen')):
        os.makedirs(os.path.join(args['output'],args['task_name'],'img_gen'))
        print('mkdir img_gen folder')


    #trainer = Trainer(args)
    test_dataset = get_data_loader_folder_pair(args['rootA'],
                                                args['rootB'],
                                                args['infotxt'],
                                                args['batch_size'], 
                                                True, 
                                                args['keep_percent'],
                                                get_direct=args['get_direct'],
                                                used_domain=args['used_domain'],
                                                train_vr=args['train_vr'],
                                                return_paths=True)
                                                
    test_sampler = DistributedGivenIterationSampler(test_dataset,
        args['max_iter'], args['batch_size'], world_size=1, rank=0, last_iter=last_iter)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['workers'],
        pin_memory=False,
        sampler=test_sampler)

        
    # for batch_id, (content_iter, style_iter, source_iter, target_iter, code_iter, imgA_aug, imgB_aug, imgC_aug, imgD_aug) in enumerate(train_loader):
    #     trainer.train(batch_id, content_iter, style_iter, source_iter, target_iter, code_iter, imgA_aug, imgB_aug, imgC_aug, imgD_aug)
    model = merge_model(args)
    #print(args['model_path'])
    if os.path.isfile(args['model_path']):
        print("--------loading checkpoint----------")
        
        checkpoint = torch.load(args['model_path'])
        checkpoint['state_dict'] = remove_prefix(checkpoint['state_dict'], 'module.')

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args['model_path']))
    else:
        raise('no checkpoint found', args['model_path'])

    vgg = net.vgg
    vgg.load_state_dict(torch.load(args['vgg']))
    encoder = net.Net(vgg).cuda()    

    # model = model.to(device)
    model.cuda()
    # model = nn.DataParallel(model)
    model.eval()
    for batch_id, (imgA, imgB, name) in enumerate(test_loader):
        base_code = encoder.cat_tensor(imgB.cuda())
        stylized = model(imgA.cuda(),domain_class=base_code.cuda())
        stylized = torch.clamp(stylized,0,1)
        output_name = os.path.join(args['output'], args['task_name'],'img_gen', name[0])
        save_image(stylized.cpu(), output_name, nrow=1)

if __name__ == "__main__":
    main()
