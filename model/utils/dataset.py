import importlib
import os
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import copy
import cv2 as cv
import skimage.draw as draw
from scipy.ndimage.morphology import distance_transform_edt as dt

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def get_mask(src):
    mask = np.zeros((src.size[1],src.size[0],3))

    mask[:,:,0] = mask[:,:,0] + np.random.randint(-20,20)
    mask[:,:,1] = mask[:,:,1] + np.random.randint(-20,20)
    mask[:,:,2] = mask[:,:,2] + np.random.randint(-20,20)
    return mask

def add_mask(img,mask):
    img_ = np.array(img)
    mask = cv.resize(mask, (img_.shape[1],img_.shape[0]), interpolation=cv.INTER_CUBIC)
    scr_arr = np.array(img_ + mask)#,dtype=np.uint8)


    scr_arr = np.clip(scr_arr,0,255)
    scr_arr = np.array(scr_arr,dtype=np.uint8)

    scr_arr = Image.fromarray(scr_arr)#.convert("RGB")
    return scr_arr

def get_imgs_from_dir(path_a,path_b):
    file_a = open(path_a)
    list_a = []
    for lines in file_a.readlines():
        line = lines.strip('\n')
        list_a.append(line)
    file_a.close()

    file_b = open(path_b)
    list_b = []
    for lines in file_b.readlines():
        line = lines.strip('\n')
        list_b.append(line)
    file_b.close()

    return list_a,list_b

class ImageFolder_pair(data.Dataset):

    def __init__(self, rootA, rootB, info_txt, keep_percent=1, transform=None, transform2=None, return_paths=False,
                 loader=default_loader, dataset_num=1,get_direct=True,used_domain=None,aug=True):

        imgsA=[]
        imgsB=[]
        imgsC=[]
        imgsD=[]
        codes=[]


        imgs_A_1,imgs_B_1 = get_imgs_from_dir(rootA,rootB)
        code_1 = [0]*len(imgs_A_1)

        print('imgs_A_1: ',len(imgs_A_1))
        print('imgs_B_1: ',len(imgs_B_1))
        imgsA += imgs_A_1
        imgsB += imgs_B_1

        random.shuffle(imgs_A_1)
        random.shuffle(imgs_B_1)
        imgsC += imgs_A_1
        imgsD += imgs_B_1

        codes += code_1
        codes += [-1]*len(codes)

        c = list(zip(imgsA, imgsB, codes))
        random.shuffle(c)
        imgsA, imgsB, codes = zip(*c)

        c = list(zip(imgsC, imgsD))
        random.shuffle(c)
        imgsC, imgsD = zip(*c)

        total_len = len(imgsA)
        self.rootA = rootA
        self.rootB = rootB
        self.imgsA = imgsA#[:int(keep_percent*total_len)+1]
        self.imgsB = imgsB#[:int(keep_percent*total_len)+1]
        self.imgsC = imgsC
        self.imgsD = imgsD

        self.code = codes#[:int(keep_percent*total_len)+1]

            
        self.transform = transform
        self.transform2 = transform2
        self.aug = aug
        self.return_paths = return_paths
        self.loader = loader

        print('len_imgA: ',len(self.imgsA))
        print('len_imgB: ',len(self.imgsB))
        print('code: ',len(self.code))

    def __getitem__(self, index):
        #index = 0
        pathA = self.imgsA[index]
        pathB = self.imgsB[index]
        imgA = self.loader(pathA)
        imgB = self.loader(pathB)
        name = pathA.split('/')[-1]
        if self.aug:
            mask1 = get_mask(imgB)
        imgA_aug = add_mask(imgA,mask1)
        imgB_aug = add_mask(imgB,mask1)

        code = self.code[index]
        if self.transform is not None:
            if self.transform2 is not None:
                imgA = self.transform(imgA)
                imgB = self.transform2(imgB)
            else:
                imgA = self.transform(imgA)
                imgB = self.transform(imgB)
            imgA_aug = self.transform(imgA_aug)
            imgB_aug = self.transform(imgB_aug)
            code = torch.tensor([code],dtype=torch.float32)
        if self.return_paths:
            return imgA, imgB, name
        else:
            return imgA, imgB, imgA, imgB, code, imgA_aug, imgB_aug, imgA_aug, imgB_aug

    def __len__(self):
        return len(self.imgsA)

def get_data_loader_folder_pair(input_folderA, input_folderB, info_txt, batch_size, train, keep_percent, new_size=None,
                           height=256, width=256, num_workers=8, crop=True,get_direct=True,used_domain=None,train_vr=False,return_paths=False):
    if train_vr is False:
        transform_list = []
        transform_list.append(transforms.RandomResizedCrop((300,400),scale=(0.7,1.0)))
        #transform_list.append(transforms.Resize((300,400)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        transform2=None
    else:
        transform_list = []
        transform_list.append(torchvision.transforms.Lambda(lambda img: crop_vr_img(img)))
        transform_list.append(transforms.Resize((300,400)))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        transform_list2 = []
        transform_list2.append(torchvision.transforms.Lambda(lambda img: crop_real_img(img)))
        transform_list2.append(transforms.Resize((300,400)))
        transform_list2.append(transforms.ToTensor())
        transform2 = transforms.Compose(transform_list2)

    dataset = ImageFolder_pair(input_folderA, input_folderB, info_txt, keep_percent=keep_percent, transform=transform,transform2=transform2,get_direct=get_direct,used_domain=used_domain,return_paths=return_paths)
    return dataset


from torchvision.transforms.functional import crop

def crop_real_img(image):
    return crop(image, 25, 170, 490, 620)

def crop_vr_img(image):
    return crop(image, 25, 170, 680, 945)
