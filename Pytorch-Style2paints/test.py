
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import sys
import time
from optparse import OptionParser
import numpy as np
import glob  
import os.path as osp
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms

from transforms import GroupRandomCrop
from transforms import GroupScale

from eval import eval_net
from unet import UNet
from unet import Discriminator
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
import easydict

from dataset_multi import ClothDataSet


# In[2]:


args = easydict.EasyDict({
    'epochs' : 100,
    'batch_size' : 12,
    'train_path' : '/data6/wangpengxiao/zalando_dataset2/train',
    'val_path' : '/data6/wangpengxiao/zalando_dataset2/val',
    'sketch_path' : "/data6/wangpengxiao/zalando_sketch2",
    'draft_path' : "/data6/wangpengxiao/zalando_final2",
    'save_path' : "/data6/wangpengxiao/style2paint2" ,
    'weight_path' : "/data6/wangpengxiao/style2paint3/25/25_Unet_checkpoint.pth.tar",
    'hanfu_path' : "/data6/wangpengxiao/hanfu",
    'img_size' : 300,
    're_size' : 256,
    'learning_rate' : 1e-3,
    'gpus' : '[0]',
    'lr_steps' : [5, 10, 15, 20,  25],
    "lr_decay" : 0.1,
    'lamda_L1' : 100,
    'workers' : 8,
    'weight_decay' : 1e-4
})


# In[3]:


Unet = UNet(in_channels=4, out_channels=3)
checkpoint = torch.load(args.weight_path)
# print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
# model.load_state_dict(base_dict)


# In[4]:


Unet.load_state_dict(base_dict)
Unet.eval()


# In[5]:


from keras.models import load_model
mod = load_model('/data2/wangpengxiao/GANs/style2paints3/V4/mod.h5')
from helper import *
def edge_detecton(path):
    '''
    get sketch
    '''
    from_mat = cv2.imread(path)
    width = float(from_mat.shape[1])
    height = float(from_mat.shape[0])
    new_width = 0
    new_height = 0
    if (width > height):
        from_mat = cv2.resize(from_mat, (512, int(512 / width * height)), interpolation=cv2.INTER_AREA)
        new_width = 512
        new_height = int(512 / width * height)
    else:
        from_mat = cv2.resize(from_mat, (int(512 / height * width), 512), interpolation=cv2.INTER_AREA)
        new_width = int(512 / height * width)
        new_height = 512
    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    #sketchKeras_colored = show_active_img_and_save('sketchKeras_colored', line_mat, 'sketchKeras_colored.jpg')
    line_mat = np.amax(line_mat, 2)
    #sketchKeras_enhanced = show_active_img_and_save_denoise_filter2('sketchKeras_enhanced', line_mat, 'sketchKeras_enhanced.jpg')
    #sketchKeras_pured = show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat, 'sketchKeras_pured.jpg')
    sketchKeras = show_active_img_and_save_denoise('sketchKeras', line_mat, 'sketchKeras.jpg')
#     cv2.waitKey(0)
    return sketchKeras


# In[6]:


#输入图片预处理
hanfu_img = glob.glob(osp.join(args.hanfu_path,'*.jpg'))
hanfu_img = sorted(hanfu_img)
gt_img = glob.glob(osp.join(args.val_path,'*.jpg'))
gt_img = sorted(gt_img)

p = gt_img[335]
gt = Image.open(p).convert('RGB')
#sk = Image.open(osp.join(args.sketch_path, osp.basename(p))).convert('L')
df = Image.open(osp.join(args.draft_path, osp.basename(p))).convert('RGB')
# plt.imshow(gt)
sketch = edge_detecton(hanfu_img[5])
sketch = Image.fromarray(sketch).convert('L')


# In[7]:


# sk = Image.open(hanfu_img[5]).convert('L')
# plt.imshow(Image.open(hanfu_img[5]).convert('RGB'))


# In[8]:


# style_path = '/data6/wangpengxiao/hanfu/style'
# sty = glob.glob(osp.join(style_path,'*.jpg'))
# sty = sorted(sty)
# df = Image.open(sty[1]).convert('RGB')
# plt.imshow(df)


# In[ ]:


def test(Unet,args,gt,sk,df):
    gt = gt.resize((args.re_size, args.re_size), Image.BICUBIC)
    sk = sk.copy()
    sk = sk.resize((args.re_size, args.re_size), Image.BICUBIC)
    # df = gt.copy()
    df = df.resize((299, 299), Image.BICUBIC)

    gt = np.array(gt)
    point_map = np.zeros(gt.shape)
    #
    coordinate = np.where(np.sum(gt,axis=2) < np.sum(np.array([240,240,240])))
    num_of_point = np.random.randint(1, 6)
    a = random.sample(range(0,max(num_of_point,len(coordinate[0]))),10)

    for i in range(len(a)):  
        r,g,b = gt[coordinate[0][a[i]],coordinate[1][a[i]],:]
        cv2.circle(point_map,(coordinate[1][a[i]],coordinate[0][a[i]]),4,(int(r),int(g),int(b)),-1) 
    #
    gt = Image.fromarray(gt)
    point_map = Image.fromarray(point_map.astype('uint8'))
    pm = point_map.copy()

    sk = transforms.ToTensor()(sk)
    point_map = transforms.ToTensor()(point_map)
    gt = transforms.ToTensor()(gt)
    df = transforms.ToTensor()(df)
    sk = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sk)
    point_map = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(point_map)
    gt = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt)
    df = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(df)


    input = torch.cat((sk,point_map),0)         

    input = input.view(1,input.shape[0],input.shape[1],input.shape[2])
    df = df.view(1,df.shape[0],df.shape[1],df.shape[2])


    input=input.float()
    df=df.float()
    with torch.no_grad():            
        input_var = input
        df_var = df

    fake_pic = Unet(input_var,df_var)
    fake = vutils.make_grid(fake_pic, normalize=True, scale_each=True)
    fake = np.transpose(fake.detach().numpy(),(1,2,0))
    
    return fake,pm


# In[ ]:


#输入图片预处理
from tqdm import tqdm
save_path = '/data6/wangpengxiao/hanfu/result'
hanfu_img = glob.glob(osp.join(args.hanfu_path,'*.jpg'))
hanfu_img = sorted(hanfu_img)
gt_img = glob.glob(osp.join(args.val_path,'*.jpg'))
gt_img = sorted(gt_img)
for i in range(len(hanfu_img)):
    sketch = edge_detecton(hanfu_img[i])
    sketch = Image.fromarray(sketch).convert('L')
    for j in tqdm(range(30)):
        p = gt_img[j]
        gt = Image.open(p).convert('RGB')
        df = Image.open(p).convert('RGB')


        pic,pm = test(Unet,args,gt,sketch,df)
        pm.save(osp.join(save_path,str(i)+'_'+str(j)+'point_map.jpg'))
        pic = Image.fromarray((pic*255).astype('uint8'))
        plt.imshow(pic)
        gt.save(osp.join(save_path,str(i)+'_'+str(j)+'gt.jpg'))
        pic.save(osp.join(save_path,str(i)+'_'+str(j)+'fake.jpg'))


# In[ ]:


# pic = test(Unet,args,gt,sketch,df)
# cv2.imwrite(osp.join(save_path,str(i)+'_'+str(j)+'fake.jpg'),pic)


# In[ ]:


# plt.imshow(pic)
# cv2.imwrite(osp.join(save_path,str(i)+'_'+str(j)+'fake.jpg'),(pic*255).astype('uint8'))

