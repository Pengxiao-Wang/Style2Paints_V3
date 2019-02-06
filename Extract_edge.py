
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

from PIL import Image
import os.path as osp
import glob  
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
from keras.models import load_model
from helper import *

mod = load_model('mod.h5')


# In[ ]:


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
    
    line_mat = np.amax(line_mat, 2)

    sketchKeras = show_active_img_and_save_denoise('sketchKeras', line_mat, 'sketchKeras.jpg')

    return sketchKeras


# In[ ]:


source_data_path = "original image path"#"/data4/wangpengxiao/danbooru2017/original"
source_img_path = glob.glob(osp.join(source_data_path,'*/*.jpg'))
source_img_path += glob.glob(osp.join(source_data_path,'*/*.png'))
source_img_path = sorted(source_img_path)


# In[ ]:


#get sketch
sketch_path = "sketch save path"#"/data4/wangpengxiao/danbooru2017/original_sketch"
os.mkdir(sketch_path)
for path in  tqdm(source_img_path):
    sketch_img = edge_detecton(path)
    cv2.imwrite(osp.join(sketch_path, osp.basename(path)), sketch_img)    

