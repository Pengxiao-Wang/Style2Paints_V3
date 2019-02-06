
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import numpy as np
import cv2
from tqdm import tqdm
import glob
import os.path as osp
import random
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import matplotlib.pyplot as plt


# In[ ]:


def get_STL(path, num_batch):
    h = 384
    w = 384
    im = cv2.imread(path[0])
    im = im / 255.
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
    
    im = im.reshape(1, h, w, 3)
    im = im.astype('float32')
    
    batch = np.append(im, im, axis=0)
    for p in path: 
        im = cv2.imread(p)
        im = im / 255.
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
        im = im.reshape(1, h, w, 3)
        im = im.astype('float32')
        batch = np.append(batch, im, axis=0)
    
    batch = batch[2:,:,:,:]

    out_size = (h, w)

    # %% Simulate batch
    x = tf.placeholder(tf.float32, [None, h, w, 3])
    x = tf.cast(batch, 'float32')

    # %% Create localisation network and convolutional layer
    with tf.variable_scope('spatial_transformer_0'):

        # %% Create a fully-connected layer with 6 output nodes
        n_fc = 6
        W_fc1 = tf.Variable(tf.zeros([h * w * 3, n_fc]), name='W_fc1')

        # %% Zoom into the image
        a, b, c, d, e, f = np.random.random(6)/10

        initial = np.array([[1-a, b, c], [d, 1-e, f]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(tf.zeros([num_batch, h * w * 3]), W_fc1) + b_fc1
        h_trans = transformer(x, h_fc1, out_size)

    # %% Run session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(h_trans, feed_dict={x: batch})
        sess.close()
    
    return y


# In[ ]:


source_data_path = "source data path"#"/data4/wangpengxiao/danbooru2017/original"
STL_path = "STL result path"#"/data4/wangpengxiao/danbooru2017/original_STL"

source_img_path = glob.glob(osp.join(source_data_path,'*/*.jpg'))
source_img_path += glob.glob(osp.join(source_data_path,'*/*.png'))
source_img_path = sorted(source_img_path)

batch_size = 16

os.makedirs(STL_path,exist_ok=True)
q = []
count = 0
c = 0
for path in tqdm(source_img_path):
    c += 1
    if c != 0 :
        if count == batch_size-1 :
            q.append(path)
            tf.reset_default_graph()
            im = get_STL(q, batch_size)
            tf.get_default_graph().finalize()
            for j in range(len(im)):
                img = im[j]
                amin, amax = img.min(), img.max() # 求最大最小值
                img = (img-amin)/(amax-amin) # (矩阵元素-最小值)/(最大值-最小值)
                
                cv2.imwrite(osp.join(STL_path, osp.basename(q[j])), (img*255).astype('uint8'))            
                
            count = 0
            q = []
        else:
            count += 1
            q.append(path)
    else:
        continue

