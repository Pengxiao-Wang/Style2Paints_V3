
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
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


# In[2]:


source_data_path = "/data4/wangpengxiao/danbooru2017/original"
source_img_path = glob.glob(osp.join(source_data_path,'*/*.jpg'))
source_img_path += glob.glob(osp.join(source_data_path,'*/*.png'))
source_img_path = sorted(source_img_path)


# In[3]:


print(len(source_img_path))


# In[ ]:


# for path in tqdm(source_img_path):
#     try:
#         Image.open(path)
#     except:
#         os.system("rm "+path)   
#         print("rm "+path)

# source_img = np.array(source_img)


# In[4]:


def RandomCenterCrop(path, min_size, max_size):
    '''
    simulate dataset step 1: Crop Randomly
    '''
    size = np.random.randint(min_size, max_size)
    
    img = cv2.imread(path)
    h, w, _ = img.shape

    top = np.random.randint(0, h - size)
    left = np.random.randint(0, w - size)

    return img[top:size+top, left:size+left, :]


# In[5]:


def get_patch(path, min_patch_size, max_patch_size):
    '''
    get patch from clothes
    '''
    patch_size = np.random.randint(min_patch_size, max_patch_size)
    
    img = cv2.imread(path)
    h, w, _ = img.shape
    
    center_h = h/2
    center_w = w/2
    
    patch = img[int(center_h - patch_size/2):int(center_h + patch_size/2), int(center_w - patch_size/2):int(center_w + patch_size/2), :]
    
    return patch


# In[6]:


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
    #cv2.waitKey(0)
    return sketchKeras


# In[7]:


def get_mask(path):
    '''
    提取衣服的mask
    返回numpy数组
    '''
    from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill,     show_fill_map
    from linefiller.thinning import thinning

    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

    fills = []
    result = binary

    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)

    fillmap = merge_fill(fillmap)


    for i in range(len(fillmap[:,0])):
        for j in range(len(fillmap[0,:])):
            if fillmap[i,j] == 1:
                fillmap[i,j] = 0
            else:
                fillmap[i,j] = 1
    
    return fillmap


# In[9]:


sketch_path = "/data4/wangpengxiao/danbooru2017/original_sketch"
os.mkdir(sketch_path)
for path in  tqdm(source_img_path):
    sketch_img = edge_detecton(path)
    cv2.imwrite(osp.join(sketch_path, osp.basename(path)), sketch_img)    


# In[ ]:


random_crop_path = "/data4/wangpengxiao/zalando_random_crop"
patch_path = "/data4/wangpengxiao/zalando_center_patch"
sketch_path = "/data4/wangpengxiao/zalando_sketch"
for path in  tqdm(source_img_path):
    try:
    #step1_1： make randomly croped rectangular patches 
        r_im = RandomCenterCrop(path, 64, 256)
        cv2.imwrite(osp.join(random_crop_path, osp.basename(path)), r_im)
    #step1_2： make randomly croped rectangular patches    
        p_im = get_patch(path, 64, 256)
        cv2.imwrite(osp.join(patch_path, osp.basename(path)), p_im)

        sketch_img = edge_detecton(path)
        cv2.imwrite(osp.join(sketch_path, osp.basename(path)), sketch_img)
    except:
        os.system("rm "+path)


# In[16]:


from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill,     show_fill_map
from linefiller.thinning import thinning
def get_region_picture(path):
    '''
    获取不规则形状的图片，背景是黑色0，方便rotate
    '''
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret, binary = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)

    fills = []
    result = binary

    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)

    fillmap = merge_fill(fillmap)

    fillmap = thinning(fillmap)

    #获得region mask
    for i in range(len(fillmap[:,0])):
        for j in range(len(fillmap[0,:])):
            if fillmap[i,j] == 0:
                fillmap[i,j] = 1
            else:
                fillmap[i,j] = 0
    #获得region picture    
    im = cv2.imread(path)
#     plt.imshow(im)
    rgb_fillmap = np.zeros(im.shape)
    rgb_fillmap[:,:,0] = fillmap
    rgb_fillmap[:,:,1] = fillmap
    rgb_fillmap[:,:,2] = fillmap
    im = im * rgb_fillmap
    
    return im.astype('uint8')


# In[17]:


region_picture_path = "/data4/wangpengxiao/danbooru2017/original_region_picture"
for path in tqdm(source_img_path):
    rp_im = get_region_picture(path)
    cv2.imwrite(osp.join(region_picture_path, osp.basename(path)), rp_im)
    


# In[ ]:


random_crop_path = "/data4/wangpengxiao/zalando_random_crop"
patch_path = "/data4/wangpengxiao/zalando_center_patch"
sketch_path = "/data4/wangpengxiao/zalando_sketch"
region_picture_path = "/data4/wangpengxiao/zalando_region_picture"
random_crop_img_path = glob.glob(osp.join(random_crop_path,'*.jpg')) 
random_crop_img_path = sorted(random_crop_img_path)

region_img_path = glob.glob(osp.join(region_picture_path,'*.jpg')) 
region_img_path = sorted(region_img_path)


# In[ ]:


def Random_paste_patch_img(ori_img, patch_img):

    paste_x = np.random.randint(0, ori_img.size[0] - patch_img.size[0])
    paste_y = np.random.randint(0, ori_img.size[1] - patch_img.size[1])
    rotate_angle = np.random.randint(1, 359)
    resize_x = np.random.randint(64, 384)
    resize_y = np.random.randint(64, 384)
    patch_img = patch_img.resize((resize_x,resize_y))
    tem = ori_img.copy()
    tem.paste(patch_img.rotate(rotate_angle),(paste_x,paste_y))
    tem = np.array(tem)
    ori_img = np.array(ori_img)
#     for i in range(ori_img.shape[0]):
#         for j in range(ori_img.shape[1]):
#             if (tem[i,j,:] == np.array([0,0,0])).all():
#                 tem[i,j,:] = ori_img[i,j,:]
    coordinate = np.where(tem == np.array([0,0,0]))
    for i in range(len(coordinate[0])):
        tem[coordinate[0][i],coordinate[1][i],:] = ori_img[coordinate[0][i],coordinate[1][i],:]
    ori_img = np.array(tem)
    ori_img = Image.fromarray(ori_img)
#     plt.imshow(ori_img)
    
    return ori_img


def Random_paste_region_img(ori_img, region_img):

    paste_x = np.random.randint(0, ori_img.size[0] - patch_img.size[0])
    paste_y = np.random.randint(0, ori_img.size[1] - patch_img.size[1])
    rotate_angle = np.random.randint(1, 359)
    resize_x = np.random.randint(64, 384)
    resize_y = np.random.randint(64, 384)
    region_img = region_img.resize((resize_x,resize_y))
    tem = ori_img.copy()
    tem.paste(region_img.rotate(rotate_angle),(paste_x,paste_y))
    tem = np.array(tem)
    ori_img = np.array(ori_img)
#     for i in range(ori_img.shape[0]):
#         for j in range(ori_img.shape[1]):
#             if (tem[i,j,:] == np.array([0,0,0])).all():
#                 tem[i,j,:] = ori_img[i,j,:]
    coordinate = np.where(tem == np.array([0,0,0]))
    for i in range(len(coordinate[0])):
        tem[coordinate[0][i],coordinate[1][i],:] = ori_img[coordinate[0][i],coordinate[1][i],:]
    ori_img = np.array(tem)
    ori_img = Image.fromarray(ori_img)
#     plt.imshow(ori_img)
    
    return ori_img


# In[ ]:


import random
# ori_img = Image.open(source_img_path[0])
step_1_path = '/data4/wangpengxiao/zalando_step_1'
for path in tqdm(source_img_path):
    ori_img = Image.open(path)
    patch_num = np.random.randint(4, 12)
    region_num = np.random.randint(4, 12)
    for i in range(patch_num):
        patch_img = Image.open(random.choice(random_crop_img_path))
        ori_img = Random_paste_patch_img(ori_img, patch_img)
    for i in range(region_num):
        region_img = Image.open(random.choice(region_img_path))
        ori_img = Random_paste_region_img(ori_img, region_img)

    ori_img.save(osp.join(step_1_path, osp.basename(path)))


# In[ ]:


ori_img.save('aaa.jpg')


# In[ ]:


ori_img = np.array(ori_img)
c = np.where(ori_img == np.array([246,246,246]))


# In[ ]:


print(c)


# In[ ]:


ori_img

