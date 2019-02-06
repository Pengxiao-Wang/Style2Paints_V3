from helper import *

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
    cv2.waitKey(0)
    return sketchKeras


def get_mask(path):
    '''
    提取衣服的mask
    返回numpy数组
    '''
    from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map
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


from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map
from linefiller.thinning import thinning
def get_region_picture(path):
    '''
    获取不规则形状的图片，背景是黑色0，方便rotate
    '''
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

    paste_x = np.random.randint(0, ori_img.size[0])
    paste_y = np.random.randint(0, ori_img.size[1])
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


def get_STL(path, num_batch):
    h = 1000
    w = 700
    im = cv2.imread(path[0])
    im = im / 255.
#     h = im.shape[0]
#     w = im.shape[1]
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
    
    im = im.reshape(1, h, w, 3)
    im = im.astype('float32')
    
    batch = np.append(im, im, axis=0)
    for p in path: 
        im = cv2.imread(p)
        im = im / 255.
    #     h = im.shape[0]
    #     w = im.shape[1]
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
        im = im.reshape(1, h, w, 3)
        im = im.astype('float32')
        batch = np.append(batch, im, axis=0)
    
#     print(batch.shape)
    batch = batch[2:,:,:,:]
#     print(batch.shape)

    out_size = (h, w)

    # %% Simulate batch
#     batch = np.append(im, im, axis=0)
    # batch.append(im)
    # batch = np.append(batch, im, axis=0)
#     num_batch = 1

    x = tf.placeholder(tf.float32, [None, h, w, 3])
    x = tf.cast(batch, 'float32')

    # %% Create localisation network and convolutional layer
    with tf.variable_scope('spatial_transformer_0'):

        # %% Create a fully-connected layer with 6 output nodes
        n_fc = 6
        W_fc1 = tf.Variable(tf.zeros([h * w * 3, n_fc]), name='W_fc1')

        # %% Zoom into the image
        a = np.random.randint(5, 10)/10
        b = np.random.randint(0, 3)/10
        c = np.random.randint(0, 3)/10
        d = np.random.randint(5, 10)/10 
#         initial = np.array([[s, 0, tx], [0, s,ty]])
        initial = np.array([[a, b, 0], [b, d, 0]])
        initial = initial.astype('float32')
        initial = initial.flatten()

        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(tf.zeros([num_batch, h * w * 3]), W_fc1) + b_fc1
        h_trans = transformer(x, h_fc1, out_size)

    # %% Run session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    y = sess.run(h_trans, feed_dict={x: batch})
#     y = batch
    
    return y


#提取图片主要色
import colorsys
 
def get_dominant_color(image):
    
#颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')
    
#生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))
    
    max_score = 0#原来的代码此处为None
    dominant_color = 0#原来的代码此处为None，但运行出错，改为0以后 运行成功，原因在于在下面的 score > max_score的比较中，max_score的初始格式不定
    
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue
        
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
       
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
       
        y = (y - 16.0) / (235 - 16)
        
        # 忽略高亮色
        if y > 0.9:
            continue
            
        # 忽略白背景
        if ((r>230)&(g>230)&(b>230)):
            continue
        
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)
    
    return dominant_color


def min_dis(point, point_list):
    dis = []
    for p in point_list:
        dis.append(np.sqrt(np.sum(np.square(np.array(point)-np.array(p)))))
    
    return min(dis) 