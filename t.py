import numpy as np
import os
import pandas as pd
from glob import glob
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

# files = glob('./data/images/*.jpg')
# print(len(files))
source_image_dir = './data/images'
target_size = (512,512,3)
def get_opposite_image( image_file, side):
    if side == 'R':
        opposite_image = image_file.replace('_R_', '_L_')
    else:
        opposite_image = image_file.replace('_L_', '_R_')

    return opposite_image


def load_pair( image_file, side):
    op_image_name =  get_opposite_image(image_file, side)
    image =  load_image(image_file,target_size)
    op_image =  load_image(op_image_name,target_size)
    if side == 'R':
        image  = np.fliplr(image)
        op_image = np.fliplr(op_image)
    comb= images_h_stack([op_image, image])
    comb = resize(comb, target_size)

    return comb

def images_h_stack( images):
    imgs_comb = np.hstack((np.asarray(i) for i in images))
    return imgs_comb


def load_image( image_file, target_size=None):
    image_path = os.path.join( source_image_dir, image_file)
    # image_array = np.random.randint(low=0, high=255, size=( target_size[0],  target_size[1], 3))

    image = Image.open(image_path)
    image_array = np.asarray(image.convert("RGB"))
    image_array = image_array / 255.
    image_array = crop_image(image_array)

    if target_size is not None:
        image_array = resize(image_array,  target_size)
    # if side == 'L':
    #     image_array = np.fliplr(image_array)
    return image_array

def crop_image(img,tol=0,margin=100):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    row_start = max(0,row_start-margin)
    col_start = max(0,col_start-margin)
    return img[row_start:row_end+margin,col_start:col_end+margin]


comb = load_pair('P28_L_CM_MLO.jpg','L')


plt.imshow(comb)
plt.show()
