import os
import random

import cv2
import h5py
import numpy as np

# synthetic data - hollow diamond, cross, diamond
num_shapes = 1000
image_size = 64
batch_size = image_size*image_size

alphabets = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

char_names = 'AILTU'

hdf5_file = h5py.File(f"svg_fonts_{char_names}_f_source.hdf5", 'w')
hdf5_file.create_dataset("pixels", [num_shapes*len(char_names), image_size, image_size, 1], np.float, compression=9)

font_id_split_name = '/home1/gaoy/svg/magenta/svg_vae_data/font_id_split_name_eval.txt'
font_id_names = open(font_id_split_name).readlines()

image_root_path = "/home1/gaoy/svg/magenta/svg_vae_data_2/ttf_fonts_images/"

def check(font_path):
    for i in range(10, 62):
        if not os.path.exists(os.path.join(font_path, f"{i:02d}.png")):
            return False
    return True

font_idx_hdf5 = 0
used_font_ids = open('used_font_ids.txt', 'w')

done = False
for font_id_name in font_id_names:
    # font_id = font_id_name.split(',')[0].strip()
    # font_split = font_id_name.split(',')[1].strip()
    # if font_split == 'eval':
    #     font_split = 'train'
    font_id = '000012'
    font_split = 'train'
    # font_name = font_id_name.split(',')[2].strip()
    font_path = os.path.join(image_root_path, font_split+'_post', font_id)
    if check(font_path):
        for char in char_names:
            char_idx = alphabets.index(char)
            img = cv2.imread(os.path.join(font_path, f'{char_idx:02d}.png'), 0)  # use 'A' only
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            img = 255 - img
            img = img / 255.0
            if font_idx_hdf5 == 0:
                cv2.imwrite('tmp.png', img * 255.0)
            hdf5_file['pixels'][font_idx_hdf5] = img
            font_idx_hdf5 += 1
            if font_idx_hdf5 % 100 == 0:
                print(font_idx_hdf5)
            used_font_ids.write(font_id + '\n')
            if font_idx_hdf5 >= num_shapes * len(char_names):
                done = True
                break
    if done:
        break

hdf5_file.close()
