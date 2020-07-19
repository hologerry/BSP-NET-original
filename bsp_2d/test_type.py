import os
import numpy as np
import h5py

data_hdf5_name_source = 'data/svg_fonts_AILTU_source.hdf5'
if os.path.exists(data_hdf5_name_source):
    data_dict_source = h5py.File(data_hdf5_name_source, 'r')
    data_voxels_source = data_dict_source['pixels'][:]
    print(type(data_voxels_source))
    print(data_voxels_source.shape)

shape_batch_size = 20
idx = 10
batch_index_list = np.arange(len(data_voxels_source))
dxb = batch_index_list[idx*shape_batch_size:(idx+1)*shape_batch_size]
style_vox3d = []
for b_idx in dxb:
    font_idx = b_idx // 5
    char_idxs = np.random.choice(5, 3)
    one = []
    for ch_idx in char_idxs:
        one.append(data_voxels_source[font_idx*5+ch_idx])
    one = np.concatenate(one, 2)
    one = np.reshape(one, (1, 64, 64, 3))
    style_vox3d.append(one)

style_vox3d = np.concatenate(style_vox3d, 0)
print(style_vox3d.shape)