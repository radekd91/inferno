"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""


import pickle as pkl
import compress_pickle as cpkl
import hickle as hkl
from pathlib import Path
import numpy as np
from timeit import default_timer as timer
import h5py


def _load_hickle_file(filename, start_frame=None, end_frame=None):
    # ## NEW version that reads only the frames we need
    return _load_hdf5_group_dict(filename, start_frame, end_frame)

    ## OLD version that reads the whole file
    # try:
    #     # no idea why but sometimes it fails to load the file just using the path but opening explicitly then works
    #     data = hkl.load(filename)
    # except OSError as e:
    #     with open(filename, "rb") as f:
    #         data = hkl.load(f)
    # return data

def _save_hickle_file(data, filename): 
    try:
        # no idea why but sometimes it fails to load the file just using the path but opening explicitly then works
        hkl.dump(data, filename)
    except OSError as e:
        import h5py
        # open a h5py file for writing 
        with h5py.File(filename, 'w') as f:
            hkl.dump(data, f)
    return data

def load_reconstruction_list(filename, start_frame=None, end_frame=None):
    reconstructions = _load_hickle_file(filename, start_frame, end_frame)
    return reconstructions


def save_reconstruction_list(filename, reconstructions):
    _save_hickle_file(reconstructions, filename)
    # hkl.dump(reconstructions, filename)

def save_reconstruction_list_v2(filename, reconstructions, overwrite=False):
    _save_hdf5_dict(reconstructions, filename)


def load_reconstruction_list_v2(filename, start_frame=None, end_frame=None):
    return _load_hdf5_dict(filename, start_frame, end_frame)


def load_emotion_list(filename, start_frame=None, end_frame=None):
    emotions = _load_hickle_file(filename, start_frame, end_frame)
    return emotions


def save_emotion_list(filename, emotions):
    _save_hickle_file(emotions, filename)
    # hkl.dump(emotions, filename)


def _load_hdf5_dict(filename, start_frame=None, end_frame=None):
    data_dict = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            dset = f[key]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = dset.shape[1]
            # for some reason we saved the data with a leading batch dimension of size 1, so let's just run with it
            data_dict[key] = dset[:, start_frame:end_frame]
    return data_dict


def _load_hdf5_group_dict(filename, start_frame=None, end_frame=None):
    data_dict = {}
    with h5py.File(filename, 'r') as f:
        group = f['data'] # that's how hickle saves it
        for key in group.keys():
            dset = group[key]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = dset.shape[1]
            key_ = key.replace('"','')
            # for some reason we saved the data with a leading batch dimension of size 1, so let's just run with it
            data_dict[key_] = dset[:, start_frame:end_frame]
    return data_dict


def _save_hdf5_dict(data_dict, filename, overwrite=False):
    if not overwrite and Path(filename).exists():
        raise RuntimeError(f"File '{filename}' already exists. Set overwrite=True to overwrite.")
    
    with h5py.File(filename, 'w') as f:
        for key, data in data_dict.items():
            dset = f.create_dataset(key, data.shape, dtype=data.dtype)
            dset[:] = data


def load_emotion_list_v2(filename, start_frame=None, end_frame=None):
    return _load_hdf5_dict(filename, start_frame, end_frame)


def save_emotion_list_v2(filename, emotions, overwrite=False):
    _save_hdf5_dict(emotions, filename, overwrite)


def save_segmentation_list(filename, seg_images, seg_types, seg_names):
    with open(filename, "wb") as f:
        # for some reason compressed pickle can only load one object (EOF bug)
        # so put it in the list
        cpkl.dump([seg_types, seg_images, seg_names], f, compression='gzip')
        # pkl.dump(seg_type, f)
        # pkl.dump(seg_image, f)


def load_segmentation_list(filename):
    try:
        with open(filename, "rb") as f:
            seg = cpkl.load(f, compression='gzip')
            seg_types = seg[0]
            seg_images = seg[1]
            seg_names = seg[2]
    except EOFError as e: 
        print(f"Error loading segmentation list: {filename}")
        raise e
    return seg_images, seg_types, seg_names


def save_segmentation_list_v2(filename, seg_images, seg_types, seg_names, overwrite=False, compression_level=1):
    if not overwrite and Path(filename).exists():
        raise RuntimeError(f"File '{filename}' already exists. Set overwrite=True to overwrite.")
    
    if isinstance(seg_images, list): 
        seg_images = np.stack(seg_images, axis=0)

    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset("frames", seg_images.shape, dtype=seg_images.dtype, compression="gzip", compression_opts=compression_level)
        dset[:] = seg_images
    
        dset_types = f.create_dataset("frame_types", (len(seg_types),), dtype=h5py.special_dtype(vlen=str))
        dset_types[:] = seg_types

        dset_names = f.create_dataset("frame_names", (len(seg_names),), dtype=h5py.special_dtype(vlen=str))
        dset_names[:] = seg_names
    

def load_segmentation_list_v2(filename, start_frame=None, end_frame=None):
    with h5py.File(filename, 'r') as f:
        dset = f["frames"]
        dset_types = f["frame_types"]
        dset_names = f["frame_names"]
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = dset.shape[0]
        seg_images = dset[start_frame:end_frame]
        seg_types = dset_types[start_frame:end_frame]
        seg_names = dset_names[start_frame:end_frame]
    return seg_images, seg_types, seg_names


def load_segmentation(filename):
    with open(filename, "rb") as f:
        seg = cpkl.load(f, compression='gzip')
        seg_type = seg[0]
        seg_image = seg[1]
        # seg_type = pkl.load(f)
        # seg_image = pkl.load(f)
    return seg_image, seg_type



def save_segmentation(filename, seg_image, seg_type):
    with open(filename, "wb") as f:
        # for some reason compressed pickle can only load one object (EOF bug)
        # so put it in the list
        cpkl.dump([seg_type, seg_image], f, compression='gzip')
        # pkl.dump(seg_type, f)
        # pkl.dump(seg_image, f)


def load_segmentation(filename):
    with open(filename, "rb") as f:
        seg = cpkl.load(f, compression='gzip')
        seg_type = seg[0]
        seg_image = seg[1]
        # seg_type = pkl.load(f)
        # seg_image = pkl.load(f)
    return seg_image, seg_type


def save_emotion(filename, emotion_features, emotion_type, version=0):
    with open(filename, "wb") as f:
        # for some reason compressed pickle can only load one object (EOF bug)
        # so put it in the list
        cpkl.dump([version, emotion_type, emotion_features], f, compression='gzip')


def load_emotion(filename):
    with open(filename, "rb") as f:
        emo = cpkl.load(f, compression='gzip')
        version = emo[0]
        emotion_type = emo[1]
        emotion_features = emo[2]
    return emotion_features, emotion_type



face_parsing_labels = {
    0: 'background',  # no
    1: 'skin',
    2: 'nose',
    3: 'eye_g',
    4: 'l_eye',
    5: 'r_eye',
    6: 'l_brow',
    7: 'r_brow',
    8: 'l_ear',  # no?
    9: 'r_ear',  # no?
    10: 'mouth',
    11: 'u_lip',
    12: 'l_lip',
    13: 'hair',  # no
    14: 'hat',  # no
    15: 'ear_r',
    16: 'neck_l',  # no?
    17: 'neck',  # no?
    18: 'cloth'  # no
}

face_parsin_inv_labels = {v: k for k, v in face_parsing_labels.items()}

default_discarded_labels = [
    face_parsin_inv_labels['background'],
    face_parsin_inv_labels['l_ear'],
    face_parsin_inv_labels['r_ear'],
    face_parsin_inv_labels['hair'],
    face_parsin_inv_labels['hat'],
    face_parsin_inv_labels['neck'],
    face_parsin_inv_labels['neck_l']
]


def process_segmentation(segmentation, seg_type, discarded_labels=None):
    if seg_type == "face_parsing":
        discarded_labels = discarded_labels or default_discarded_labels
        # start = timer()
        # segmentation_proc = np.ones_like(segmentation, dtype=np.float32)
        # for label in discarded_labels:
        #     segmentation_proc[segmentation == label] = 0.
        segmentation_proc = np.isin(segmentation, discarded_labels)
        segmentation_proc = np.logical_not(segmentation_proc)
        segmentation_proc = segmentation_proc.astype(np.float32)
        # end = timer()
        # print(f"Segmentation label discarding took {end - start}s")
        return segmentation_proc
    elif seg_type == "face_segmentation_focus":
        segmentation = segmentation > 0.5 
        segmentation = segmentation.astype(np.float32)
        return segmentation
    else:
        raise ValueError(f"Invalid segmentation type '{seg_type}'")


def load_and_process_segmentation(path):
    seg_image, seg_type = load_segmentation(path)
    seg_image = seg_image[np.newaxis, :, :, np.newaxis]
    # end = timer()
    # print(f"Segmentation reading took {end - start} s.")

    # start = timer()
    seg_image = process_segmentation(
        seg_image, seg_type).astype(np.uint8)
    return seg_image
