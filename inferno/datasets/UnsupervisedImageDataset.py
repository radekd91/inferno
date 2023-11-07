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


import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from torchvision.transforms import ToTensor

from inferno.utils.FaceDetector import load_landmark


class UnsupervisedImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_list, landmark_list=None, image_transforms=None, im_read=None, 
                align_landmarks=False):
        super().__init__()
        self.image_list = image_list
        self.landmark_list = landmark_list
        if landmark_list is not None and len(landmark_list) != len(image_list):
            raise RuntimeError("There must be a landmark for every image")
        self.image_transforms = image_transforms
        self.im_read = im_read or 'skio'
        if self.im_read in ['skvreader', 'skvffmpeg']:
            self.ordered = True
            self.last_index = -1
        else: 
            self.ordered = False
        if self.im_read == 'skvffmpeg':
            self.next_frame_it = self.image_list.nextFrame()

        if isinstance(self.image_list, np.ndarray): 
            self.im_read = None

    def __getitem__(self, index):
        if self.ordered: 
            if index != self.last_index + 1:
                raise RuntimeError("The images must be read in order because of the skvideo reader")
            self.last_index = index
        # if index < len(self.image_list):
        #     x = self.mnist_data[index]
        # raise IndexError("Out of bounds")
        try:
            if isinstance(self.image_list, np.ndarray):
                img = self.image_list[index].transpose([2, 0, 1]).astype(np.float32)
                img_torch = torch.from_numpy(img)
                path = f"{index:05d}"
            elif self.im_read == 'skio':
                img = imread(self.image_list[index])
                img = img.transpose([2, 0, 1]).astype(np.float32)
                img_torch = torch.from_numpy(img)
                path = str(self.image_list[index])
            elif self.im_read == 'pil':
                img = Image.open(self.image_list[index])
                img_torch = ToTensor()(img)
                path = str(self.image_list[index])
                # path = f"{index:05d}"
            elif self.im_read == 'skvreader':
                img = next(self.image_list)
                img = img.transpose([2, 0, 1]).astype(np.float32)
                img_torch = torch.from_numpy(img) / 255.
                path = f"{index:05d}"
            elif self.im_read == 'skvffmpeg':
                img = next(self.next_frame_it)
                img = img.transpose([2, 0, 1]).astype(np.float32)
                img_torch = torch.from_numpy(img) / 255.
                path = f"{index:05d}"
            else:
                raise ValueError(f"Invalid image reading method {self.im_read}")
        except Exception as e:
            print(f"Failed to read '{self.image_list[index]}'. File is probably corrupted. Rerun data processing")
            raise e

        if self.image_transforms is not None:
            img_torch = self.image_transforms(img_torch)

        batch = {"image" : img_torch,
                "path" : path}

        if self.landmark_list is not None:
            landmark_type, landmark = load_landmark(self.landmark_list[index])
            landmark_torch = torch.from_numpy(landmark)

            if self.image_transforms is not None:
                landmark_torch = self.image_transforms(landmark_torch)

            batch["landmark"] = landmark_torch

        return batch

    def __len__(self):
        if self.im_read in ['skvreader', 'skvffmpeg']:
            return self.image_list.getShape()[0]
        return len(self.image_list)