# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os.path as osp
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from numpy.lib import math
from tqdm import tqdm

input_mean = 127.5
input_std = 127.5


def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_arcface_input(face, img):
    aimg = face_align.norm_crop(img, landmark=face.kps)
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    return blob[0], aimg


def get_center(bboxes, img):
    img_center = img.shape[0] // 2, img.shape[1] // 2
    size = bboxes.shape[0]
    distance = np.Inf
    j = 0
    for i in range(size):
        x1, y1, x2, y2 = bboxes[i, 0:4]
        dx = abs(x2 - x1) / 2.0
        dy = abs(y2 - y1) / 2.0
        current = dist((x1 + dx, y1 + dy), img_center)
        if current < distance:
            distance = current
            j = i

    return j


def get_image(name, to_rgb=False):
    images_dir = osp.join(Path(__file__).parent.absolute(), '../images')
    ext_names = ['.jpg', '.png', '.jpeg']
    image_file = None
    for ext_name in ext_names:
        _image_file = osp.join(images_dir, "%s%s" % (name, ext_name))
        if osp.exists(_image_file):
            image_file = _image_file
            break
    assert image_file is not None, '%s not found' % name
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:, :, ::-1]
    return img


class DetectorMica():
    def __init__(self, device='cuda:0'):
        self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(224, 224))

    def run(self):
        print('Start arcface...')
        min_det_score = 0.5
        src = ''
        for image_path in tqdm(sorted(glob(f'{src}/images/*/*'))):
            dst = ''
            Path(dst).parent.mkdir(exist_ok=True, parents=True)
            img = get_image(image_path[0:-4])
            bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] == 0:
                continue
            i = get_center(bboxes, img)
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            if det_score < min_det_score:
                continue
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            blob, aimg = get_arcface_input(face, img)
            np.save(dst[0:-4], blob)
            cv2.imwrite(dst, face_align.norm_crop(img, landmark=face.kps, image_size=224))
