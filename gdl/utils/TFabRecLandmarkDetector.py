from abc import abstractmethod, ABC
import numpy as np
import torch
import pickle as pkl
from gdl.utils.FaceDetector import FaceDetector, MTCNN
import os, sys
from gdl.utils.other import get_path_to_externals 
from pathlib import Path
from torchvision import transforms as tf
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from face_alignment.utils import get_preds_fromhm, crop

# path_to_3fabrec = (Path(get_path_to_externals()) / ".." / ".." / "3FabRec").absolute()
path_to_3fabrec = (Path(get_path_to_externals())  / "3FabRec").absolute()

if str(path_to_3fabrec) not in sys.path:
    sys.path.insert(0, str(path_to_3fabrec))

from csl_common.utils import nn, cropping
from csl_common import utils
from landmarks import fabrec


INPUT_SIZE = 256


class TFabRec(FaceDetector):

    def __init__(self, device = 'cuda', instantiate_detector='sfd', threshold=0.5):
        # model =  path_to_3fabrec / 'data/models/snapshots/demo'
        # model =  path_to_3fabrec / 'data/models/snapshots/lms_wflw'
        # self.num_landmarks = 98
        
        # model =  path_to_3fabrec / 'data/models/snapshots/lms_aflw'
        model =  path_to_3fabrec / 'data/models/snapshots/lms_300w'
        self.num_landmarks = 68

        self.net = fabrec.load_net(str(model), num_landmarks= self.num_landmarks)
        self.net.eval()

        self.detector = None
        if instantiate_detector == 'mtcnn':
            self.detector = MTCNN()
        elif instantiate_detector == 'sfd': 
            # Get the face detector

            face_detector_kwargs =  {
                "filter_threshold": threshold
            }
            self.detector = SFDDetector(device=device, verbose=False, **face_detector_kwargs)

        elif instantiate_detector is not None: 
            raise ValueError("Invalid value for instantiate_detector: {}".format(instantiate_detector))
        
        # self.transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
        self.transforms = [utils.transforms.ToTensor()]
        self.transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
        self.crop_to_tensor = tf.Compose(self.transforms)


    
    # @profile
    @torch.no_grad()
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        if detected_faces is None: 
            bboxes = self.detector.detect_from_image(image)
        else:
            print("Image size: {}".format(image.shape)) 
            bboxes = [np.array([0, 0, image.shape[1], image.shape[0]])]

        for bbox in bboxes:
            center = torch.tensor(
                [bbox[2] - (bbox[2] - bbox[0]) / 2.0, bbox[3] - (bbox[3] - bbox[1]) / 2.0])
            # center[1] = center[1] - (bbox[3] - bbox[1]) * 0.12
            center[1] = center[1] + (bbox[3] - bbox[1])  * 0.05
            # scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / self.detector.reference_scale
            scale = 0.9
            images = crop(image, center, scale, resolution=256.0)
            images = self.crop_to_tensor(images)
            images = nn.atleast4d(images).cuda()

            X_recon, lms, X_lm_hm = self.detect_in_crop(images)
            pts, pts_img = get_preds_fromhm(X_lm_hm, center.numpy(), scale)
            # torch.cuda.empty_cache()
            if lms is None:
                del lms
                if with_landmarks:
                    return [],  f'kpt{self.num_landmarks}', []
                else:
                    return [],  f'kpt{self.num_landmarks}'
            else:
                boxes = []
                kpts = []

                # import matplotlib.pyplot as plt
                # # image to numpy array
                # images_np = images.cpu().numpy()[0].transpose((1, 2, 0))
                # print("images_np.shape: {}".format(images_np.shape))
                # plt.figure(1)
                # plt.imshow(images_np)
                # plt.figure(2)
                # plt.imshow(image)
                # for i in range(len(lms)):
                for i in range(len(pts_img)):
                    kpt = pts_img[i][:68].squeeze()
                    left = np.min(kpt[:, 0])
                    right = np.max(kpt[:, 0])
                    top = np.min(kpt[:, 1])
                    bottom = np.max(kpt[:, 1])
                    bbox = [left, top, right, bottom]
                    boxes += [bbox]
                    kpts += [kpt]

                    # plot points                 
                    # plt.figure(1)
                    # plt.plot(kpt[:, 0], kpt[:, 1], 'ro')
                    # plt.figure(2)
                    # plt.plot(pts_img[i][:, 0], pts_img[i][:, 1], 'ro')
                # print("Plotting landmarks")
                # plt.show()

        # del lms # attempt to prevent memory leaks
        if with_landmarks:
            return boxes, f'kpt{self.num_landmarks}', kpts
        else:
            return boxes, f'kpt{self.num_landmarks}'


    @torch.no_grad()
    def detect_in_crop(self, crop):
        with torch.no_grad():
            X_recon, lms_in_crop, X_lm_hm = self.net.detect_landmarks(crop)
        lms_in_crop = utils.nn.to_numpy(lms_in_crop.reshape(1, -1, 2))
        return X_recon, lms_in_crop, X_lm_hm


