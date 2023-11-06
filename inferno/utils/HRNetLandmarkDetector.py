from abc import abstractmethod, ABC
import numpy as np
import torch
import pickle as pkl
from inferno.utils.FaceDetector import FaceDetector, MTCNN
import os, sys
from inferno.utils.other import get_path_to_externals 
from pathlib import Path
from torchvision import transforms as tf
from face_alignment.detection.sfd.sfd_detector import SFDDetector
from face_alignment.utils import get_preds_fromhm, crop
from collections import OrderedDict
import torch.nn.functional as F


path_to_hrnet = (Path(get_path_to_externals()) / ".." / ".." / "HRNet-Facial-Landmark-Detection").absolute()
# path_to_hrnet = (Path(get_path_to_externals())  / "HRNet-Facial-Landmark-Detection").absolute()

if str(path_to_hrnet) not in sys.path:
    sys.path.insert(0, str(path_to_hrnet))

from lib.config import config, update_config
from lib.core import function
import lib.models as models
from lib.core.evaluation import decode_preds, compute_nme

INPUT_SIZE = 256


class HRNet(FaceDetector):

    def __init__(self, device = 'cuda', instantiate_detector='sfd', threshold=0.5):

        cfg = path_to_hrnet / "experiments/300w/face_alignment_300w_hrnet_w18.yaml"
        model_file = path_to_hrnet / "hrnetv2_pretrained" / "HR18-300W.pth"

        # cfg = path_to_hrnet / "experiments/aflw/face_alignment_aflw_hrnet_w18.yaml"
        # model_file = path_to_hrnet / "hrnetv2_pretrained" / "HR18-AFLW.pth"

        # model_file = path_to_hrnet / "hrnetv2_pretrained" / "hrnetv2_w18_imagenet_pretrained.pth"
        config.defrost()
        config.merge_from_file(cfg)
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        self.model = models.get_face_alignment_net(config)
        # self.num_landmarks = 68
        self.num_landmarks = config.MODEL.NUM_JOINTS
        self.config = config

        state_dict = torch.load(model_file)

        if model_file.name == "HR18-300W.pth":
            prefix_to_remove = "module."
            state_dict = OrderedDict({k[len(prefix_to_remove):]: v for k, v in state_dict.items()})

        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            res = self.model.load_state_dict(state_dict)
        else:
            # self.model.module.load_state_dict(state_dict)
            self.model.load_state_dict(state_dict)

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
        
        # # self.transforms = [utils.transforms.CenterCrop(INPUT_SIZE)]
        self.transforms = [tf.ToTensor()]
        # self.transforms += [utils.transforms.Normalize([0.518, 0.418, 0.361], [1, 1, 1])]
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

        final_boxes = []
        final_kpts = []

        for bbox in bboxes:
            center = torch.tensor(
                [bbox[2] - (bbox[2] - bbox[0]) / 2.0, bbox[3] - (bbox[3] - bbox[1]) / 2.0])
            # center[1] = center[1] - (bbox[3] - bbox[1]) * 0.12 # this might result in clipped chin
            center[1] = center[1] + (bbox[3] - bbox[1])  * 0.00 # this appears to be a good value
            # scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / self.detector.reference_scale
            # scale = 1.2
            # scale = 1.3
            # scale = 1.4
            scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / self.detector.reference_scale * 0.65 # this appears to be a good value
            # print("Scale: {}".format(scale))
            # print("Bbox: {}".format(bbox))
            # print("Width: {}".format(bbox[2] - bbox[0]))
            # print("Height: {}".format(bbox[3] - bbox[1]))
            # scale = (bbox[2] - bbox[0] + bbox[3] - bbox[1]) / 256
            # scale = ((bbox[2] - bbox[0] + bbox[3] - bbox[1]) / image.shape[0] ) * 0.85
            images_ = crop(image, center, scale, resolution=256.0)
            images = self.crop_to_tensor(images_)
            if images.ndimension() == 3:
                images = images.unsqueeze(0)
            # images = nn.atleast4d(images).cuda()

            # X_recon, lms, X_lm_hm = self.detect_in_crop(images)
            pts_img, X_lm_hm = self.detect_in_crop(images, center.unsqueeze(0), torch.tensor([scale]))
            # pts, pts_img = get_preds_fromhm(X_lm_hm, center.numpy(), scale)
            # torch.cuda.empty_cache()
            if pts_img is None:
                del pts_img
                if with_landmarks:
                    return [],  f'kpt{self.num_landmarks}', []
                else:
                    return [],  f'kpt{self.num_landmarks}'
            else:
                import matplotlib.pyplot as plt
                # # image to numpy array
                # images_np = images.cpu().numpy()[0].transpose((1, 2, 0))
                # images_np = images_ / 255.
                # print("images_np.shape: {}".format(images_np.shape))
                # plt.figure(1)
                # plt.imshow((images_np * 255.).clip(0, 255).astype(np.uint8))
                # plt.figure(2)
                # plt.imshow(image)
                # for i in range(len(lms)):
                for i in range(len(pts_img)):
                    kpt = pts_img[i][:68].squeeze().detach().cpu().numpy()
                    left = np.min(kpt[:, 0])
                    right = np.max(kpt[:, 0])
                    top = np.min(kpt[:, 1])
                    bottom = np.max(kpt[:, 1])
                    final_bbox = [left, top, right, bottom]
                    final_boxes += [final_bbox]
                    final_kpts += [kpt]

                    # plot points                 
                    # plt.figure(1)
                    # plt.plot(kpt[:, 0], kpt[:, 1], 'ro')
                    # plt.figure(2)
                    # plt.plot(pts_img[i][:, 0], pts_img[i][:, 1], 'ro')
                # print("Plotting landmarks")
                # plt.show()

        # del lms # attempt to prevent memory leaks
        if with_landmarks:
            return final_boxes, f'kpt{self.num_landmarks}', final_kpts
        else:
            return final_boxes, f'kpt{self.num_landmarks}'


    @torch.no_grad()
    def detect_in_crop(self, crop, center, scale):
        with torch.no_grad():
            output = self.model(crop)
            score_map = output.data.cpu()
            # preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # center = torch.tensor(crop.shape[2:] ).repeat(crop.shape[0], 1)/ 2
            # scale = torch.ones((crop.shape[0]), dtype=torch.float32)
            score_map = F.interpolate(score_map,  crop.shape[2:], mode='bicubic', align_corners=False)
            preds = decode_preds(score_map, center, scale, crop.shape[2:])

            # resize score map to original image size with F interpolate 

            # # NME
            # # nme_temp = compute_nme(preds, meta)
            # nme_temp = compute_nme(preds, None)

        # lms_in_crop = utils.nn.to_numpy(lms_in_crop.reshape(1, -1, 2))

        return preds, score_map

