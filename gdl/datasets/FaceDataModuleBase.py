from email import generator
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from skimage.io import imread, imsave
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, Normalize
from tqdm import tqdm

# from gdl.datasets.FaceVideoDataset import FaceVideoDataModule
from gdl.datasets.IO import save_segmentation
from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp
from gdl.datasets.UnsupervisedImageDataset import UnsupervisedImageDataset
from gdl.utils.FaceDetector import FAN, MTCNN, save_landmark
# try:
#     from gdl.utils.TFabRecLandmarkDetector import TFabRec
# except ImportError:
#     pass
# try:
    # from gdl.utils.Deep3DFaceLandmarkDetector import Deep3DFaceLandmarkDetector
# except ImportError:
#     pass
import pickle as pkl
import types


class FaceDataModuleBase(pl.LightningDataModule):

    def __init__(self, root_dir, output_dir, processed_subfolder, device=None,
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 bb_center_shift_x=0., # in relative numbers
                 bb_center_shift_y=0., # in relative numbers (i.e. -0.1 for 10% shift upwards, ...)
                 processed_ext=".png",                 
                 save_detection_images=True, 
                 save_landmarks=True,
                 ):
        super().__init__()
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.bb_center_shift_x = bb_center_shift_x
        self.bb_center_shift_y = bb_center_shift_y
        self.processed_ext = processed_ext
        self.save_detection_images=save_detection_images
        self.save_landmarks=save_landmarks

        if processed_subfolder is None:
            import datetime
            date = datetime.datetime.now()
            processed_folder = os.path.join(output_dir, "processed_%s" % date.strftime("%Y_%b_%d_%H-%M-%S"))
        else:
            processed_folder = os.path.join(output_dir, processed_subfolder)
        self.output_dir = processed_folder

        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.face_detector_type = face_detector
        self.face_detector_threshold = face_detector_threshold

        self.image_size = image_size
        self.scale = scale

    def _get_max_faces_per_image(self): 
        return 1
    
    def _is_video_dataset(self): 
        return False

    # @profile
    def _instantiate_detector(self, overwrite = False):
        if hasattr(self, 'face_detector'):
            if not overwrite:
                return
            del self.face_detector
        if self.face_detector_type == 'fan':
            self.face_detector = FAN(self.device, threshold=self.face_detector_threshold)
        elif self.face_detector_type == 'mtcnn':
            self.face_detector = MTCNN(self.device)
        elif self.face_detector_type == '3fabrec': 
            from gdl.utils.TFabRecLandmarkDetector import TFabRec
            self.face_detector = TFabRec(instantiate_detector='sfd', threshold=self.face_detector_threshold)
        elif self.face_detector_type == 'mediapipe': 
            from gdl.utils.MediaPipeLandmarkDetector import MediaPipeLandmarkDetector
            self.face_detector = MediaPipeLandmarkDetector(threshold=self.face_detector_threshold, 
                video_based=self._is_video_dataset(), max_faces=self._get_max_faces_per_image())
        elif self.face_detector_type == 'deep3dface': 
            from gdl.utils.Deep3DFaceLandmarkDetector import Deep3DFaceLandmarkDetector
            self.face_detector = Deep3DFaceLandmarkDetector(instantiate_detector='mtcnn')
        else:
            raise ValueError("Invalid face detector specifier '%s'" % self.face_detector)

    # @profile
    def _detect_faces_in_image(self, image_or_path, detected_faces=None):
        # imagepath = self.imagepath_list[index]
        # imagename = imagepath.split('/')[-1].split('.')[0]
        if isinstance(image_or_path, (str, Path)):
            image = np.array(imread(image_or_path))
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else: 
            raise ValueError("Invalid image type '%s'" % type(image_or_path)) 
            
        if len(image.shape) == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        self._instantiate_detector()
        bounding_boxes, bbox_type, landmarks = self.face_detector.run(image,
                                                                      with_landmarks=True,
                                                                      detected_faces=detected_faces)
        image = image / 255.
        detection_images = []
        detection_centers = []
        detection_sizes = []
        detection_landmarks = [] # landmarks wrt the detection image
        # original_landmarks = [] # landmarks wrt the original image
        original_landmarks = landmarks # landmarks wrt the original image
        # detection_embeddings = []
        if len(bounding_boxes) == 0:
            # print('no face detected! run original image')
            return detection_images, detection_centers, detection_images, \
                   bbox_type, detection_landmarks, original_landmarks
            # left = 0
            # right = h - 1
            # top = 0
            # bottom = w - 1
            # bounding_boxes += [[left, right, top, bottom]]

        for bi, bbox in enumerate(bounding_boxes):
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]
            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)

            center[0] += abs(right-left)*self.bb_center_shift_x
            center[1] += abs(bottom-top)*self.bb_center_shift_y

            size = int(old_size * self.scale)

            dst_image, dts_landmark = bbpoint_warp(image, center, size, self.image_size, landmarks=landmarks[bi])

            # dst_image = dst_image.transpose(2, 0, 1)
            #
            detection_images += [(dst_image*255).astype(np.uint8)]
            detection_centers += [center]
            detection_sizes += [size]

            # imsave(os.path.join("detection_%d.png" % bi), dst_image)

            # to be checked
            detection_landmarks += [dts_landmark]

        del image
        return detection_images, detection_centers, detection_sizes, bbox_type, detection_landmarks, original_landmarks

    # @profile
    def _detect_faces_in_image_wrapper(self, frame_list, fid, out_detection_folder, out_landmark_folder, bb_outfile,
                                       centers_all, sizes_all, detection_fnames_all, landmark_fnames_all, 
                                       ):

        if isinstance(frame_list, (str, Path, list)):\
            # if frame list is a list of image paths
            frame_fname = frame_list[fid]
            # detect faces in each frames
            detection_ims, centers, sizes, bbox_type, landmarks, orig_landmarks = self._detect_faces_in_image(Path(self.output_dir) / frame_fname)
        elif isinstance(frame_list, (np.ndarray, types.GeneratorType)): 
            # frame_list is an array of many images, or a generator (like a video reader)
            frame_fname =Path(f"{fid:05d}.png")
            if isinstance(frame_list, np.ndarray):
                frame = frame_list[fid]
            else:
                frame = next(frame_list)
            detection_ims, centers, sizes, bbox_type, landmarks, orig_landmarks = self._detect_faces_in_image(frame)
            # if len(detection_ims) > 0: # debug visualization
            #     imsave(frame_fname, detection_ims[0])
        
        # self.detection_lists[sequence_id][fid] += [detections]
        centers_all += [centers]
        sizes_all += [sizes]

        # save detections
        detection_fnames = []
        landmark_fnames = []
        for di, detection in enumerate(detection_ims):
            # save detection
            stem = frame_fname.stem + "_%.03d" % di
            if self.save_detection_images:
                out_detection_fname = out_detection_folder / (stem + self.processed_ext)
                detection_fnames += [out_detection_fname.relative_to(self.output_dir)]
                if self.processed_ext in ['.JPG', '.jpg', ".jpeg", ".JPEG"]:
                    imsave(out_detection_fname, detection, quality=100)
                else:
                    imsave(out_detection_fname, detection)
            # save landmarks
            if self.save_landmarks:
                if self.save_detection_images:
                    out_landmark_fname = out_landmark_folder / (stem + ".pkl")
                    landmark_fnames += [out_landmark_fname.relative_to(self.output_dir)]
                    save_landmark(out_landmark_fname, landmarks[di], bbox_type)
                else: 
                    out_landmark_fname = out_landmark_folder / (stem + ".pkl")
                    landmark_fnames += [out_landmark_fname.relative_to(self.output_dir)]
                    save_landmark(out_landmark_fname, orig_landmarks[di], bbox_type)

        detection_fnames_all += [detection_fnames]
        landmark_fnames_all += [landmark_fnames]

        torch.cuda.empty_cache()
        checkpoint_frequency = 100
        if fid % checkpoint_frequency == 0:
            FaceDataModuleBase.save_detections(bb_outfile, detection_fnames_all, landmark_fnames_all,
                                                centers_all, sizes_all, fid)

    def _segment_images(self, detection_fnames_or_ims, out_segmentation_folder, path_depth = 0, landmarks=None):
        import time

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        net, seg_type, batch_size = self._get_segmentation_net(device)

        if self.save_detection_images:
            ref_im = imread(detection_fnames_or_ims[0])
        else: 
            ref_im = detection_fnames_or_ims[0]
        ref_size = Resize((ref_im.shape[0], ref_im.shape[1]), interpolation=Image.NEAREST)

        # transforms = Compose([
        #     Resize((512, 512)),
        #     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
        transforms=None
        # batch_size = 16

        dataset = UnsupervisedImageDataset(detection_fnames_or_ims, image_transforms=transforms,
                                           landmark_list = landmarks,
                                           im_read='pil' if not isinstance(detection_fnames_or_ims[0], np.ndarray) else None)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        # import matplotlib.pyplot as plt

        for i, batch in enumerate(tqdm(loader)):
            # facenet_pytorch expects this stanadrization for the input to the net
            # images = fixed_image_standardization(batch['image'].to(device))
            images = batch['image'].cuda()
            # start = time.time()
            with torch.no_grad():
                segmentation = net(images)
            segmentation = segmentation.cpu()
            # end = time.time()
            segmentation = ref_size(segmentation)
            segmentation = segmentation.numpy()

            start = time.time()
            for j in range(segmentation.shape[0]):
                image_path = batch['path'][j]
                # if isinstance(out_segmentation_folder, list):
                if path_depth > 0:
                    rel_path = Path(image_path).parent.relative_to(Path(image_path).parents[path_depth])
                    segmentation_path = out_segmentation_folder / rel_path / (Path(image_path).stem + ".pkl")
                else:
                    segmentation_path = out_segmentation_folder / (Path(image_path).stem + ".pkl")
                segmentation_path.parent.mkdir(exist_ok=True, parents=True)
                # im = images[j]
                # im = im.transpose([1,2,0])
                # seg = process_segmentation(segmentation[j], seg_type)
                # imsave("seg.png", seg)
                # imsave("im.png", im)
                # FaceVideoDataModule.vis_parsing_maps(im, segmentation[j], stride=1, save_im=True,
                #                  save_path='overlay.png')
                # plt.figure()
                # plt.imshow(im)
                # plt.show()
                # plt.figure()
                # plt.imshow(seg)
                # plt.show()
                save_segmentation(segmentation_path, segmentation[j], seg_type)
            end = time.time()
            print(f" Saving batch {i} took: {end - start}")


    def _get_segmentation_net(self, device, method='gpen'):
        if method == 'bisenet':
            seg_type = 'face_parsing'
            from gdl.models.external.BiSeNetFaceParsing import BiSeNetFaceParsing
            net = BiSeNetFaceParsing()
            batch_size = 64
        elif method == "gpen": 
            seg_type = 'face_parsing'
            from gdl.models.external.GPENFaceParsing import GPENFaceParsing
            net = GPENFaceParsing()
            batch_size = 16
        else: 
            raise ValueError(f"Unknown segmentation type: {method}" )

        # from gdl.utils.other import get_path_to_externals
        # path_to_segnet = get_path_to_externals() / "face-parsing.PyTorch"
        # if not(str(path_to_segnet) in sys.path  or str(path_to_segnet.absolute()) in sys.path):
        #     sys.path += [str(path_to_segnet)]

        # from model import BiSeNet
        # n_classes = 19
        # net = BiSeNet(n_classes=n_classes)
        # # net.cuda()
        # save_pth = path_to_segnet / 'res' / 'cp' / '79999_iter.pth'
        # net.load_state_dict(torch.load(save_pth))
        # # net.eval()
        # net.eval().to(device)

        # labels = {
        #     0: 'background',
        #     1: 'skin',
        #     2: 'nose',
        #     3: 'eye_g',
        #     4: 'l_eye',
        #     5: 'r_eye',
        #     6: 'l_brow',
        #     7: 'r_brow',
        #     8: 'l_ear',
        #     9: 'r_ear',
        #     10: 'mouth',
        #     11: 'u_lip',
        #     12: 'l_lip',
        #     13: 'hair',
        #     14: 'hat',
        #     15: 'ear_r',
        #     16: 'neck_l',
        #     17: 'neck',
        #     18: 'cloth'
        # }

        return net, "face_parsing" , batch_size

    @staticmethod
    def save_detections(fname, detection_fnames, landmark_fnames, centers, sizes, last_frame_id):
        with open(fname, "wb" ) as f:
            pkl.dump(detection_fnames, f)
            pkl.dump(centers, f)
            pkl.dump(sizes, f)
            pkl.dump(last_frame_id, f)
            pkl.dump(landmark_fnames, f)

    @staticmethod
    def load_detections(fname):
        with open(fname, "rb" ) as f:
            detection_fnames = pkl.load(f)
            centers = pkl.load(f)
            sizes = pkl.load(f)
            try:
                last_frame_id = pkl.load(f)
            except:
                last_frame_id = -1
            try:
                landmark_fnames = pkl.load(f)
            except:
                landmark_fnames = [None]*len(detection_fnames)

        return detection_fnames, landmark_fnames, centers, sizes, last_frame_id