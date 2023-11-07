import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from insightface.utils import face_align
import cv2

input_mean = 127.5
input_std = 127.5


class MicaInputProcessor(object):

    def __init__(self, mode, crash_on_no_detection=False):
        super().__init__()
        assert mode in ['default', 'ported_insightface', 'none', 'fan'], "mode must be one of 'default', 'ported_insightface', 'none'"
        self.mode = mode
        self.crash_on_no_detection = crash_on_no_detection
        
        if mode is True or mode == 'default':
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(224, 224))
        elif mode == 'ported_insightface':
            from .FaceAnalysisAppTorch import FaceAnalysis as FaceAnalysisTorch
            self.app = FaceAnalysisTorch(name='antelopev2')
            self.app.prepare(det_size=(224, 224))
        elif mode == 'fan': 
            self.app = None
        
    def to(self, *args, device=None, **kwargs):
        if device is not None:
            if self.mode == 'ported_insightface':
                self.app.det_model = self.app.det_model.to(device)

    def __call__(self, input_image, fan_landmarks=None, landmarks_validity=None):
        batched = len(input_image.shape) == 4
        if not batched:
            input_image = input_image.unsqueeze(0)
        if self.mode in [True,  'default']:
            mica_image = self._dirty_image_preprocessing(input_image)
        elif self.mode == 'ported_insightface':
            mica_image = self._dirty_image_preprocessing(input_image)
        elif self.mode in [False, 'none']: 
            mica_image = F.interpolate(input_image, (112,112), mode='bilinear', align_corners=False)
        elif self.mode == 'fan':
            mica_image = self._fan_image_preprocessing(input_image, fan_landmarks, landmarks_validity=landmarks_validity)
        else: 
            raise ValueError(f"Invalid mica_preprocessing option: '{self.mode}'")
        if not batched:
            mica_image = mica_image.squeeze(0)
        return mica_image

    def _fan_image_preprocessing(self, input_image, fan_landmarks, landmarks_validity=None):
        # landmarks_torch = False
        if isinstance(fan_landmarks, torch.Tensor):
            fan_landmarks = fan_landmarks.detach().cpu().numpy()
            # landmarks_torch = True

        image_torch = False
        if isinstance(input_image, torch.Tensor):
            dev = input_image.device
            input_image = input_image.detach().cpu().numpy()
            # b,c,h,w to b,h,w,c
            input_image = input_image.transpose((0,2,3,1))
            image_torch = True

        is_float=False
        if input_image.dtype == np.float32:
            is_float=True
            input_image = (input_image * 255).astype(np.uint8)


        lmk51 = fan_landmarks[:, 17:, :]
        kpss = lmk51[:, [20, 27, 13, 43, 47], :]  # left eye, right eye, nose, left mouth, right mouth
        kpss[:, 0, :] = lmk51[:, [21, 24], :].mean(1)  # center of eye
        kpss[:, 1, :] = lmk51[:, [27, 29], :].mean(1)
        
        ## from [-1, 1] to [o, input_image_size]
        kpss = (kpss + 1) * (input_image.shape[1] / 2)

        B = input_image.shape[0]
        # norm_crop_images = norm_crop(input_image, torch.tensor(kpss), image_size=112, mode='arcface')
        
        # blobs = blob_from_tensor_images(norm_crop_images, scalefactor=1.0 / input_std, size=(112, 112), mean=(input_mean, input_mean, input_mean), swapRB=True)
        # blobs = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
        
        aligned_image_list = []
        for i in range(B): 
            if landmarks_validity is not None and landmarks_validity[i].item() == 0.:
                aimg = resize(input_image[i], output_shape=(112,112), preserve_range=True).astype(np.uint8)
            else:
                aimg = face_align.norm_crop(input_image[i], landmark=kpss[i])
            aligned_image_list.append(aimg)
            # blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=False)

            # # plot original image and aligned image
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(input_image[i])
            # plt.subplot(122)
            # plt.imshow(aimg)
            # plt.show()
        
        blob = cv2.dnn.blobFromImages(aligned_image_list, 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=False)
            
        
        # don't do this, blob is already returned as float
        # if is_float:
        #     blob = blob.astype(np.float32) / 255.
        
        if image_torch:
            # to torch to correct device 
            blob = torch.from_numpy(blob).to(dev)
        
        
        # if landmarks_torch:
        #     kpss = torch.from_numpy(kpss).to(input_image.device)
        #     # return aligned_images, kpss
        return blob

    def _dirty_image_preprocessing(self, input_image): 
        # breaks whatever gradient flow that may have gone into the image creation process
        from inferno.models.mica.detector import get_center, get_arcface_input
        from insightface.app.common import Face
        
        image = input_image.detach().clone().cpu().numpy() * 255. 
        # b,c,h,w to b,h,w,c
        image = image.transpose((0,2,3,1))
    
        min_det_score = 0.5
        image_list = list(image)
        aligned_image_list = []
        for i, img in enumerate(image_list):
            bboxes, kpss = self.app.det_model.detect(img, max_num=0, metric='default')
            if bboxes.shape[0] == 0:
                aimg = resize(img, output_shape=(112,112), preserve_range=True)
                aligned_image_list.append(aimg)
                if self.crash_on_no_detection:
                    raise RuntimeError("No faces detected")
                else: 
                    print("[WARNING] No faces detected in MicaInputProcessor")
                    continue
            i = get_center(bboxes, image)
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            # if det_score < min_det_score:
            #     continue
            kps = None
            if kpss is not None:
                kps = kpss[i]

            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            blob, aimg = get_arcface_input(face, img)
            aligned_image_list.append(aimg)
        aligned_images = np.array(aligned_image_list)
        # b,h,w,c to b,c,h,w
        aligned_images = aligned_images.transpose((0,3,1,2))
        # to torch to correct device 
        aligned_images = torch.from_numpy(aligned_images).to(input_image.device)
        return aligned_images
    

# # Assuming arcface_src and src_map are defined somewhere else in your code
# arcface_src = torch.tensor(face_align.arcface_src, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')
# src_map = torch.tensor(face_align.src_map, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')

# def estimate_norm(lmk_batch, image_size=112, mode='arcface'):
#     batch_size = lmk_batch.size(0)
#     assert lmk_batch.size(1) == 5 and lmk_batch.size(2) == 2
    
#     lmk_tran_batch = torch.cat((lmk_batch, torch.ones(batch_size, 5, 1).to(lmk_batch.device)), dim=2)

#     if mode == 'arcface':
#         if image_size == 112:
#             src = arcface_src
#         else:
#             src = float(image_size) / 112 * arcface_src
#     else:
#         src = src_map[image_size]

#     src = torch.tensor(src, dtype=torch.float32)
#     A = torch.cat([lmk_batch, torch.ones(batch_size, 5, 1).to(lmk_batch.device)], dim=2)
#     A = A.unsqueeze(1).repeat(1, src.size(0), 1, 1)

#     # Solve the least squares problem for each src[i]
#     M_lst, _, _, _ = torch.lstsq(src.view(-1, 1), A.view(-1, A.size(-1)))
#     M_lst = M_lst[:, 0].view(batch_size, src.size(0), 2, 3)
    
#     # Calculate results for each M
#     results = torch.einsum('bik,bkij->bkj', lmk_tran_batch, M_lst)
    
#     # Calculate errors
#     errors = torch.sum(torch.sqrt(torch.sum((results - src) ** 2, dim=2)), dim=2)
    
#     # Find min error and corresponding M for each item in the batch
#     min_error, min_indices = torch.min(errors, dim=1)
#     min_M = torch.stack([M_lst[i, min_indices[i]] for i in range(batch_size)])

#     return min_M, min_indices

# def norm_crop(img_batch, landmark_batch, image_size=112, mode='arcface'):
#     batch_size = img_batch.size(0)
#     M_batch, _ = estimate_norm(landmark_batch, image_size, mode)
#     theta_batch = M_batch
#     grid = torch.nn.functional.affine_grid(theta_batch, torch.Size((batch_size, img_batch.size(1), image_size, image_size)))
#     warped = torch.nn.functional.grid_sample(img_batch, grid)
#     return warped

# def blob_from_tensor_images(image_tensor, scalefactor=1.0, size=(224, 224), mean=(0, 0, 0), swapRB=False):
#     # Assuming image_tensor is of shape [N, H, W, C] or [N, C, H, W]

#     # Resize images
#     if size != (image_tensor.shape[2], image_tensor.shape[3]):
#         image_tensor = torch.nn.functional.interpolate(image_tensor, size=size, mode='bilinear', align_corners=False)
    
#     # Swap RB channels if needed
#     if swapRB:
#         image_tensor = image_tensor[:, [2, 1, 0], :, :]
    
#     # Subtract mean and scale
#     mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
#     image_tensor = (image_tensor - mean_tensor) * scalefactor

#     return image_tensor



class MicaDatasetWrapper(Dataset): 
    """
    This class is a wrapper around any dataset that returns images. It adds MICA preprocessing to 
    the dictionary returned by the dataset. 
    """
    
    def __init__(self, dataset, mica_preprocessing='ported_insightface', crash_on_no_detection=False):
        self.dataset = dataset
        self.mica_preprocessing = MicaInputProcessor(mica_preprocessing, crash_on_no_detection=crash_on_no_detection)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        data['mica_images'] = self.mica_preprocessing(data['image'])
        return data