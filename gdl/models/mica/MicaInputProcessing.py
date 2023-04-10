import torch
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize


class MicaInputProcessor(object):

    def __init__(self, mode):
        super().__init__()
        assert mode in ['default', 'ported_insightface', 'none'], "mode must be one of 'default', 'ported_insightface', 'none'"
        self.mode = mode
        
        if mode is True or mode == 'default':
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(224, 224))
        elif mode == 'ported_insightface':
            from .FaceAnalysisAppTorch import FaceAnalysis as FaceAnalysisTorch
            self.app = FaceAnalysisTorch(name='antelopev2')
            self.app.prepare(det_size=(224, 224))
        
    def to(self, *args, device=None, **kwargs):
        if device is not None:
            if self.mode == 'ported_insightface':
                self.app.det_model = self.app.det_model.to(device)

    def __call__(self, input_image):
        batched = len(input_image.shape) == 4
        if not batched:
            input_image = input_image.unsqueeze(0)
        if self.mode in [True,  'default']:
            mica_image = self._dirty_image_preprocessing(input_image)
        elif self.mode == 'ported_insightface':
            mica_image = self._dirty_image_preprocessing(input_image)
        elif self.mode in [False, 'none']: 
            mica_image = F.interpolate(input_image, (112,112), mode='bilinear', align_corners=False)
        else: 
            raise ValueError(f"Invalid mica_preprocessing option: '{self.mode}'")
        if not batched:
            mica_image = mica_image.squeeze(0)
        return mica_image

    def _dirty_image_preprocessing(self, input_image): 
        # breaks whatever gradient flow that may have gone into the image creation process
        from gdl.models.mica.detector import get_center, get_arcface_input
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
                raise RuntimeError("No faces detected")
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