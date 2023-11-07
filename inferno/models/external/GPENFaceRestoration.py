import torch 
from inferno.utils.other import get_path_to_externals
import os, sys
from inferno.models.ImageTranslationNetBase import ImageTranslationNetBase
import torch.nn.functional as F

# path_to_gpen = get_path_to_externals() / ".." / ".." / "GPEN" / "face_model"
path_to_gpen = get_path_to_externals() / "GPEN" / "face_model"

if str(path_to_gpen) not in sys.path:
    sys.path.insert(0, str(path_to_gpen))

# try:
from face_gan import FaceGAN
# except ImportError: 
    # print("Could not import FaceGAN") 
    # del sys.path[0]


class GPENFaceRestoration(ImageTranslationNetBase):

    def __init__(self, model_name, device='cuda'):
        super().__init__()
        if model_name == "GPEN-512": 
            self.im_size = 512
            self.model_name = "GPEN-BFR-512"
        elif model_name == 'GPEN-256': 
            self.im_size = 256
            self.model_name = "GPEN-BFR-256"
        else: 
            raise NotImplementedError()

        self.network = FaceGAN(str(path_to_gpen / "..") , self.im_size, model=self.model_name, 
            channel_multiplier=2, narrow=1, key=None, device=device) 

    @property
    def input_size(self):
        return (self.im_size, self.im_size)

    def forward(self, input_image, resize_to_input_size=False):
        """
        input_image: torch.Tensor [b,c,w,h] in RGB format 0,1
        """
        input_shape = input_image.shape
        # RGB to BGR
        input_image = input_image[:, [2, 1, 0], :, :]

        # normalization expected by GPEN 
        input_image = (input_image - 0.5) / 0.5

        # if image size does not match, resize 
        if input_image.shape[2] != self.im_size or input_image.shape[3] != self.im_size:
            input_image = F.interpolate(input_image, size=(self.im_size, self.im_size), 
            mode='bicubic', align_corners=False)

        restored_images_torch, _ = self.network.model(input_image)
        # denormalization 
        restored_images_torch = (restored_images_torch * 0.5 + 0.5) #.clamp(0, 1) * 255.

        if resize_to_input_size:
            # resize to input size 
            restored_images_torch = F.interpolate(restored_images_torch, size=(input_shape[2], input_shape[3]), 
                mode='bicubic', align_corners=False)

        # BGR to RGB
        restored_images_torch = restored_images_torch[:, [2, 1, 0], :, :]
        return restored_images_torch