import torch 
from gdl.utils.other import get_path_to_externals
import os, sys
from gdl.models.ImageTranslationNetBase import ImageTranslationNetBase
import torch.nn.functional as F

# path_to_gpen = get_path_to_externals() / ".." / ".." / "GPEN" / "face_model"
path_to_gpen = get_path_to_externals() / "GPEN" / "face_parse"

if str(path_to_gpen) not in sys.path:
    sys.path.insert(0, str(path_to_gpen))

# try:
from face_parsing import ParseNet
# except ImportError: 
    # print("Could not import FaceGAN") 
    # del sys.path[0]

class GPENFaceParsing(ImageTranslationNetBase):

    def __init__(self, device='cuda'):
        super().__init__()
        self.im_size = 512
        self.model_name = "ParseNet-latest"

        self.network = ParseNet(self.im_size, self.im_size, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
        self.network = self.network.to(device=device) 
        self.network.eval()

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
        input_image = input_image * 2 - 1

        # if image size does not match, resize 
        if input_image.shape[2] != self.im_size or input_image.shape[3] != self.im_size:
            input_image = F.interpolate(input_image, size=(self.im_size, self.im_size), 
            mode='bicubic', align_corners=False)

        pred_mask, sr_img_tensor = self.network(input_image)
        mask = pred_mask.argmax(dim=1)

        if resize_to_input_size:
            # resize to input size, with nearest neighbor interpolation
            mask = F.interpolate(mask, size=(input_shape[2], input_shape[3]), 
                mode='nearest', align_corners=False) 

        return mask
