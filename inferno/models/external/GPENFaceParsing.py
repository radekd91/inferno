import torch 
from inferno.utils.other import get_path_to_externals
import os, sys
from inferno.models.ImageTranslationNetBase import ImageTranslationNetBase
import torch.nn.functional as F
import numpy as np

# path_to_gpen = get_path_to_externals() / ".." / ".." / "GPEN" / "face_model"
path_to_gpen = get_path_to_externals() / "GPEN" / "face_parse"

if str(path_to_gpen) not in sys.path:
    sys.path.insert(0, str(path_to_gpen))

# try:
from face_parsing import ParseNet
# except ImportError: 
    # print("Could not import FaceGAN") 
    # del sys.path[0]

MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
##MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]] = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [0, 0, 0], [0, 0, 0]]
# MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]

class GPENFaceParsing(ImageTranslationNetBase):

    def __init__(self, device='cuda'):
        super().__init__()
        self.im_size = 512
        self.model_name = "ParseNet-latest"
        mfile =  path_to_gpen / ".." / 'weights' / (self.model_name+'.pth')
        self.network = ParseNet(self.im_size, self.im_size, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
        self.network.load_state_dict(torch.load(str(mfile)))
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
        
        # # RGB to BGR - dont
        # # input_image = input_image[:, [2, 1, 0], :, :]

        # normalization expected by GPEN 
        input_image = input_image * 2 - 1

        # if image size does not match, resize 
        if input_image.shape[2] != self.im_size or input_image.shape[3] != self.im_size:
            input_image = F.interpolate(input_image, size=(self.im_size, self.im_size), 
                mode='bicubic', align_corners=False)

        pred_mask, sr_img_tensor = self.network(input_image)

        if resize_to_input_size:
            # resize to input size, with nearest neighbor interpolation
            pred_mask = F.interpolate(pred_mask, size=(input_shape[2], input_shape[3]), 
                mode='bicubic', align_corners=False)

        mask = pred_mask.argmax(dim=1, keepdim=True)
        return mask

    def tensor2mask(self, tensor):
        if len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] > 1:
            tensor = tensor.argmax(dim=1) 

        tensor = tensor.squeeze(1).data.cpu().numpy()
        color_maps = []
        for t in tensor:
            tmp_img = np.zeros(tensor.shape[1:] + (3,))
            # tmp_img = np.zeros(tensor.shape[1:])
            for idx, color in enumerate(MASK_COLORMAP):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps