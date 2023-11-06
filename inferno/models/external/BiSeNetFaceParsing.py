import torch 
from inferno.utils.other import get_path_to_externals
import os, sys
from inferno.models.ImageTranslationNetBase import ImageTranslationNetBase
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, Normalize
import numpy as np
from inferno.utils.other import get_path_to_externals
path_to_segnet = get_path_to_externals() / "face-parsing.PyTorch"
if not(str(path_to_segnet) in sys.path  or str(path_to_segnet.absolute()) in sys.path):
    sys.path += [str(path_to_segnet)]

from model import BiSeNet

MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
##MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]] = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [0, 0, 0], [0, 0, 0]]
# MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]


class BiSeNetFaceParsing(ImageTranslationNetBase):

    def __init__(self, device='cuda'):
        super().__init__()
        self.im_size = 512
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        save_pth = path_to_segnet / 'res' / 'cp' / '79999_iter.pth'
        self.net.load_state_dict(torch.load(save_pth))
        self.net.eval().to(device)

        self.transforms = Compose([
            Resize((512, 512)),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @property
    def input_size(self):
        return (self.im_size, self.im_size)

    def forward(self, input_image, resize_to_input_size=False):
        """
        input_image: torch.Tensor [b,c,w,h] in RGB format 0,1
        """

        input_shape = input_image.shape
        # # RGB to BGR
        # input_image = input_image[:, [2, 1, 0], :, :]

        # # normalization expected by GPEN 
        # input_image = input_image * 2 - 1

        # # if image size does not match, resize 
        # if input_image.shape[2] != self.im_size or input_image.shape[3] != self.im_size:
        #     input_image = F.interpolate(input_image, size=(self.im_size, self.im_size), 
        #     mode='bicubic', align_corners=False)

        # pred_mask, sr_img_tensor = self.network(input_image)
        # mask = pred_mask.argmax(dim=1)

        # # if resize_to_input_size:
        # #     # resize to input size, with nearest neighbor interpolation
        # #     restored_images_torch = F.interpolate(mask, size=(input_shape[2], input_shape[3]), 
        # #         mode='nearest', align_corners=False) 

        input_image = self.transforms(input_image)
        out = self.net(input_image)[0]
        if resize_to_input_size:
            # resize to input size, with nearest neighbor interpolation
            out = F.interpolate(out, size=(input_shape[2], input_shape[3]), 
                mode='bicubic', align_corners=False) 
        segmentation = out.cpu().argmax(1, keepdim=True)
        return segmentation

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
            for idx, color in enumerate( MASK_COLORMAP):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps