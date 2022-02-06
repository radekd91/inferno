import torch 
from gdl.utils.other import get_path_to_externals
import os, sys
from gdl.models.ImageTranslationNetBase import ImageTranslationNetBase
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, Normalize

from gdl.utils.other import get_path_to_externals
path_to_segnet = get_path_to_externals() / "face-parsing.PyTorch"
if not(str(path_to_segnet) in sys.path  or str(path_to_segnet.absolute()) in sys.path):
    sys.path += [str(path_to_segnet)]

from model import BiSeNet


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

        # input_shape = input_image.shape
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
        segmentation = out.cpu().argmax(1)
        return self.net(input_image)[0]
