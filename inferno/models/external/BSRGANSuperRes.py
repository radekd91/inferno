import torch 
from inferno.utils.other import get_path_to_externals
import os, sys
from inferno.models.ImageTranslationNetBase import ImageTranslationNetBase
import torch.nn.functional as F
from munch import Munch

# path_to_gpen = get_path_to_externals() / ".." / ".." / "KAIR" #/ "models"
path_to_gpen = get_path_to_externals() / "KAIR" #/ "models"
if str(path_to_gpen) not in sys.path:
    sys.path.insert(0, str(path_to_gpen))

# try:
# from network_swinir import SwinIR
# from main_test_swinir import define_model, test
from models.network_rrdbnet import RRDBNet
# except ImportError: 
    # print("Could not import FaceGAN") 
    # del sys.path[0]


class BSRImageTranslation(ImageTranslationNetBase):

    def __init__(self, im_size, model_name="BSRGAN", device='cuda',  **kwargs):
        super().__init__()
        self.im_size = im_size

        accepted_model_names = ["BSRGAN",'BSRGANx2', 'RRDB','ESRGAN','FSSR_DPED','FSSR_JPEG','RealSR_DPED', 'RealSR_JPEG']

        assert  model_name in accepted_model_names , "model_name must be one of {}".format(accepted_model_names)
            
        model_path = path_to_gpen / "model_zoo" / (model_name + ".pth")
        assert model_path.is_file(), f"Model file {model_path} not found."

        sf = 4
        if model_name in ['BSRGANx2']:
            sf = 2

        # logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # torch.cuda.set_device(0)      # set GPU ID
        # logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        # torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        self.model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

#            model_old = torch.load(model_path)
#            state_dict = model.state_dict()
#            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
#                state_dict[key2] = param
#            model.load_state_dict(state_dict, strict=True)

        self.model.load_state_dict(torch.load(str(model_path)), strict=True)
        self.model.eval()
        # for k, v in model.named_parameters():
            # v.requires_grad = False
        self.model = self.model.to(device)


    @property
    def input_size(self):
        return (self.im_size, self.im_size)

    def forward(self, input_image, resize_to_input_size=False, pad_to_match_size=False):
        """
        input_image: torch.Tensor [b,c,w,h] in RGB, range [0-1]
        """
        input_shape = input_image.shape
        # # RGB to BGR
        # input_image = input_image[:, [2, 1, 0], :, :]

        # pad input image to be a multiple of window_size
        if pad_to_match_size:
            _, _, h_old, w_old = input_image.size()
            h_pad = (h_old // self.args.window_size + 1) * self.args.window_size - h_old
            w_pad = (w_old // self.args.window_size + 1) * self.args.window_size - w_old
            input_image = torch.cat([input_image, torch.flip(input_image, [2])], 2)[:, :, :h_old + h_pad, :]
            input_image = torch.cat([input_image, torch.flip(input_image, [3])], 3)[:, :, :, :w_old + w_pad]

        translated_image = self.model(input_image)

        # unpad the image 
        if pad_to_match_size:
            translated_image = translated_image[..., :h_old * self.args.scale, :w_old * self.args.scale]

        if resize_to_input_size:
            # resize to input size 
            translated_image = F.interpolate(translated_image, size=(input_shape[2], input_shape[3]), 
                mode='bicubic', align_corners=False)

        # # BGR to RGB
        # restored_images_torch = restored_images_torch[:, [2, 1, 0], :, :]
        return translated_image


class BSRSuperRes(BSRImageTranslation):

    def __init__(self, im_size, scale=4, device='cuda'):
        assert scale in [2,4], "BSRSuperRes scale must be 2 or 4"
        model_name = "BSRGANx2" if scale == 2 else "BSRGAN"
        super().__init__(im_size, device=device, model_name=model_name)

    