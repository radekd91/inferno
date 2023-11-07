import torch 
from inferno.utils.other import get_path_to_externals
import os, sys
from inferno.models.ImageTranslationNetBase import ImageTranslationNetBase
import torch.nn.functional as F
from torchvision.transforms import Resize, Compose, Normalize
import numpy as np
from inferno.utils.other import get_path_to_externals, get_path_to_assets
from munch import Munch
path_to_focus = get_path_to_externals() / "FOCUS"
if not(str(path_to_focus) in sys.path  or str(path_to_focus.absolute()) in sys.path):
    sys.path += [str(path_to_focus)]

from FOCUS_model.FOCUS_basic import FOCUSmodel

# MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
# ##MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]] = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [0, 0, 0], [0, 0, 0]]
# # MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]


class FocusSegmentation(ImageTranslationNetBase):

    def __init__(self, model=None, device='cuda'):
        super().__init__()
        # model = model or "CelebAHQ"
        model = model or "NoW"
        assert model in ["CelebAHQ", "NoW"]
        args = Munch()
        if model == "CelebAHQ":
            encnet_path = path_to_focus / "MoFA_UNet_Save/MoFA_UNet_CelebAHQ/enc_net_200000.model"
            unet_path = path_to_focus / "MoFA_UNet_Save/MoFA_UNet_CelebAHQ/unet_200000.model"
        else: 
            encnet_path = path_to_focus / "MoFA_UNet_Save/For_NoW_Challenge/enc_net_210000.model"
            unet_path = path_to_focus / "MoFA_UNet_Save/For_NoW_Challenge/unet_210000.model"
        # bfm_path = get_path_to_assets() / "BFM_2017" / "model2017-1_face12_nomouth.h5"
        bfm_path = get_path_to_assets() / "BFM_2017" / "model2017-1_bfm_nomouth.h5" 
        assert bfm_path.exists(), f"BFM_2017 model not found in assets: {bfm_path}"
        args.model_path = str(bfm_path)
        args.pretrained_encnet_path = str(encnet_path)
        args.pretrained_unet_path = str(unet_path)
        args.width = 224 
        args.height = 224
        args.device = device
        args.where_occmask = 'unet'
        self.net = FOCUSmodel(args)
        self.net.init_for_now_challenge()

        self.transforms = Compose([
            Resize((args.width, args.height)), 
        ])

        self._fix_upsample_crash()

    def _fix_upsample_crash(self): 
        # fixes a pytorch bug, which crashes the forward pass on Upsample modules due to a missing member 
        module_list = list(self.net.unet_for_mask.modules()) + list(self.net.enc_net.modules())
        for module in module_list: 
            if isinstance(module, torch.nn.Upsample): 
                setattr(module, "recompute_scale_factor", None)
                # module.recompute_scale_factor = False

    @property
    def input_size(self):
        return (self.im_size, self.im_size)

    def forward(self, input_image, resize_to_input_size=False, return_other_results=False):
        """
        input_image: torch.Tensor [b,c,w,h] in RGB format 0,1
        """
        input_image = self.transforms(input_image)
        data = {'img' : input_image}
        data= self.net.data_to_device(data)

        # lm=data['landmark']
        # images=data['img']
        # image_paths = data['filename'] 
        
        reconstructed_results = self.net.forward_intactfaceshape_NOW(data)
        occlusion_fg_mask = reconstructed_results['est_mask']
        # image_results = reconstructed_results['imgs_fitted']
        # raster_mask = reconstructed_results['raster_masks']
        # lmring = reconstructed_results['lm_NoW']
        # vertex_3d = reconstructed_results['nonexp_intact_verts']

        # use plotly to visualize the the input image, image_results and occlusion_fg_mask
        # import plotly.graph_objects as go
        # import plotly.express as px
        # import plotly.io as pio

        # # plot the image
        # raster_mask_repeated = raster_mask[:, None, ...].repeat(1,3,1,1)
        # occlusion_fg_mask_repeated = occlusion_fg_mask.repeat(1,3,1,1)
        # result_concatenated = torch.cat((input_image, image_results, raster_mask_repeated, occlusion_fg_mask_repeated), axis=3)

        # fig = px.imshow(result_concatenated[0].permute(1,2,0).cpu().numpy())
        # fig.show()
        if return_other_results:
            return occlusion_fg_mask, reconstructed_results

        return occlusion_fg_mask

