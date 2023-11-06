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
from main_test_swinir import define_model, test
# except ImportError: 
    # print("Could not import FaceGAN") 
    # del sys.path[0]


class SwinIR(ImageTranslationNetBase):

    def __init__(self, im_size, device='cuda', **kwargs):
        super().__init__()
        self.im_size = im_size
        args = Munch(**kwargs)

        tasks =  ['classical_sr', 'lightweight_sr', 'real_sr', 'gray_dn' , 'color_dn', 'jpeg_car']
        assert args.task in tasks,  f"Task {args.task} not supported. Supported tasks are: {args.tasks}"
        if 'dn' not in args.task:
            scales = [1, 2, 3, 4, 8]
        else: 
            scales = [1]

        assert args.scale in scales, f"Scale {args.scale} not supported for task '{args.task}'. Supported scales are: {scales}"


        # args = Munch() 
        # args.task = task 
        # args.scale = scale 
        # args.noise = 15 # noise level: 15, 25, 50
        # args.jpeg = 10 # 10, 20, 30, 40
        # args.training_patch_size = 128 # probably not important for testing
        # args.tile = None # probably not important for testing
        # self.tile_overlap = 32 # probably not important for testing
        # # args.large_model = task in ['classical_sr', 'lightweight_sr', 'real_sr']
        # args.large_model = task in ['real_sr']
        # args.window_size = 8

        args.window_size = 8
        if args.task == 'classical_sr':
            model_path = f"001_classicalSR_DF2K_s64w{args.window_size}_SwinIR-M_x{args.scale}.pth"
        elif args.task == 'lightweight_sr':
            model_path = f"002_lightweightSR_DIV2K_s64w{args.window_size}_SwinIR-S_x{args.scale}.pth"
        elif args.task == 'real_sr':
            model_path = f"003_realSR_BSRGAN_DFO_s64w{args.window_size}_SwinIR-{'L' if args.large_model else 'M'}_x{args.scale}_GAN.pth"
            # model_path = f"003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
            # model_path = f"003_realSR_BSRGAN_DFO_s64w{args.window_size}_SwinIR-M_x{scale}_PSNR.pth" 
        elif args.task == 'gray_dn':
            model_path = f"004_grayDN_DFWB_s128w{args.window_size}_SwinIR-M_noise{args.noise}.pth"
        elif args.task == 'color_dn':
            model_path = f"005_colorDN_DFWB_s128w{args.window_size}_SwinIR-M_noise{args.noise}.pth"
        elif args.task == 'jpeg_car':
            args.window_size = 7
            model_path = f"006_CAR_DFWB_s126w{args.window_size}_SwinIR-M_jpeg{args.jpeg}.pth"
        else:
            raise ValueError(f"Task {args.task} not supported.")
        model_path = path_to_gpen / "model_zoo" / model_path
        assert model_path.is_file(), f"Model file {model_path} not found."
        args.model_path = str(model_path)

        assert args.tile is None or args.tile % args.window_size == 0 , "tile size should be a multiple of window_size"

        self.network = define_model(args)
        self.network.eval()
        self.network = self.network.to(device)
        
        self.args = args

    @property
    def input_size(self):
        return (self.im_size, self.im_size)

    def forward(self, input_image, resize_to_input_size=False, pad_to_match_size=False):
        """
        input_image: torch.Tensor [b,c,w,h] in RGB,  range [0-1]
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

        translated_image = test(input_image, self.network, self.args, self.args.window_size)

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


class SwinIRCompressionArtifact(SwinIR):

    def __init__(self, im_size, device='cuda', jpeg=40, tile=None):
        task = "jpeg_car"
        scale = 1
        assert jpeg in [10, 20, 30, 40], "jpeg quality must be 10, 20, 30 or 40" 
        super().__init__(im_size, device=device, task=task, scale=scale, jpeg=jpeg, tile=tile)

    
    def forward(self, image, resize_to_input_size=False):        
        """
        input_image: torch.Tensor [b,c,w,h] in RGB,  range [0-1]
        """
        image = image * 255. 
        output_image = super().forward(image.reshape((image.shape[0]*image.shape[1], 1, image.shape[2], image.shape[3])), resize_to_input_size=resize_to_input_size)
        output_image = output_image.reshape(image.shape)
        # output_image = None
        # for c in range(image.shape[1]):
        #     out = super().forward(image[:, c:c+1, ...], resize_to_input_size=resize_to_input_size)
        #     if output_image is None: 
        #         # out_shape = out.shape[0], image.shape[1], out.shape[2], out.shape[3]
        #         output_image = torch.zeros_like(out).tile((1, image.shape[1], 1, 1))
        #     output_image[:, c:c+1, ...] += out
        output_image = output_image / 255. 
        return output_image


class SwinIRRealSuperRes(SwinIR):

    def __init__(self, im_size, device='cuda', tile=None, large_model=False):
        task = "real_sr"
        scale = 4
        super().__init__(im_size, device=device, task=task, scale=scale, tile=tile, large_model=large_model)
