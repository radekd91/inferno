from gdl.utils.other import get_path_to_externals
from pathlib import Path
import sys
import torch

path_to_ext = str(get_path_to_externals())
if path_to_ext not in sys.path:
    sys.path.insert(0, path_to_ext)

path_to_lipreading = str(Path(path_to_ext) / "spectre" / "external" / "Visual_Speech_Recognition_for_Multiple_Languages")
if path_to_lipreading not in sys.path:
    sys.path.insert(0, path_to_lipreading)

try:
    from configparser import ConfigParser
    from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading
    from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
    import espnet
except ImportError as e:
    print("Error: Lipreading model not found. Please install the Visual_Speech_Recognition_for_Multiple_Languages package.")    
    # print the error message of e
    print(e)


class LipReadingNet(torch.nn.Module):

    def __init__(self, device): 
        super().__init__()
        cfg_path = get_path_to_externals() / "spectre" / "configs" / "lipread_config.ini"
        config = ConfigParser()
        config.read(cfg_path)

        model_path = str(get_path_to_externals() / "spectre" / config.get("model","model_path"))
        model_conf = str(get_path_to_externals() / "spectre" / config.get("model","model_conf"))
        
        config.set("model", "model_path", model_path)
        config.set("model", "model_conf", model_conf)

        self.lip_reader = Lipreading(
            config,
            device=device
        )
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()]
        )


    def forward(self, lip_images):
        channel_dim = 1
        lip_images = self.mouth_transform(lip_images.squeeze(channel_dim)).unsqueeze(channel_dim)
        lip_features = self.lip_reader.model.encoder(
            lip_images,
            None,
            extract_resnet_feats=True
        )
        return lip_features


class LipReadingLoss(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.model = LipReadingNet(device)
        self.model.eval()
        # freeze model
        for param in self.parameters(): 
            param.requires_grad = False

    def _forward_input(self, images):
        # there is no need to keep gradients for input (even if we're finetuning, which we don't, it's the output image we'd wannabe finetuning on)
        with torch.no_grad():
            result = self.model(images)
        return result

    def _forward_output(self, images):
        return self.model(images)

    def compute_loss(self, mouth_images_gt, mouth_images_pred):
        lip_features_gt = self._forward_input(mouth_images_gt)
        lip_features_pred = self._forward_output(mouth_images_pred)

        lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
        lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
        
        lr = (lip_features_gt*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_gt,dim=1)

        loss = 1-torch.mean(lr)
        return loss

