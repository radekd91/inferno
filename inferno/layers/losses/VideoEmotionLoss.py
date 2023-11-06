import copy
import omegaconf
import torch
from inferno.layers.losses.EmonetLoader import get_emonet
from pathlib import Path
import torch.nn.functional as F
from inferno.models.video_emorec.VideoEmotionClassifier import VideoEmotionClassifier
from inferno.models.IO import get_checkpoint_with_kwargs
from inferno.utils.other import class_from_str
from .EmoNetLoss import create_emo_loss
from inferno.layers.losses.emotion_loss_loader import emo_network_from_path
import sys
from omegaconf import OmegaConf
from .Metrics import metric_from_str
from inferno.utils.other import get_path_to_assets


def load_video_emotion_recognition_net(network_path):
    model_config_path = Path(network_path) / "cfg.yaml"
    if not model_config_path.is_absolute():
        model_config_path = get_path_to_assets() / model_config_path
    # load config 
    model_config = OmegaConf.load(model_config_path)

    class_ = class_from_str(model_config.model.pl_module_class, sys.modules[__name__])

    # instantiate the model
    checkpoint_mode = 'best' # resuming in the same stage, we want to pick up where we left of
    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(
        model_config, "", 
        checkpoint_mode=checkpoint_mode,
        pattern="val"
        )

    network = class_.instantiate(model_config, None, None, checkpoint, checkpoint_kwargs)
    return network


def create_video_emotion_loss(cfg):
    model_config_path = Path(cfg.network_path) / "cfg.yaml"
    if not model_config_path.is_absolute():
        model_config_path = get_path_to_assets() / model_config_path
    # load config 
    model_config = OmegaConf.load(model_config_path)

    sequence_model = load_video_emotion_recognition_net(cfg.network_path)

    ## see if the model has a feature extractor
    feat_extractor_cfg = model_config.model.get('feature_extractor', None)

    # video_emotion_loss_cfg.network_path = str(Path(video_network_folder) / video_emotion_loss_cfg.video_network_name)
    # video_emotion_loss = create_video_emotion_loss( video_emotion_loss_cfg).to(device)
     
    # if feat_extractor_cfg is None and hasattr(sequence_model, 'feature_extractor_path'):
    if (feat_extractor_cfg is None or feat_extractor_cfg.type is False) and hasattr(cfg, 'feature_extractor_path'):
        # default to the affecnet trained resnet feature extractor
        feature_extractor_path = Path(cfg.feature_extractor_path)
        if not feature_extractor_path.is_absolute():
            feature_extractor_path = get_path_to_assets() / feature_extractor_path
        feature_extractor = emo_network_from_path(str(feature_extractor_path))
    elif cfg.feature_extractor == "no":
        feature_extractor = None
    else: 
        # feature_extractor_path = feat_extractor_cfg.path
        feature_extractor = None
    
    metric = metric_from_str(cfg.metric)
    loss = VideoEmotionRecognitionLoss(sequence_model, metric, feature_extractor)
    return loss


from .Metrics import get_metric
from .BarlowTwins import BarlowTwinsLossHeadless, BarlowTwinsLoss
from .Masked import MaskedLoss


class VideoEmotionRecognitionLoss(torch.nn.Module):

    def __init__(self, video_emotion_recognition : VideoEmotionClassifier, metric, feature_extractor=None, ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_emotion_recognition = video_emotion_recognition
        self.metric = metric

    def forward(self, input, target):
        raise NotImplementedError()

    # def _forward_input(self, 
    #     input_images=None, 
    #     input_emotion_features=None,
    #     mask=None,
    #     ):
    #     if input_images is not None:
    #         B, T = input_images.shape[:2]
    #     else: 
    #         B, T = input_emotion_features.shape[:2]
    #     # there is no need to keep gradients for input (even if we're finetuning, which we don't, it's the output image we'd wannabe finetuning on)
    #     with torch.no_grad():
    #         if input_emotion_features is None:
    #             feat_extractor_sample = {"image" : input_images.view(B*T, *input_images.shape[2:])}
    #             input_emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)
    #         # result_ = self.model.forward_old(images)
    #         if mask is not None:
    #             input_emotion_features = input_emotion_features * mask

    #         video_emorec_batch_input = {
    #             "gt_emo_feature": input_emotion_features,
    #         }
    #         video_emorec_batch_input = self.video_emotion_recognition(video_emorec_batch_input)

    #         input_emotion_feat = video_emorec_batch_input["pooled_sequence_feature"]
    #     return input_emotion_feat

    def _forward_input(self, 
        input_images=None, 
        input_emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        with torch.no_grad():
            return self.forward(input_images, input_emotion_features, mask, return_logits)

    def _forward_output(self, 
        output_images=None, 
        output_emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        return self.forward(output_images, output_emotion_features, mask, return_logits)

    def forward(self, 
        images=None, 
        emotion_features=None,
        mask=None,
        return_logits=False,
        ):
        assert images is not None or emotion_features is not None, \
            "One and only one of input_images or input_emotion_features must be provided"
        if images is not None:
            B, T = images.shape[:2]
        else: 
            B, T = emotion_features.shape[:2]
        if emotion_features is None:
            feat_extractor_sample = {"image" : images.view(B*T, *images.shape[2:])}
            emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)
            # result_ = self.model.forward_old(images)
        if mask is not None:
            emotion_features = emotion_features * mask

        video_emorec_batch = {
            "gt_emo_feature": emotion_features,
        }
        video_emorec_batch = self.video_emotion_recognition(video_emorec_batch)

        emotion_feat = video_emorec_batch["pooled_sequence_feature"]
        
        if return_logits:
            if "predicted_logits" in video_emorec_batch:
                predicted_logits = video_emorec_batch["predicted_logits"]
                return emotion_feat, predicted_logits
            logit_list = {}
            if "predicted_logits_expression" in video_emorec_batch:
                logit_list["predicted_logits_expression"] = video_emorec_batch["predicted_logits_expression"]
            if "predicted_logits_intensity" in video_emorec_batch:
                logit_list["predicted_logits_intensity"] = video_emorec_batch["predicted_logits_intensity"]
            if "predicted_logits_identity" in video_emorec_batch:
                logit_list["predicted_logits_identity"] = video_emorec_batch["predicted_logits_identity"]
            return emotion_feat, logit_list

        return emotion_feat


    def compute_loss(
        self, 
        input_images=None, 
        input_emotion_features=None,
        output_images=None, 
        output_emotion_features=None,
        mask=None, 
        return_logits=False,
        ):
        # assert input_images is not None or input_emotion_features is not None, \
        #     "One and only one of input_images or input_emotion_features must be provided"
        # assert output_images is not None or output_emotion_features is not None, \
        #     "One and only one of output_images or output_emotion_features must be provided"
        # # assert mask is None, "Masked loss not implemented for video emotion recognition"
        # if input_images is not None:
        #     B, T = input_images.shape[:2]
        # else: 
        #     B, T = input_emotion_features.shape[:2]

        # if input_emotion_features is None:
        #     feat_extractor_sample = {"image" : input_images.view(B*T, *input_images.shape[2:])}
        #     input_emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)
        # if output_emotion_features is None:
        #     feat_extractor_sample = {"image" : output_images.view(B*T, *output_images.shape[2:])}
        #     output_emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)

        # if mask is not None:
        #     input_emotion_features = input_emotion_features * mask
        #     output_emotion_features = output_emotion_features * mask

        # video_emorec_batch_input = {
        #     "gt_emo_feature": input_emotion_features,
        # }
        # video_emorec_batch_input = self.video_emotion_recognition(video_emorec_batch_input)

        # video_emorec_batch_output = {
        #     "gt_emo_feature": output_emotion_features,
        # }
        # video_emorec_batch_output = self.video_emotion_recognition(video_emorec_batch_output)

        # input_emotion_feat = video_emorec_batch_input["pooled_sequence_feature"]
        # output_emotion_feat = video_emorec_batch_output["pooled_sequence_feature"]

        if return_logits:
            input_emotion_feat, in_logits = self._forward_input(input_images, input_emotion_features, mask, return_logits=return_logits)
            output_emotion_feat, out_logits = self._forward_output(output_images, output_emotion_features, mask, return_logits=return_logits)
            return self._compute_feature_loss(input_emotion_feat, output_emotion_feat), in_logits, out_logits

        input_emotion_feat = self._forward_input(input_images, input_emotion_features, mask)
        output_emotion_feat = self._forward_output(output_images, output_emotion_features, mask)
        return self._compute_feature_loss(input_emotion_feat, output_emotion_feat)


    def _compute_feature_loss(self, input_emotion_feat, output_emotion_feat):
        loss = self.metric(input_emotion_feat, output_emotion_feat)
        # for i in range(input_emotion_feat.shape[0]):
        #     print("In:\t", input_emotion_feat[i:i+1,:5]) 
        #     print("Out:\t", output_emotion_feat[i:i+1,:5]) 
        #     print(self.metric(input_emotion_feat[i:i+1], output_emotion_feat[i:i+1]))
        return loss