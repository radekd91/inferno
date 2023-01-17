import copy
import omegaconf
import torch
from gdl.layers.losses.EmonetLoader import get_emonet
from pathlib import Path
import torch.nn.functional as F
from gdl.models.video_emorec.VideoEmotionClassifier import VideoEmotionClassifier
from gdl.models.IO import get_checkpoint_with_kwargs
from gdl.utils.other import class_from_str
from .EmoNetLoss import create_emo_loss
from gdl.layers.losses.emotion_loss_loader import emo_network_from_path
import sys
from omegaconf import OmegaConf
from .Metrics import metric_from_str


def create_video_emotion_loss(cfg):
    from gdl.models.video_emorec.VideoEmotionClassifier import VideoEmotionClassifier 
    
    model_config_path = Path(cfg.network_path) / "cfg.yaml"
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

    sequence_model = class_.instantiate(model_config, None, None, checkpoint, checkpoint_kwargs)

    ## see if the model has a feature extractor
    feat_extractor_cfg = model_config.model.get('feature_extractor', None) 
    if feat_extractor_cfg is None:
        # default to the affecnet trained resnet feature extractor
        feature_extractor_path = cfg.feature_extractor_path
        feature_extractor = emo_network_from_path(feature_extractor_path)
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

    def __init__(self, video_emotion_recognition, metric, feature_extractor=None, ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_emotion_recognition = video_emotion_recognition
        self.metric = metric

    def forward(self, input, target):
        raise NotImplementedError()

    def compute_loss(
        self, 
        input_images=None, 
        input_emotion_features=None,
        output_images=None, 
        output_emotion_features=None,
        mask=None
        ):
        assert input_images is not None or input_emotion_features is not None, \
            "One and only one of input_images or input_emotion_features must be provided"
        assert output_images is not None or output_emotion_features is not None, \
            "One and only one of output_images or output_emotion_features must be provided"
        # assert mask is None, "Masked loss not implemented for video emotion recognition"
        if input_images is not None:
            B, T = input_images.shape[:2]
        else: 
            B, T = input_emotion_features.shape[:2]

        if input_emotion_features is None:
            feat_extractor_sample = {"image" : input_images.view(B*T, *input_images.shape[2:])}
            input_emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)
        if output_emotion_features is None:
            feat_extractor_sample = {"image" : output_images.view(B*T, *output_images.shape[2:])}
            output_emotion_features = self.feature_extractor(feat_extractor_sample)['emo_feat_2'].view(B, T, -1)

        if mask is not None:
            input_emotion_features = input_emotion_features * mask
            output_emotion_features = output_emotion_features * mask

        video_emorec_batch_input = {
            "gt_emo_feature": input_emotion_features,
        }
        video_emorec_batch_input = self.video_emotion_recognition(video_emorec_batch_input)

        video_emorec_batch_output = {
            "gt_emo_feature": output_emotion_features,
        }
        video_emorec_batch_output = self.video_emotion_recognition(video_emorec_batch_output)

        input_emotion_feat = video_emorec_batch_input["pooled_sequence_feature"]
        output_emotion_feat = video_emorec_batch_output["pooled_sequence_feature"]

        loss = self.metric(input_emotion_feat, output_emotion_feat)
        return loss
