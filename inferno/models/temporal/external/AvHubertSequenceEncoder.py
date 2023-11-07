from ..SequenceEncoders import SequenceEncoder
from inferno.utils.other import get_path_to_externals
path_to_av_hubert = get_path_to_externals() / "av_hubert"
# path_to_av_hubert = get_path_to_externals() / "av_hubert" / "avhubert"
# path_to_fairseq = get_path_to_externals() / "av_hubert" / "fairseq"
if str(path_to_av_hubert) not in sys.path:
    sys.path.insert(0, str(path_to_av_hubert))
import avhubert


def load_avhubert_model(ckpt_path):
    from fairseq import checkpoint_utils, options, tasks, utils
    if str(path_to_av_hubert) not in sys.path:
        sys.path.insert(0, str(path_to_av_hubert))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    #   models = [model.eval().cuda() for model in models]
    return models, saved_cfg, task


class AvHubertSequenceEncoder(SequenceEncoder): 

    def __init__(self, avhubert_model):
        super().__init__()

        assert isinstance(avhubert_model, avhubert.AVHubertSeq2Seq)
        self.avhubert = avhubert_model.encoder.w2v_model 
        del self.avhubert.feature_extractor_video 
        del self.avhubert.feature_extractor_audio
        # del self.avhubert.encoder # the transformed encoder will not be used here
        del self.avhubert.final_proj # figure out what to do with this one

        self.trainable = True
        # self.avhubert = avhubert_model.encoder. 

    def input_feature_dim(self):
        return self.avhubert.embed

    def output_feature_dim(self):
        return self.avhubert.encoder_embed_dim

    def forward(self, sample, train=True):

        target_list = None # TODO: figure out what this is for

        input_feature = sample["fused_feature"]

        features = self.avhubert.layer_norm(input_feature)
        # features = features.transpose(1, 2)
        
        padding_mask = sample.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = self.avhubert.forward_padding_mask(features, padding_mask)

        if self.avhubert.post_extract_proj is not None:
            features = self.avhubert.post_extract_proj(features)

        features = self.avhubert.dropout_input(features)

        # mask = train
        mask = False 
        # this is avhubert's piece of code for the masking. They do in network after the modality feature 
        # extraction - they drop stuff then (make features zero or potentially something more elaborate?) 
        # for randomized segments of time. 
        # I think it's best to disable this for now. We will do data masking (potentially more elaborate)
        # inside the data augmentation process.
        if self.avhubert.masking_type == 'feature' and mask:
            x, mask_indices = self.avhubert.apply_feature_mask(features, padding_mask, target_list)
        else:
            x = features

        output_layer = None
        x, _  = self.avhubert.encoder(
            features,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )
        # proj_x = self.avhubert.final_proj(x)

        sample["seq_encoder_output"] = x
        return sample

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.parameters())
        return []
