from inferno.utils.other import get_path_to_externals
path_to_av_hubert = get_path_to_externals() / "av_hubert"
# path_to_av_hubert = get_path_to_externals() / "av_hubert" / "avhubert"
# path_to_fairseq = get_path_to_externals() / "av_hubert" / "fairseq"
if str(path_to_av_hubert) not in sys.path:
    sys.path.insert(0, str(path_to_av_hubert))
import avhubert


class AvHubertAudioEncoder(TemporalAudioEncoder):

    def __init__(self, avhubert_model, trainable):
        super().__init__()
        assert isinstance(avhubert_model, avhubert.AVHubertSeq2Seq)
        self.avhubert = avhubert_model.encoder.w2v_model 
        del self.avhubert.feature_extractor_video 
        del self.avhubert.encoder # the transformed encoder will not be used here
        del self.avhubert.final_proj # figure out what to do with this one
        self.trainable = trainable

    def forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        audio = sample["audio"]
        # B = audio.shape[0]
        # T = audio.shape[1] 
        # audio = audio.view(B * T, -1)
        audio_in = audio.transpose(1, 2)
        if self.trainable:
            features = self.avhubert.forward_features(audio_in, modality="audio")
        else: 
            with torch.no_grad(): 
                features = self.avhubert.forward_features(audio_in, modality="audio")
        # features = features.view(B, T, -1)
        features = features.transpose(1, 2)                
        sample["audio_feature"] = features
        return sample

    def train(self, mode: bool = True):
        # return super().train(mode)
        mode = mode and self.trainable 
        self.avhubert.train(mode)
        return self

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.avhubert.parameters())
        return []

    def output_feature_dim(self):
        return self.avhubert.feature_extractor_audio.proj.out_features


