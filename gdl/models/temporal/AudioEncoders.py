
import sys
import torch
from gdl.models.temporal.Bases import TemporalAudioEncoder
from gdl.utils.other import get_path_to_externals
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

    def forward(self, sample, train=False): 
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



from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Config


class Wav2Vec2Encoder(TemporalAudioEncoder):

    def __init__(self, model_specifier, trainable):
        super().__init__() 
        self.model_specifier = model_specifier
        self.cfg  =  Wav2Vec2Config.from_pretrained(model_specifier)
        self.input_processor = Wav2Vec2Processor.from_pretrained(model_specifier)
        self.model = Wav2Vec2Model.from_pretrained(model_specifier)
        self.trainable = trainable
        if not trainable: 
            self.model.requires_grad_(False)

    def get_trainable_parameters(self): 
        if self.trainable:
            return list(self.model.parameters())
        return []

    def _forward(self, sample, train=False): 
        B = sample["raw_audio"].shape[0]
        T = sample["raw_audio"].shape[1]
        # proc = self.input_processor(sample["raw_audio"], sampling_rate=sample["samplerate"], return_tensors="pt")[0]
        # raw_audio = sample["raw_audio"].view( B, -1)
        raw_audio = sample["raw_audio"].view( B, -1)
        proc = self.input_processor(raw_audio, sampling_rate=sample["samplerate"][0], return_tensors="pt")
        feats_ = self.model(proc.input_values[0].to(device=raw_audio.device))
        F = feats_.last_hidden_state.shape[-1]
        T2 = feats_.last_hidden_state.shape[1]

        assert T2 + 1  == 2*T # Wav2Vec doubles the feature dimensionality and then this is reduced by 1 
        # (probably because of a temporal convolution window of 3) 

        # feats = torch.zeros((B, T2 + 1, F),
        #     device=feats_.last_hidden_state.device, dtype=feats_.last_hidden_state.dtype)
        # feats[:,:T2, :] = feats_.last_hidden_state
        
        # feats = torch.zeros((B, T2, F),
        #     device=feats_.last_hidden_state.device, dtype=feats_.last_hidden_state.dtype)
        padding = torch.zeros((B, 1, F),
            device=feats_.last_hidden_state.device, dtype=feats_.last_hidden_state.dtype)

        feats = torch.cat((feats_.last_hidden_state, padding), dim=1).contiguous()

        # TODO: question. The sequence length seems to have doubled. Why? Should I subsample the temporal dimension (i.e. interpolate) or should I reshape?
        # 1) reshape
        feats = feats.view(B, T, -1)
        # 2) subsample - dummy version, only works when T2(+1) is a multiple of T
        # feats = feats[:,::2,:]
        sample["audio_feature"] = feats 
        return sample

    def train(self, mode: bool = True):
        # return super().train(mode)
        mode = mode and self.trainable 
        self.model.train(mode)
        return self

    def forward(self, sample, train=False): 
        if self.trainable:
            return self._forward(sample, train=train)
        else: 
            with torch.no_grad(): 
                return self._forward(sample, train=train)

    def output_feature_dim(self):
        return self.cfg.hidden_size * 2