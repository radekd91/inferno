import numpy as np
import torch
from .AffectNetDataModule import AffectNetExpressions
from .VideoDatasetBase import VideoDatasetBase


class ConditionedVideoTestDatasetWrapper(torch.utils.data.Dataset): 

    def __init__(self,
                 dataset : VideoDatasetBase,
                 condition_source, 
                 condition_settings, 
                 key_prefix = "",
                 conditioned_intensity = None, # defaults to 3 if needed and not provided
                 ):    
        self.dataset = dataset
        self.condition_source = condition_source or "original"
        self.condition_settings = condition_settings or None
        self.expand_temporal = True
        self.condition_prefix = key_prefix
        
        if self.condition_source == "original":
            self.condition_settings = None
        elif self.condition_source == "expression":
            if self.condition_settings is None: 
                self.condition_settings = list(range(8)) 
            
            for i, cond in enumerate(self.condition_settings):
                if isinstance(cond, str):
                    self.condition_settings[i] = AffectNetExpressions[cond]
                    if self.condition_settings[i] is None or self.condition_settings[i] > 7:
                        raise ValueError(f"Invalid basic expression {cond}")
            assert isinstance(self.condition_settings, list), "Condition_settings must be a list of integers"

        elif self.condition_source in ["gt_expression", "gt_expression_intensity", "gt_expression_intensity_identity"]:
            if self.condition_settings is None: 
                self.condition_settings = list(range(8)) 
            
            for i, cond in enumerate(self.condition_settings):
                if isinstance(cond, str):
                    self.condition_settings[i] = AffectNetExpressions[cond]
                    if self.condition_settings[i] is None or self.condition_settings[i] > 7:
                        raise ValueError(f"Invalid basic expression {cond}")
                    
            self.conditioned_intensity = conditioned_intensity or 3
            assert isinstance(self.condition_settings, list), "Condition_settings must be a list of integers"

        elif self.condition_source == "valence_arousal":
            if isinstance(self.condition_settings, list):
                self.valence = np.array([self.condition_settings[0]])
                self.arousal = np.array([self.condition_settings[1]])
            else:
                if self.condition_settings is None: 
                    self.va_step_size = 0.25 
                elif isinstance(self.condition_settings, float):
                    self.va_step_size = self.condition_settings
                else:
                    raise ValueError("Condition settings must be a list or a float when using valence_arousal as source.")
                # create grid of valence and arousal
                self.valence = np.arange(-1, 1+self.va_step_size, self.va_step_size)
                self.arousal = np.arange(-1, 1+self.va_step_size, self.va_step_size)
        
        elif self.condition_source == "ravdess_expression":
            if self.condition_settings is None: 
                self.condition_settings = list(range(8)) 
            assert isinstance(self.condition_settings, list), "Condition_settings must be a list of integers"

        elif self.condition_source == "iemocap_expression":
            if self.condition_settings is None: 
                self.condition_settings = list(range(4)) 
            
            assert isinstance(self.condition_settings, list), "Condition_settings must be a list of integers"

        else:
            raise ValueError("Condition source must be either original, expression or valence_arousal or original")

    def __len__(self):
        if self.condition_source == "expression":
            return len(self.dataset) * len(self.condition_settings)
        if self.condition_source in ["gt_expression", "gt_expression_intensity", "gt_expression_intensity_identity"]:
            return len(self.dataset) * len(self.condition_settings)
        elif self.condition_source == "valence_arousal":
            return len(self.dataset) * len(self.valence) * len(self.arousal)
        elif self.condition_source == "original":
            return len(self.dataset)
        elif self.condition_source == "ravdess_expression": # ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition 
            # {'angry': 0, 'calm': 1, 'disgust': 2, 'fearful': 3, 'happy': 4, 'neutral': 5, 'sad': 6, 'surprised': 7} 
            return len(self.dataset) * 8
        elif self.condition_source == 'iemocap_expression': # superb/wav2vec2-base-superb-er
            # {0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad'} 
            return len(self.dataset) * 4
        raise NotImplementedError(f"Condition source {self.condition_source} not implemented")

    def __getitem__(self, index):
        if self.condition_source == "expression":
            video_index = index // len(self.condition_settings)
            expression_index = index % len(self.condition_settings)
            sample = self.dataset[video_index]
            sample[self.condition_prefix + "expression"] = torch.nn.functional.one_hot(torch.tensor(expression_index), len(self.condition_settings)).to(torch.float32)
            # hack for when the conditioning comes from a video emotion net during training and hence needs to be inserted for conditioned generation here
            sample["gt_emotion_video_logits"] = {}
            cam = "front" # ugly hack
            sample["gt_emotion_video_logits"][cam] = {}
            sample["gt_emotion_video_logits"][cam] = - (1.-sample[self.condition_prefix + "expression"].clone()) * 99
            sample["condition_name"] = AffectNetExpressions(expression_index).name
        elif self.condition_source in ["gt_expression", "gt_expression_intensity", "gt_expression_intensity_identity"]:
            video_index = index // len(self.condition_settings)
            expression_index = index % len(self.condition_settings)
            sample = self.dataset[video_index]
            # sample[self.condition_prefix + "expression_label"] = torch.nn.functional.one_hot(torch.tensor(expression_index), len(self.condition_settings)).to(torch.float32)
            sample[self.condition_prefix + "expression_label"] = torch.tensor(expression_index)
            if self.condition_source == "gt_expression_intensity":
                # intensity = 3 # highest intensity (1 gets subtracted and then one-hot encoded but this happens in the styling module)
                intensity = 3 if self.conditioned_intensity is None else self.conditioned_intensity
                sample[self.condition_prefix + "expression_intensity"] = torch.nn.functional.one_hot(torch.tensor(intensity), 3).to(torch.float32)
            elif self.condition_source == "gt_expression_intensity_identity":
                # sample[self.condition_prefix + "expression_intensity"] = torch.nn.functional.one_hot(torch.tensor(intensity), 3).to(torch.float32)
                # intensity = 3 # highest intensity (1 gets subtracted and then one-hot encoded but this happens in the styling module)
                intensity = 3 if self.conditioned_intensity is None else self.conditioned_intensity
                sample[self.condition_prefix + "expression_intensity"] = torch.tensor(intensity)
                identity = 0 # just use the style of the first identity
                # n_identities = len(self.dataset.identity_labels)
                # sample[self.condition_prefix + "expression_identity"] = torch.nn.functional.one_hot(torch.tensor(identity), n_identities).to(torch.float32)
                sample[self.condition_prefix + "expression_identity"] = torch.tensor(identity)
            sample["condition_name"] = AffectNetExpressions(expression_index).name
            # hack for when the conditioning comes from a video emotion net during training and hence needs to be inserted for conditioned generation here
            sample["gt_emotion_video_logits"] = {}
            cam = "front" # ugly hack
            sample["gt_emotion_video_logits"][cam] = {}
            sample["gt_emotion_video_logits"] = sample[self.condition_prefix + "expression_label"].clone()
            sample["gt_expression_label"] = sample[self.condition_prefix + "expression_label"].clone()
            if self.condition_source == "gt_expression_intensity":
                sample["condition_name"] += f"_int_{intensity}"
            elif self.condition_source == "gt_expression_intensity_identity":
                sample["condition_name"] += f"_int_{intensity}_id_{identity}"
        elif self.condition_source == "valence_arousal":
            video_index = index // (len(self.valence) * len(self.arousal))
            va_index = index % (len(self.valence) * len(self.arousal))
            valence_index = va_index // len(self.arousal)
            arousal_index = va_index % len(self.arousal)
            # sample = self.dataset._getitem(video_index)
            sample = self.dataset[video_index]
            sample[self.condition_prefix + "valence"] = torch.tensor(self.valence[valence_index], dtype=torch.float32)
            sample[self.condition_prefix + "arousal"] = torch.tensor(self.arousal[arousal_index], dtype=torch.float32)
            sample["condition_name"] = f"valence_{self.valence[valence_index]:0.2f}_arousal_{self.arousal[arousal_index]:0.2f}"
            
        elif self.condition_source == "original":
            video_index = index
            sample = self.dataset[video_index]
            return sample
        elif self.condition_source == "ravdess_expression":
            exp_dict = {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful', 4: 'happy', 5:'neutral', 6:'sad', 7: 'surprised'} 
            video_index = index // len(exp_dict)
            expression_index = index % len(exp_dict)
            sample = self.dataset[video_index]
            sample[self.condition_prefix + "expression"] = torch.nn.functional.one_hot(torch.tensor(expression_index), len(self.exp_dict)).to(torch.float32)
            sample["condition_name"] = exp_dict[expression_index] 
            
        elif self.condition_source == "iemocap_expression":
            exp_dict = {0: 'neu', 1: 'hap', 2: 'ang', 'sad': 3} 
            video_index = index // len(exp_dict)
            expression_index = index % len(exp_dict)
            sample = self.dataset[video_index]
            sample[self.condition_prefix + "expression"] = torch.nn.functional.one_hot(torch.tensor(expression_index), len(self.exp_dict)).to(torch.float32)
            sample["condition_name"] = exp_dict[expression_index] 
        else:
            raise NotImplementedError(f"Condition source '{self.condition_source}' not implemented")
        
        try:
            T =  sample["raw_audio"].size(0)
        except KeyError:
            try:
                T =  sample["video"].size(0)
            except KeyError:
                T =  sample["gt_vertices"].size(0)
        if self.expand_temporal: 
            if self.condition_source in ["expression", "iemocap_expression", "ravdess_expression"]:
                if self.condition_prefix + "expression" in sample:
                    sample[self.condition_prefix +  "expression"] = sample[self.condition_prefix + "expression"][None, ...].repeat(T, 1)
            elif self.condition_source == "valence_arousal":
                if self.condition_prefix +  "valence" in sample:
                    sample[self.condition_prefix + "valence"] = sample[self.condition_prefix + "valence"][None, ...].repeat(T, 1)
                if self.condition_prefix +   "arousal" in sample:
                    sample[self.condition_prefix + "arousal"] = sample[self.condition_prefix + "arousal"][None, ...].repeat(T, 1)
            # TODO: expression intensity 
            elif self.condition_source == "gt_expression":
                if self.condition_prefix + "expression_label" in sample:
                    sample[self.condition_prefix + "expression_label"] = sample[self.condition_prefix + "expression_label"][None, ...].repeat(T)
            elif self.condition_source == "gt_expression_intensity":
                if self.condition_prefix + "expression_label" in sample:
                    sample[self.condition_prefix + "expression_label"] = sample[self.condition_prefix + "expression_label"][None, ...].repeat(T)
                if self.condition_prefix + "expression_intensity" in sample:
                    sample[self.condition_prefix + "expression_intensity"] = sample[self.condition_prefix + "expression_intensity"][None, ...].repeat(T)
            elif self.condition_source == "gt_expression_intensity_identity":
                if self.condition_prefix + "expression_label" in sample:
                    sample[self.condition_prefix + "expression_label"] = sample[self.condition_prefix + "expression_label"][None, ...].repeat(T)
                if self.condition_prefix + "expression_intensity" in sample:
                    sample[self.condition_prefix + "expression_intensity"] = sample[self.condition_prefix + "expression_intensity"][None, ...].repeat(T)
                if self.condition_prefix + "expression_identity" in sample:
                    sample[self.condition_prefix + "expression_identity"] = sample[self.condition_prefix + "expression_identity"][None, ...].repeat(T)
            else:
                raise NotImplementedError(f"Condition source '{self.condition_source}' not implemented")
            sample["condition_name"] =  [sample["condition_name"] ] * T
        # add video name to sample
        # sample["video_name"] = str(self.dataset.video_list[video_index])
        return sample