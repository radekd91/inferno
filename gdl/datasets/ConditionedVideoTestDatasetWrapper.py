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
            sample["condition_name"] = AffectNetExpressions(expression_index).name
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
            raise NotImplementedError(f"Condition source {self.condition_source} not implemented")
        
        T =  sample["video"].size(0)
        if self.expand_temporal: 
            if self.condition_prefix + "expression" in sample:
                sample[self.condition_prefix +  "expression"] = sample[self.condition_prefix + "expression"][None, ...].repeat(T, 1)
            if self.condition_prefix +  "valence" in sample:
                sample[self.condition_prefix + "valence"] = sample[self.condition_prefix + "valence"][None, ...].repeat(T, 1)
            if self.condition_prefix +   "arousal" in sample:
                sample[self.condition_prefix + "arousal"] = sample[self.condition_prefix + "arousal"][None, ...].repeat(T, 1)
            sample["condition_name"] =  [sample["condition_name"] ] * T
        # add video name to sample
        # sample["video_name"] = str(self.dataset.video_list[video_index])
        return sample