import numpy as np
import torch
from .AffectNetDataModule import AffectNetExpressions
from .VideoDatasetBase import VideoDatasetBase


class ConditionedVideoTestDatasetWrapper(torch.utils.data.Dataset): 

    def __init__(self,
                 dataset : VideoDatasetBase,
                 condition_source, 
                 condition_settings
                 ):    
        self.dataset = dataset
        self.condition_source = condition_source or "original"
        self.condition_settings = condition_settings or None
        if self.condition_source == "original":
            self.condition_settings = None
        elif self.condition_source == "basic_expression":
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
        
            assert isinstance(self.condition_settings, list), "Condition_settings must be a list of integers"
        
        else:
            raise ValueError("Condition source must be either original, basic_expression or valence_arousal or original")

    def __len__(self):
        if self.condition_source == "basic_expression":
            return len(self.dataset) * len(self.condition_settings)
        elif self.condition_source == "valence_arousal":
            return len(self.dataset) * len(self.valence) * len(self.arousal)
        elif self.condition_source == "original":
            return len(self.dataset)
        raise NotImplementedError(f"Condition source {self.condition_source} not implemented")

    def __getitem__(self, index):
        if self.condition_source == "basic_expression":
            video_index = index // 7
            expression_index = index % 7
            sample = self.dataset[video_index]
            sample["basic_expression"] = torch.tensor(expression_index)
            sample["condition_name"] = AffectNetExpressions(expression_index).name
            return sample
        elif self.condition_source == "valence_arousal":
            video_index = index // (len(self.valence) * len(self.arousal))
            va_index = index % (len(self.valence) * len(self.arousal))
            valence_index = va_index // len(self.arousal)
            arousal_index = va_index % len(self.arousal)
            # sample = self.dataset._getitem(video_index)
            sample = self.dataset[video_index]
            sample["valence"] = torch.tensor(self.valence[valence_index])
            sample["arousal"] = torch.tensor(self.arousal[arousal_index])
            sample["condition_name"] = f"valence_{self.valence[valence_index]:0.2f}_arousal_{self.arousal[arousal_index]:0.2f}"
            return sample 
        elif self.condition_source == "original":
            video_index = index
            # sample = self.dataset._getitem(video_index)
            sample = self.dataset[video_index]
        else:
            raise NotImplementedError(f"Condition source {self.condition_source} not implemented")
        
        # add video name to sample
        # sample["video_name"] = str(self.dataset.video_list[video_index])
        return sample