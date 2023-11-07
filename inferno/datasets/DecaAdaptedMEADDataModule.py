from .MEADDataModule import * 
from inferno.models.mica.MicaInputProcessing import MicaDatasetWrapper
from torch.utils.data import Dataset

class DecaAdaptedMeadDataModule(MEADDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        super().setup(stage)
        self.training_set = MicaDatasetWrapper(DecaAdaptedMEADDataset(self.training_set))
        self.validation_set = MicaDatasetWrapper(DecaAdaptedMEADDataset(self.validation_set))
        self.test_set = MicaDatasetWrapper(DecaAdaptedMEADDataset(self.test_set))



class DecaAdaptedMEADDataset(Dataset):

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # MEAD gives these keys: 
        # ['video', 'frame_indices', 'landmarks', 'landmarks_validity', 'segmentation', 'video_masked', 'segmentation_masked', 'masked_frames']

        # DECA sample is supposed to have the following keys:
        # ['image', 'landmark', 'mask', 'path',] 
        # ['image', 'path', 'mica_images', 'affectnetexp', 'va', 'label', 'expression_weight', 'expression_sample_weight', 'valence_sample_weight', 'arousal_sample_weight', 'va_sample_weight', 'landmark', 'landmark_mediapipe', 'mask']
        # ['image', 'path', 'mica_images', 'landmark', 'landmark_mediapipe', 'mask']
        deca_sample = {}
        deca_sample['image'] = sample['video']
        deca_sample['landmark'] = sample['landmarks']['fan']
        deca_sample['landmark_mediapipe'] = sample['landmarks']['mediapipe']
        deca_sample['mask'] = sample['segmentation']
        # deca_sample['path'] = sample['path']
        return deca_sample