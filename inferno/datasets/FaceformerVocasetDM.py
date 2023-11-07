# THIS CODE IS TAKEN OVER FROM THE FACEFORMER REPO AND ADAPTED TO OUR NEEDS


from distutils.log import debug
from re import template
import pytorch_lightning as pl
from collections import defaultdict
import os, sys
from tqdm import tqdm
import pickle
import librosa
import numpy as np
from transformers import Wav2Vec2Processor
import torch


class FaceformerVocasetDM(pl.LightningDataModule): 

    def __init__(self, 
            root_dir,
            template_file,
            train_subjects,
            val_subjects,
            test_subjects,
            batch_size_train=1,
            batch_size_val=1,
            batch_size_test=1,
            preprocessor_identifier="facebook/wav2vec2-base-960h",
            sequence_length_train=None,
            sequence_length_val=None,
            sequence_length_test=None,
            debug_mode=False,
            num_workers=1):
        super().__init__()
        self.root_dir = root_dir
        self.train_subjects = train_subjects
        self.val_subjects = val_subjects
        self.test_subjects = test_subjects
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.sequence_length_train = sequence_length_train
        self.sequence_length_val = sequence_length_val
        self.sequence_length_test = sequence_length_test
        self.num_workers = num_workers

        self.preprocessor_identifier = preprocessor_identifier
        self.template_file = template_file
        self.wav_path = "wav"
        self.vertices_path = "vertices_npy"
        self.expression_path = "exp_npy"
        self.jaw_path = "jaw_npy"
        self.dataset_name = "vocaset"
        self.use_flame = True
        self.debug_mode = debug_mode

    def prepare_data(self):
        pass 

    def setup(self, stage=None):
        print("Loading data...")
        data = defaultdict(dict)
        self.train_data = []
        self.valid_data = []
        self.test_data = []

        audio_path = os.path.join(self.root_dir, self.wav_path)
        vertices_path = os.path.join(self.root_dir, self.vertices_path)
        exp_path = os.path.join(self.root_dir, self.expression_path)
        jaw_path = os.path.join(self.root_dir, self.jaw_path)
        processor = Wav2Vec2Processor.from_pretrained(self.preprocessor_identifier)

        template_file = os.path.join(self.root_dir, '..', self.template_file)
        with open(template_file, 'rb') as fin:
            templates = pickle.load(fin,encoding='latin1')
        
        for r, ds, fs in os.walk(audio_path):
            idx = 0
            for f in tqdm(fs):
                if f.endswith("wav"):
                    wav_path = os.path.join(r,f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                    key = f.replace("wav", "npy")
                    data[key]["audio"] = input_values
                    subject_id = "_".join(key.split("_")[:-1])
                    temp = templates[subject_id]
                    data[key]["name"] = f
                    data[key]["template"] = temp.reshape((-1)) 
                    vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                    exp_file = os.path.join(exp_path,f.replace("wav", "npy"))
                    jaw_file = os.path.join(jaw_path,f.replace("wav", "npy"))
                    if not os.path.exists(vertice_path):
                        del data[key]
                    else:
                        if self.dataset_name == "vocaset":
                            data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                            if self.use_flame: 
                                try:
                                    exp = np.load(exp_file,allow_pickle=True).reshape(data[key]["vertice"].shape[0]*2, -1)[::2,:]
                                except ValueError: 
                                    exp = np.load(exp_file,allow_pickle=True).reshape(data[key]["vertice"].shape[0]*2-1, -1)[::2,:]
                                assert exp.shape[0] == data[key]["vertice"].shape[0]
                                data[key]["exp"] = exp
                                # data[key]["exp"] = np.load(exp_file,allow_pickle=True)[::2,:].astype(np.float32).view(data[key]["vertice"].shape[0],-1)
                                data[key]["jaw"] = np.load(jaw_file,allow_pickle=True)[::2,:].astype(np.float32)
                                # # data[key]["exp"] = np.zeros((data[key]["vertice"].shape[0], 100))
                                # data[key]["exp"] = np.random.randn(data[key]["vertice"].shape[0], 100).astype(np.float32)
                                # data[key]["jaw"] = np.random.rand(data[key]["vertice"].shape[0], 3).astype(np.float32)
                        elif self.dataset_name == "BIWI":
                            raise NotImplementedError("BIWI dataset is not supported")
                            data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)
                        else: 
                            raise NotImplementedError("Dataset not implemented")
                        idx += 1
                if self.debug_mode and idx == 10: 
                # if self.debug_mode and idx == 50: 
                # if self.debug_mode and idx == 100: 
                    break


        subjects_dict = {}
        # subjects_dict["train"] = [i for i in self.train_subjects.split(" ")]
        # subjects_dict["val"] = [i for i in self.val_subjects.split(" ")]
        # subjects_dict["test"] = [i for i in self.test_subjects.split(" ")]
        subjects_dict["train"] = self.train_subjects
        subjects_dict["val"] = self.val_subjects
        subjects_dict["test"] = self.test_subjects

        splits = {
            'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
            'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}
        }
    
        for k, v in data.items():
            subject_id = "_".join(k.split("_")[:-1])
            sentence_id = int(k.split(".")[0][-2:])
            if subject_id in subjects_dict["train"] and sentence_id in splits[self.dataset_name]['train']:
                self.train_data.append(v)
            if subject_id in subjects_dict["val"] and sentence_id in splits[self.dataset_name]['val']:
                self.valid_data.append(v)
            if subject_id in subjects_dict["test"] and sentence_id in splits[self.dataset_name]['test']:
                self.test_data.append(v)

        
        self.training_set = VocaSet( self.train_data, subjects_dict, "train", sequence_length = self.sequence_length_train)
        self.validation_set = VocaSet( self.valid_data, subjects_dict, "val", sequence_length = self.sequence_length_val)
        self.test_set_train = VocaSet( self.train_data, subjects_dict, "train", sequence_length = self.sequence_length_test, sequence_sampling="start", limit_size=10)
        self.test_set_val = VocaSet( self.valid_data, subjects_dict, "val", sequence_length = self.sequence_length_test, sequence_sampling="start", limit_size=10)
        # self.test_set_test = VocaSet( self.test_data, subjects_dict, "test", sequence_length = self.sequence_length_test, sequence_sampling="start")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.training_set, batch_size=self.batch_size_train, 
            shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.validation_set, batch_size=self.batch_size_val, 
            shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # return torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.batch_size_test, 
        #     shuffle=False, num_workers=self.num_workers)
        return [torch.utils.data.DataLoader(dataset=self.test_set_train, batch_size=self.batch_size_test, 
                shuffle=False, num_workers=self.num_workers),
            torch.utils.data.DataLoader(dataset=self.test_set_val, batch_size=self.batch_size_test, 
                shuffle=False, num_workers=self.num_workers), 
            ]
    
    def get_num_training_identities(self):
        return len(self.train_subjects)


class VocaSet(torch.utils.data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train", sequence_length=None, sequence_sampling="random", limit_size=None):
        self.data = data
        self.len = min(limit_size, len(self.data)) if limit_size is not None else len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.sequence_length = sequence_length or "all"
        self.sequence_sampling = sequence_sampling

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        # audio = self.data[index]["audio"]
        audio = self.data[index]["audio"]

        sample_rate = 16000
        sample_rate_ratio = sample_rate // 25

        gt_vertices = self.data[index]["vertice"]

        T = gt_vertices.shape[0]
        audio_cut_off_index = T * sample_rate_ratio
        if audio.shape[0] < audio_cut_off_index:
            audio = np.concatenate([audio, np.zeros(audio_cut_off_index - audio.shape[0])])
        audio = audio[:audio_cut_off_index]
        audio = audio.reshape(T, sample_rate_ratio)

        template = self.data[index]["template"]
        if 'exp' in self.data[index].keys():
            exp = self.data[index]["exp"] 
        else:
            exp = np.array([])

        if 'jaw' in self.data[index].keys():
            jaw = self.data[index]["jaw"]
        else:
            jaw = np.array([])
            
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        
        sample = {} 
        # sample["raw_audio"] = torch.FloatTensor(audio)
        sample["processed_audio"] = torch.FloatTensor(audio) # "processed_audio" because it has been run through the "processor" of w2v
        sample["gt_vertices"] = torch.FloatTensor(gt_vertices)
        sample["template"] = torch.FloatTensor(template)
        sample["gt_exp"] = torch.FloatTensor(exp)
        sample["gt_jaw"] = torch.FloatTensor(jaw)
        sample["one_hot"] = torch.FloatTensor(one_hot)
        sample["filename"] = file_name
        sample["frame_indices"] = torch.arange(gt_vertices.shape[0])
        sample["samplerate"] = sample_rate
        
        if self.sequence_length != "all":
            # start at a random_index
            if self.sequence_sampling == "random":
                random_index = np.random.randint(0, gt_vertices.shape[0] - self.sequence_length)
                # sample["processed_audio"] = sample["processed_audio"][random_index:random_index+self.sequence_length]
                # sample["raw_audio"] = sample["raw_audio"][(random_index * sample_rate_ratio) :((random_index+self.sequence_length)*sample_rate_ratio)]
                # sample["raw_audio"] = sample["raw_audio"][ sample_rate_ratio :(random_index+self.sequence_length)]
                sample["processed_audio"] = sample["processed_audio"][ sample_rate_ratio :(random_index+self.sequence_length)]
                sample["gt_vertices"] = sample["gt_vertices"][random_index:random_index+self.sequence_length]
                sample["gt_exp"] = sample["gt_exp"][random_index:random_index+self.sequence_length]
                sample["gt_jaw"] = sample["gt_jaw"][random_index:random_index+self.sequence_length]
                sample["frame_indices"] = sample["frame_indices"][random_index:random_index+self.sequence_length]
            elif self.sequence_sampling == "start": 
                # sample["raw_audio"] = sample["raw_audio"][:self.sequence_length* sample_rate_ratio]
                # sample["raw_audio"] = sample["raw_audio"][:self.sequence_length]
                sample["processed_audio"] = sample["processed_audio"][:self.sequence_length]
                sample["gt_vertices"] = sample["gt_vertices"][:self.sequence_length]
                sample["gt_exp"] = sample["gt_exp"][:self.sequence_length]
                sample["gt_jaw"] = sample["gt_jaw"][:self.sequence_length]
                sample["frame_indices"] = sample["frame_indices"][:self.sequence_length]
            else: 
                raise ValueError("sequence_sampling must be either 'random' or 'start'")

        return sample

    def __len__(self):
        return self.len
