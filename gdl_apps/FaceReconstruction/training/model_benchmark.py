import os, sys 
from pathlib import Path
from gdl.models.FaceReconstruction.FaceRecBase import FaceReconstructionBase
from gdl.utils.other import class_from_str
from omegaconf import OmegaConf, DictConfig
import gdl_apps.FaceReconstruction.training.train_face_reconstruction as script
from gdl.utils.batch import dict_to_device
import timeit
import torch
import copy
from tqdm import auto

def main():
    cfg_path = "/is/cluster/work/rdanecek/face_reconstruction/trainings/2023_09_07_18-16-11_3411518403464256796_FaceReconstructionBase_MEADDEmicaEncoder_Aug/cfg.yaml"
    cfg = OmegaConf.load(cfg_path)
    cfg = cfg.coarse
    
    batch_size = 1
    ring_size = 8

    cfg.learning.batching.batch_size_train = batch_size
    cfg.learning.batching.batch_size_val = batch_size
    cfg.learning.batching.batch_size_test = batch_size
    cfg.learning.batching.ring_size_train = ring_size
    cfg.learning.batching.ring_size_val = ring_size
    cfg.learning.batching.ring_size_test = ring_size

    model_class = class_from_str(cfg.model.pl_module_class, sys.modules[__name__])

    model = model_class.instantiate(cfg)
    opt = model.configure_optimizers()
    optimizer = opt['optimizer']

    dm, name = script.prepare_data(cfg)
    dm.prepare_data()
    dm.setup()

    dl = dm.train_dataloader() 

    # Create an iterator from the DataLoader
    dataiter = iter(dl)

    batch = next(dataiter)

    device = torch.device( "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    model = model.to(device)
    batch = dict_to_device(batch, device)

    times_forward = []
    times_backward = []
    for i in auto.tqdm(range(100)):
        batch_ = copy.deepcopy(batch)
        time = timeit.default_timer()
        total_loss = model.training_step(batch_, 0)
        time_taken = timeit.default_timer() - time
        # print("Time taken forward pass:", time_taken)
        times_forward += [time_taken]

        # backward pass
        time = timeit.default_timer()
        model.backward(total_loss, optimizer, 0, None)
        time_taken = timeit.default_timer() - time
        # print("Time taken backward pass:", time_taken)
        times_backward += [time_taken]
        # zero the parameter gradients
        optimizer.zero_grad()


    print("Average time taken forward pass:", sum(times_forward) / len(times_forward))
    print("Average time taken backward pass:", sum(times_backward) / len(times_backward))





if __name__ == '__main__':
    main()

    
