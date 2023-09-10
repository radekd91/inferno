"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
from gdl.utils.condor import execute_on_cluster
from pathlib import Path
import gdl_apps.FaceReconstruction.training.train_face_reconstruction as script

import datetime
from omegaconf import DictConfig, OmegaConf, open_dict
import time as t
import copy
import sys
import timeit


submit_ = False
# submit_ = True



def benchmark_data_loading():
    from hydra.core.global_hydra import GlobalHydra

    coarse_conf = "emica_pretrain_stage" 
    # coarse_conf = "emica_deca_stage"
    # coarse_conf = "emica_emoca_stage"


    finetune_modes = [
        [ 
            [
            ]
        ],
    ]
    
    batch_sizes = [8]
    ring_size = 8
    new_finetune_modes = []

    for mode in finetune_modes: 
        for batch_size in batch_sizes:
            # num_workers = int(batch_size * 1)
            # num_workers = 8
            num_workers = 15
            # num_workers = 0
            mode = copy.deepcopy(mode)
            mode[0] += [ 
                f'learning.batching.batch_size_train={batch_size}',
                f'learning.batching.batch_size_val={batch_size}',
                f'learning.batching.batch_size_test={batch_size}',
                f'learning.batching.ring_size_train={ring_size}',
                f'learning.batching.ring_size_val={ring_size}',
                f'learning.batching.ring_size_test={ring_size}',
                f'data.num_workers={num_workers}'
            ]
            new_finetune_modes += [mode]
    finetune_modes = new_finetune_modes

    # # 2. What occlusions probability is optimal? 
    # mouth_occlusions_probabilities = [0.2, 0.4, 0.6, 0.8, 1.0]
    # new_finetune_modes = []
    # for mode in finetune_modes:
    #     for mouth_occlusions_probability in mouth_occlusions_probabilities:
    #         mode = copy.deepcopy(mode)
    #         mode[0] += [ 
    #             F'data.occlusion_settings_train.occlusion_probability_mouth={mouth_occlusions_probability}'
    #         ]
    #         new_finetune_modes += [mode]
    # finetune_modes = new_finetune_modes

    # 3.
   

    fixed_overrides_coarse = [
        ## LRS3
        # # 'data.split=random_by_identity_pretrain_80_20',
        # 'data.split=specific_identity_80_20_pretrain/0af00UcTOSc', # training on a single identity 
        
        # # ## MEAD 
        # 'data/datasets=mead', 
        # # 'data.split=specific_identity_sorted_80_20_M003',
        # # 'data.split=random_by_sequence_sorted_70_15_15',
        # 'data.split=random_by_identityV2_sorted_70_15_15',
        
        # ## CelebV-Text
        'data/datasets=celebvtext', 
        'data.split=random_70_15_15',
        # 'data/augmentations=default',

        # ## CelebV-Text
        # 'data/datasets=', 
        # 'data.split=specific_identity_sorted_80_20_M003',

        ## CelebV-HQ 
        # 'data/datasets=celebvhq_no_occlusion', # training on a single video (and therefore identity)
        # # 'data.split=specific_video_temporal_eknCAJ0ik8c_0_0_80_10_10',
        # 'data.split=specific_video_temporal_6jRVZQMKlxw_1_0_80_10_10', 
        # 'data.preload_videos=true',
        # 'data.inflate_by_video_size=true',
    ]

    # config_pairs = []
    for fmode in finetune_modes:
        coarse_overrides = fixed_overrides_coarse.copy()
        coarse_overrides += fmode[0]

        # coarse_overrides += [emonet_weight_override]
        # detail_overrides += [emonet_weight_override]

        conf = script.configure(
            coarse_conf, coarse_overrides,
            # detail_conf, detail_overrides
        )
        # cfgs = list(cfgs)

        GlobalHydra.instance().clear()
        # config_pairs += [cfgs]


        OmegaConf.set_struct(conf, False)
        with open_dict(conf) as d:
            tags = ["INITIAL_SMALL_TESTS"]
            if not submit_:
                tags += ["DEBUG_FROM_WORKSTATION"]
            if d.learning.tags is None:
                d.learning.tags = tags
        cfg = OmegaConf.to_container(conf)


        conf = OmegaConf.create(cfg)

        dm, name = script.prepare_data(conf)
        dm.prepare_data()
        dm.setup()

        dl = dm.train_dataloader() 

        # Create an iterator from the DataLoader
        dataiter = iter(dl)

        # Measure time for fetching each batch
        while True:
            try:
                start_time = timeit.default_timer()
                batch = next(dataiter)
                end_time = timeit.default_timer()
                
                print(f"---------- Time taken to load a batch: {end_time - start_time:0.04f} -------------")
                
            except StopIteration:
                print("End of dataset")
                break





if __name__ == "__main__":
    benchmark_data_loading()

