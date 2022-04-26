'''
Default config for MICA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_mica_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
cfg.mica_dir = abs_mica_dir
cfg.device = 'cuda'
cfg.device_id = '0'
cfg.output_dir = ''


# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join(cfg.mica_dir, 'data', 'texture_data_256.npy')
cfg.model.fixed_displacement_path = os.path.join(cfg.mica_dir, 'data', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'generic_model.pkl')
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'landmark_embedding.npy')
cfg.model.face_mask_path = os.path.join(cfg.mica_dir, 'data/masks', 'uv_face_mask.png')
cfg.model.face_eye_mask_path = os.path.join(cfg.mica_dir, 'data/masks', 'uv_face_eye_mask.png')
cfg.model.mean_tex_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'mean_texture.jpg')
cfg.model.tex_path = os.path.join(cfg.mica_dir, 'data/FLAME2020', 'FLAME_albedo_from_BFM.npz')
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uv_size = 256
cfg.model.n_shape = 300
cfg.model.n_tex = 100
cfg.model.n_exp = 100
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.mapping_layers = 3

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.max_epochs = 500
cfg.train.max_steps = 100000
cfg.train.lr = 1e-4

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--type', type=str, help='test type')
    parser.add_argument('--checkpoint', type=str, help='checkpoint')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg, args
