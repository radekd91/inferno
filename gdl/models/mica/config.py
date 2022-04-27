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
cfg.model.topology_path = os.path.join(cfg.mica_dir, '/scratch/wzielonka/FLAME/2020', 'head_template.obj')
cfg.model.flame_model_path = os.path.join(cfg.mica_dir, '/scratch/wzielonka/FLAME/2020', 'generic_model.pkl')
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.mica_dir, '/scratch/wzielonka/FLAME/2020', 'landmark_embedding.npy')
cfg.model.face_mask_path = os.path.join(cfg.mica_dir, '/scratch/wzielonka/FLAME/masks', 'uv_face_mask.png')
cfg.model.face_eye_mask_path = os.path.join(cfg.mica_dir, '/scratch/wzielonka/FLAME/masks', 'uv_face_eye_mask.png')
cfg.model.mean_tex_path = os.path.join(cfg.mica_dir, '/scratch/wzielonka/FLAME/2020', 'mean_texture.jpg')
cfg.model.tex_path = os.path.join(cfg.mica_dir, '/scratch/wzielonka/FLAME/2020', 'FLAME_albedo_from_BFM.npz')
cfg.model.tex_type = 'BFM'
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
    return cfg.clone()
