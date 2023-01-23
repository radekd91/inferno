import os, sys 
from pathlib import Path
from gdl.models.video_emorec.VideoEmotionClassifier import VideoEmotionClassifier
from gdl.layers.losses.VideoEmotionLoss import VideoEmotionRecognitionLoss, create_video_emotion_loss
# from gdl.layers.losses.EmoNetLoss import 
from gdl.models.temporal.external.LipReadingLoss import LipReadingLoss
from gdl.layers.losses.EmoNetLoss import create_emo_loss
from gdl.datasets.MEADPseudo3DDM import MEADPseudo3DDM
from gdl.models.DecaFLAME import FLAME, FLAMETex
from omegaconf import OmegaConf
from gdl_apps.TalkingHead.training.train_talking_head import get_condition_string_from_config
from gdl.models.temporal.BlockFactory import renderer_from_cfg
from gdl.models.temporal.TemporalFLAME import FlameShapeModel
from gdl.models.rotation_loss import compute_rotation_loss
from skvideo.io import vwrite, vread
import torch
from munch import Munch, munchify
import datetime
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import copy


def rename_inconsistencies(sample):
    sample['vertices'] = sample['verts'] 
    sample['jaw'] = sample['jawpose']
    sample['exp'] = sample['expcode']
    sample['shape'] = sample['shapecode']

    # del sample['verts']
    return sample


class ExpressionRegLoss(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def prepare_target(self, target_sample):
        pass 
        
    def forward(self, sample):
        return (sample['exp'] ** 2).mean()


class TargetJawRegLoss(torch.nn.Module):

    def __init__(self, input_space, output_space) -> None:
        super().__init__()
        self.jaw_pose = None
        self.input_space = input_space 
        self.output_space = output_space

    def prepare_target(self, target_sample):
        self.jaw_pose = target_sample['jawpose']
        
    def forward(self, sample):
        return compute_rotation_loss(sample['jawpose'], self.jaw_pose, mask=None, 
        r1_input_rep=self.input_space, r2_input_rep=self.output_space)


class LipReadingTargetLoss(torch.nn.Module):

    def __init__(self, lip_reading_loss) -> None:
        super().__init__()
        self.lip_reading_loss = lip_reading_loss
        self.target_features = None
        self.camera = "front"

        for param in self.lip_reading_loss.parameters():
            param.requires_grad = False
        

    def prepare_target(self, target_sample):
        # target = rename_inconsistencies(target)
        self.target_features = self.lip_reading_loss._forward_input(target_sample["predicted_mouth_video"][self.camera])
        

    def forward(self, sample):
        assert self.target_features is not None, "You need to call prepare_target first"
        features = self.lip_reading_loss._forward_output(sample["predicted_mouth_video"][self.camera])
        return self.lip_reading_loss._compute_feature_loss(self.target_features, features)


class VideoEmotionTargetLoss(torch.nn.Module):

    def __init__(self, video_emotion_loss) -> None:
        super().__init__()
        self.video_emotion_loss = video_emotion_loss
        # freeze the video emotion loss
        for param in self.video_emotion_loss.parameters():
            param.requires_grad = False

        self.target_features = None
        self.camera = "front"

    def prepare_target(self, target_per_frame_emotion_feature):
        self.target_features = self.video_emotion_loss._forward_output(output_emotion_features=target_per_frame_emotion_feature)
        

    def forward(self, sample):
        assert self.target_features is not None, "You need to call prepare_target first"
        features = self.video_emotion_loss._forward_output(sample["predicted_video"][self.camera])
        return self.video_emotion_loss._compute_feature_loss(self.target_features, features)


class EmotionTargetLoss(torch.nn.Module):

    def __init__(self, emotion_loss) -> None:
        super().__init__()
        self.emotion_loss = emotion_loss
        for param in self.emotion_loss.parameters():
            param.requires_grad = False
        
        self.target_features = None
        self.camera = "front"

    def prepare_target(self, target_sample):
        target_emotion_images = target_sample["predicted_video"][self.camera]
        B, T = target_emotion_images.shape[:2]
        target_emotion_images = target_emotion_images.view(B*T, *target_emotion_images.shape[2:])    
        self.target_features  = self.emotion_loss._forward_input(target_emotion_images)['emo_feat_2'].view(B, T, -1)
        
    def forward(self, sample):
        assert self.target_features is not None, "You need to call prepare_target first"
        features = self.emotion_loss._forward_output(sample["predicted_mouth_video"][self.camera])
        return self.emotion_loss._compute_feature_loss(self.target_features, features)


def detach_dict(sample):
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if value.requires_grad:
                sample[key] = value
            else:
                sample[key] = value.detach()
        elif isinstance(value, dict):
            sample[key] = detach_dict(value)
        elif isinstance(value, list):
            sample[key] = [detach_dict(v) for v in value]
        else:
            raise NotImplementedError(f"Cannot detach {type(value)}")
    return sample


def copy_dict(sample):
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value.clone()
        elif isinstance(value, dict):
            sample[key] = copy_dict(value)
        elif isinstance(value, list):
            sample[key] = [copy_dict(v) for v in value]
        else:
            raise NotImplementedError(f"Cannot detach {type(value)}")
    return sample

class OptimizationProblem(object): 

    def __init__(self, renderer, flame, losses, output_folder) -> None:
        self.renderer = renderer
        self.flame = flame
        self.losses = losses
        self.optimizer = None
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=False)

    def forward(self, sample,):
        """
        The sample is a dictionary with the following keys
        - 'shapecode': the shape code of the flame model
        - 'expression_code': the squence of expression codes of the flame model
        - 'jaw_pose': the squence of jaw poses of the flame model
        - 'tex_code': the texture codes of the flame model
        """
        # 1. Render the current values
        sample = self.flame(sample)
        # sample['vertices'] = sample['verts'] 
        # del sample['verts']
        sample = rename_inconsistencies(sample)
        sample = self.renderer(sample, input_key_prefix="", output_prefix="predicted_")
        # check_nan(sample)

        # # render the sequence
        # sample = self.render_sequence(sample)
        return sample

    def configure_optimizers(self, variables, optim, lr=1e-3):
        assert variables is not None, "You need to specify the variables to optimize"
        self.variables = variables
        vars_ = list(self.variables.values())
        if optim == "adam":
            self.optimizer = torch.optim.Adam(vars_, lr=lr)
        elif optim == "sgd":
            self.optimizer = torch.optim.SGD(vars_, lr=lr)
        elif optim == "lbfgs":
            self.optimizer = torch.optim.LBFGS(vars_, lr=lr)
        else: 
            raise ValueError("Unknown optimizer {}".format(optim))

    def optimize(self, init_sample, n_iter=None, patience=None): 
        sample = init_sample.copy()
        n_iter = n_iter or 100

        visualization_frequency = 10
        
        best_loss = 1e10
        best_iter = 0
        best_sample = None
        iter_since_best = 0
        for i in range(n_iter):
            
            # delete keys from old forward pass that correspond to the vertices and images 
            del sample['verts']
            del sample['predicted_video']
            del sample['predicted_mouth_video'] 
            del sample['predicted_landmarks2d_flame_space']
            del sample['predicted_landmarks3d_flame_space']
            del sample['predicted_landmarks_2d']

            sample, total_loss, losses, weighted_losses = self.optimization_step(sample)
            print(f"Iteration {i} - Total loss: {total_loss.item():.4f}")

            if total_loss < best_loss:
                best_loss = total_loss
                best_iter = i
                best_sample = copy_dict( detach_dict( sample.copy()))
                iter_since_best = 0
            else:
                iter_since_best += 1

            dict_to_log = {}
            if i % visualization_frequency == 0:
                
                for cam in sample['predicted_video'].keys():
                    predicted_video = (sample['predicted_video'][cam].detach().cpu().numpy()[0] * 255).astype(np.uint8).transpose(0, 2, 3, 1)
                    # predicted_mouth_video = sample['predicted_mouth_video'].detach().cpu().numpy()
                    # write the video at 25 fps
                    video_fname = self.output_folder / f"{cam}_iter_{i:05d}.mp4"
                    vwrite(video_fname, predicted_video)

                    # add the video to wandb
                    video_wandb = wandb.Video(str(video_fname), fps=25, format="mp4")
                    dict_to_log[f"video/{cam}"] = video_wandb

            dict_to_log.update({f"loss/{k}": v.item() for k, v in losses.items()})
            dict_to_log.update({f"weighted_loss/{k}": v.item() for k, v in weighted_losses.items()})
            dict_to_log.update({"total_loss": total_loss.item()})
            wandb.log(dict_to_log, step=i)

            if patience is not None and iter_since_best > 100:
                print(f"Stopping optimization after {i} iterations because the loss did not improve for {patience} iterations.")
                break

        print(f"Best loss: {best_loss.item():.4f} at iteration {best_iter}")


        dict_to_log = {}
        for cam in sample['predicted_video'].keys():
            predicted_video = (best_sample['predicted_video'][cam].detach().cpu().numpy()[0] * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            video_fname = self.output_folder / f"best.mp4"
            vwrite(video_fname, predicted_video)
            video_wandb = wandb.Video(str(video_fname), fps=25, format="mp4")
            dict_to_log[f"best_video/{cam}"] = video_wandb
        wandb.log(dict_to_log)
        return sample

    def optimization_step(self, sample):
        # 0. Compute the forward pass
        sample = self.forward(sample)
        # 1. Compute the losses
        total_loss, losses, weighted_losses = self.compute_losses(sample)
        # 2. Compute the gradients
        total_loss.backward()
        # 3. Update the parameters
        
        # exp1 = sample['exp'].detach().clone()
        self.optimizer.step()
        # exp2 = sample['exp'].detach().clone()
        # exp3 = self.variables['expcode'].detach().clone()
        # print((exp1-exp2).sum())
        # print((exp1-exp3).sum())
        
        self.optimizer.zero_grad()    
        
        # sample = detach_dict(sample)
        losses = detach_dict(losses)
        weighted_losses = detach_dict(weighted_losses)
        total_loss.detach_()

        return sample, total_loss, losses, weighted_losses

    def compute_losses(self, sample):
        losses = {}
        for loss_name in self.losses.keys():
            losses[loss_name] = self.losses[loss_name].object(sample)

        final_loss = 0 
        weighted_losses = {}
        for loss_name in losses.keys():
            weighted_losses[loss_name] = losses[loss_name] * self.losses[loss_name].weight
            final_loss += weighted_losses[loss_name]

        return final_loss, losses, weighted_losses


def create_experiment_name(cfg): 
    name = "Opt_"

    if cfg.settings.optimize_exp: 
        name += "E"
    if cfg.settings.optimize_jaw_pose: 
        name += "J"

    name += cfg.data.data_class[:4]
    name += f"-{cfg.init.geometry_type}"

    name += f"_s{cfg.init.source_sample_idx}"
    name += f"-t{cfg.init.source_sample_idx}"

    name += "_L"
    if not cfg.losses.emotion_loss.active:
        name += "-E" + "s" if cfg.losses.emotion_loss.from_source else "t"
    if not cfg.losses.video_emotion_loss.active:
        name += "-EV"  + "s" if cfg.losses.video_emotion_loss.from_source else "t"
    if not cfg.losses.lip_reading_loss.active:
        nane += "-LR"  + "s" if cfg.losses.lip_reading_loss.from_source else "t"
    if not cfg.losses.expression_reg.active:
        name += "-ExR"
    
    return name


def optimize(cfg): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emotion_feature_loss_cfg = cfg.losses.emotion_loss
    
    emotion_loss = create_emo_loss(device, 
                    emoloss=emotion_feature_loss_cfg.network_path,
                    trainable=emotion_feature_loss_cfg.trainable, 
                    emo_feat_loss=emotion_feature_loss_cfg.emo_feat_loss,
                    normalize_features=emotion_feature_loss_cfg.normalize_features, 
                    dual=False).to(device)

    # ## Video emotion loss
    # # load the video emotion classifier loss
    # video_emotion_loss_cfg = cfg.losses.video_emotion_loss 
    # video_network_folder = "/is/cluster/work/rdanecek/video_emotion_recognition/trainings/"
    
    # video_emotion_loss_cfg.network_path = str(Path(video_network_folder) / video_emotion_loss_cfg.video_network_name)
    # video_emotion_loss = create_video_emotion_loss( video_emotion_loss_cfg).to(device)
    # video_emotion_loss.feature_extractor = emotion_loss.backbone

    ## Video emotion loss
    # load the video emotion classifier loss
    video_emotion_loss_cfg = cfg.losses.video_emotion_loss 
    # video_network_folder = "/is/cluster/work/rdanecek/video_emotion_recognition/trainings/"
    
    # TODO: experiment with different nets
    # video_network_name = "2023_01_09_12-42-15_7763968562013076567_VideoEmotionClassifier_MEADP_TSC_PE_Lnce"
    # video_emotion_loss_cfg.network_path = str(Path(video_network_folder) / video_network_name)
    video_emotion_loss = create_video_emotion_loss( video_emotion_loss_cfg).to(device)
    video_emotion_loss.feature_extractor = emotion_loss.backbone



    ## Lip reading loss
    # load the lip reading loss
    lip_reading_loss_cfg = cfg.losses.lip_reading_loss
    lip_reading_loss = LipReadingLoss(device, lip_reading_loss_cfg.get('metric', 'cosine_similarity')).to(device)
    

    renderer = renderer_from_cfg(cfg.settings.renderer).to(device)


    condition_source, condition_settings = get_condition_string_from_config(cfg)
    # instantiate a data module 
    dm = MEADPseudo3DDM(
                cfg.data.input_dir, 
                cfg.data.output_dir, 
                processed_subfolder=cfg.data.processed_subfolder, 
                face_detector=cfg.data.face_detector,
                landmarks_from=cfg.data.get('landmarks_from', None),
                face_detector_threshold=cfg.data.face_detector_threshold, 
                image_size=cfg.data.image_size, 
                scale=cfg.data.scale, 
                batch_size_train=cfg.data.batching.batch_size_train,
                batch_size_val=cfg.data.batching.batch_size_val, 
                batch_size_test=cfg.data.batching.batch_size_test, 
                sequence_length_train=cfg.data.batching.sequence_length_train, 
                sequence_length_val=cfg.data.batching.sequence_length_val, 
                sequence_length_test=cfg.data.batching.sequence_length_test, 
                # occlusion_settings_train = OmegaConf.to_container(cfg.data.occlusion_settings_train), 
                # occlusion_settings_val = OmegaConf.to_container(cfg.data.occlusion_settings_val), 
                # occlusion_settings_test = OmegaConf.to_container(cfg.data.occlusion_settings_test), 
                split = cfg.data.split,
                num_workers=cfg.data.num_workers,
                include_processed_audio = cfg.data.include_processed_audio,
                include_raw_audio = cfg.data.include_raw_audio,
                drop_last=cfg.data.drop_last,
                ## end args of FaceVideoDataModule
                ## begin MEADPseudo3DDM specific params
                # training_sampler=cfg.data.training_sampler,
                landmark_types = cfg.data.landmark_types,
                landmark_sources=cfg.data.landmark_sources,
                segmentation_source=cfg.data.segmentation_source,
                inflate_by_video_size = cfg.data.inflate_by_video_size,
                preload_videos = cfg.data.preload_videos,
                test_condition_source=condition_source,
                test_condition_settings=condition_settings,
                read_video=cfg.data.get('read_video', True),
                reconstruction_type=cfg.data.get('reconstruction_type', None),
                return_appearance=cfg.data.get('return_appearance', None),
                average_shape_decode=cfg.data.get('average_shape_decode', None),
                emotion_type=cfg.data.get('emotion_type', None),
                return_emotion_feature=cfg.data.get('return_emotion_feature', None),
        )

    dm.prepare_data()
    dm.setup()

    # # flame_cfg = munchify({
    # #     "flame": { 
    # #         "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
    # #         "n_shape": 100 ,
    # #         # n_exp: 100,
    # #         "n_exp": 50,
    # #         "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy" ,
    # #     },
    # #     # use_texture: true

    # #     "flame_tex": {
    # #         "tex_type": "BFM",
    # #         "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
    # #         "n_tex": 50,
    # #     },
    # # })

    # flame_cfg = munchify({
    #     "type": "flame",
    #     "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
    #     "n_shape": 100 ,
    #     # n_exp: 100,
    #     "n_exp": 50,
    #     "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy" ,
    #     "tex_type": "BFM",
    #     "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
    #     "n_tex": 50,
    # })

    flame_cfg = cfg.settings.flame_cfg 

    # instantiate FLAME model
    flame = FlameShapeModel(flame_cfg).to(device)
    # flame = FLAME(flame_cfg.flame)
    # flame_tex = FLAMETex(flame_cfg.flame_tex)

    renderer.set_shape_model(flame.flame)

    dl = dm.val_dataloader()

    dataset = dl.dataset

    # source_sample_idx = 60 # 'M003/video/front/neutral/level_1/008.mp4'
    # target_sample_idx = 58 # 'M003/video/front/happy/level_3/024.mp4'
    source_sample_idx = cfg.init.source_sample_idx
    target_sample_idx = cfg.init.target_sample_idx

    source_sample = dataset[source_sample_idx]
    target_sample = dataset[target_sample_idx]

    # create the source parameters to optimize for 
    # source_params = {
    #     "expression": source_sample["expression"],
    #     "pose": source_sample["pose"],
    #     "shape": source_sample["shape"],
    #     "tex": source_sample["tex"],
    # }


    optimize_jaw_pose = cfg.settings.optimize_jaw_pose
    optimize_exp = cfg.settings.optimize_exp
    geometry_type = cfg.init.geometry_type
    
    variable_list = {}
    # exp_sequence = torch.autograd.Variable(source_sample['reconstruction'][geometry_type]['gt_exp'])
    if optimize_exp:
        exp_sequence = torch.tensor(source_sample['reconstruction'][geometry_type]['gt_exp'], requires_grad=True, device=device)
        variable_list['expcode'] = exp_sequence
    else:
        exp_sequence = source_sample['reconstruction'][geometry_type]['gt_exp'].to(device)

    # optimize_jaw_pose = True
    if optimize_jaw_pose:
        # jaw_sequence = torch.autograd.Variable(source_sample['reconstruction'][geometry_type]['gt_jaw'])
        jaw_sequence = torch.tensor(source_sample['reconstruction'][geometry_type]['gt_jaw'], requires_grad=True, device=device)
        # variable_list.append(jaw_sequence)
        variable_list['jawpose'] = exp_sequence
    else: 
        jaw_sequence = source_sample['reconstruction'][geometry_type]['gt_jaw'].to(device)
    # source_sample['reconstruction'][geometry_type]['gt_shape'].keys()


    target_sample_ = {
        'expcode': target_sample['reconstruction'][geometry_type]['gt_exp'],
        'jawpose': target_sample['reconstruction'][geometry_type]['gt_jaw'],
        # 'pose': source_sample['reconstruction'][target_geometry]['gt_pose'],
        'globalpose': torch.zeros_like( source_sample['reconstruction'][geometry_type]['gt_jaw']),
        'shapecode': target_sample['reconstruction'][geometry_type]['gt_shape'][None, ...].repeat(exp_sequence.shape[0], 1).contiguous(),
        'texcode': target_sample['reconstruction'][geometry_type]['gt_tex'].repeat(exp_sequence.shape[0], 1).contiguous(),
    }

    # unsqeueze the batch dimension
    for k in target_sample_:
        target_sample_[k] = target_sample_[k].unsqueeze(0).to(device)

    # do the same for source sample
    source_sample_ = {
        'expcode': exp_sequence,
        'jawpose': jaw_sequence,
        # 'pose': source_sample['reconstruction'][target_geometry]['gt_pose'],
        'globalpose': torch.zeros_like( jaw_sequence),
        'shapecode': source_sample['reconstruction'][geometry_type]['gt_shape'][None, ...].repeat(exp_sequence.shape[0], 1).contiguous(),
        'texcode': source_sample['reconstruction'][geometry_type]['gt_tex'].repeat(exp_sequence.shape[0], 1).contiguous(),
    }

    # unsqeueze the batch dimension
    for k in source_sample_:
        source_sample_[k] = source_sample_[k].unsqueeze(0).to(device)

    losses = munchify({
        "expression_reg": {
            "weight": cfg.losses.expression_reg.weight,
            "object": ExpressionRegLoss()
        }, 
        "video_emotion_loss": {
            "weight": cfg.losses.video_emotion_loss.weight,
            "object": VideoEmotionTargetLoss(video_emotion_loss=video_emotion_loss)
        }, 
        "lip_reading_loss": {
            "weight": cfg.losses.lip_reading_loss.weight,
            "object": LipReadingTargetLoss(lip_reading_loss=lip_reading_loss)
        },
        "emotion_loss": {
            "weight": cfg.losses.emotion_loss.weight,
            "object": EmotionTargetLoss(emotion_loss=emotion_loss)
        }
    })

    if optimize_jaw_pose:
        losses.jaw_pose_reg = munchify({
            "weight": cfg.losses.jaw_pose_reg.weight,
            "object": TargetJawRegLoss(cfg.losses.jaw_pose_reg.input_space, cfg.losses.jaw_pose_reg.output_space)
        })

    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    random_id = str(hash(time))
    name = "Opt"
    experiment_name = time + "_" + random_id + "_" + name
    model = OptimizationProblem(renderer, flame, losses, Path(cfg.inout.result_root) / experiment_name)
    

    wandb_logger = WandbLogger(
        project="VideoEmotionOptimization", 
        name=name,
    )
    wandb_logger.experiment

    with torch.no_grad():
        source_sample_ = model.forward(source_sample_)
        target_sample_ = model.forward(target_sample_)
    
    for cam in source_sample_['predicted_video'].keys():
        source_vid = (source_sample_['predicted_video'][cam].detach().cpu().numpy()[0] * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        target_vid = (target_sample_['predicted_video'][cam].detach().cpu().numpy()[0] * 255).astype(np.uint8).transpose(0, 2, 3, 1)

        # write the video at 25 fps
        source_vid_path = model.output_folder / f"{cam}_source.mp4" 
        target_vid_path = model.output_folder / f"{cam}_target.mp4"
        vwrite(source_vid_path, source_vid)
        vwrite(target_vid_path, target_vid)
        source_video_wandb = wandb.Video(str(source_vid_path), fps=25, format="mp4")
        target_video_wandb = wandb.Video(str(target_vid_path), fps=25, format="mp4")

        # wandb.log({f"{cam}_source": source_video_wandb, f"{cam}_target": target_video_wandb})
        wandb.log({f"{cam}_source": source_video_wandb, f"{cam}_target": target_video_wandb})


    # image-based emotion loss
    if cfg.losses.emotion_loss.from_source:
        losses.emotion_loss.object.prepare_target(source_sample_)
    else:
        losses.emotion_loss.object.prepare_target(target_sample_)

    # # emotion from target
    # if cfg.losses.video_emotion_loss.from_source:
    losses.video_emotion_loss.object.prepare_target(target_per_frame_emotion_feature=losses.emotion_loss.object.target_features)
    
    if cfg.losses.lip_reading_loss.from_source:
        losses.lip_reading_loss.object.prepare_target(source_sample_)
    else:
        losses.lip_reading_loss.object.prepare_target(target_sample_)

    if optimize_jaw_pose:
        if cfg.losses.jaw_pose_reg.from_source:
            losses.jaw_pose_reg.object.prepare_target(source_sample_)
        else: 
            losses.jaw_pose_reg.object.prepare_target(target_sample_)

    if not cfg.losses.emotion_loss.active:
        losses.pop('emotion_loss')
    if not cfg.losses.video_emotion_loss.active:
        losses.pop('video_emotion_loss')
    if not cfg.losses.lip_reading_loss.active:
        losses.pop('lip_reading_loss')
    if not cfg.losses.expression_reg.active:
        losses.pop('expression_reg')
    if optimize_jaw_pose and not cfg.losses.jaw_pose_reg.active:
        losses.pop('jaw_pose_reg')
    

    model.configure_optimizers(variable_list, cfg.optimizer.type, cfg.optimizer.lr)

    output_sample = model.optimize(source_sample_, n_iter=cfg.optimizer.n_iter, patience=cfg.optimizer.patience)

    print("Optimization done")


def main(): 

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        cfg = OmegaConf.load(config_file)
    else:
        
        cfg = Munch()
        path_to_config = Path("/is/cluster/work/rdanecek/talkinghead/trainings/2023_01_17_19-30-52_3072465630873207634_FaceFormer_MEADP_Awav2vec2T_Elinear_DFlameBertDecoder_SemlEXS_PPE_Tff_predEJ_LVmmmLmm/cfg.yaml")
        helper_config = OmegaConf.load(path_to_config)
        cfg.data = munchify(OmegaConf.to_container( helper_config.data))
        cfg.data.batching = Munch() 
        cfg.data.batching.batch_size_train = 1
        cfg.data.batching.batch_size_val = 1
        cfg.data.batching.batch_size_test = 1 
        cfg.data.batching.sequence_length_train = 1
        cfg.data.batching.sequence_length_val = 1
        cfg.data.batching.sequence_length_test = 1

        # cfg.data.batching.sequence_length_train = 25
        # cfg.data.batching.sequence_length_val = 25
        # cfg.data.batching.sequence_length_test = 25

        cfg.data.batching.sequence_length_train = 150
        cfg.data.batching.sequence_length_val = 150
        cfg.data.batching.sequence_length_test = 150

        cfg.losses = Munch() 

        ## Emotion feature loss
        emotion_feature_loss_cfg = {
            "weight": 1.0,
            "network_path": "/is/cluster/work/rdanecek/emoca/emodeca/2021_11_09_05-15-38_-8198495972451127810_EmoCnn_resnet50_shake_samp-balanced_expr_Aug_early",
            # emo_feat_loss: mse_loss
            "emo_feat_loss": "masked_mse_loss",
            "trainable": False, 
            "normalize_features": False, 
            "target_method_image": "emoca",
            "mask_invalid": "mediapipe_landmarks", # frames with invalid mediapipe landmarks will be masked for loss computation
        }
        cfg.losses.emotion_loss = Munch(emotion_feature_loss_cfg)
        cfg.losses.emotion_loss.from_source = False
        cfg.losses.emotion_loss.active = False
        cfg.losses.emotion_loss.weight = 1.0

        cfg.losses.video_emotion_loss = Munch()
        # TODO: experiment with different nets
        cfg.losses.video_emotion_loss.video_network_folder = "/is/cluster/work/rdanecek/video_emotion_recognition/trainings/"
        cfg.losses.video_emotion_loss.video_network_name = "2023_01_09_12-42-15_7763968562013076567_VideoEmotionClassifier_MEADP_TSC_PE_Lnce"
        cfg.losses.video_emotion_loss.network_path = str(Path(cfg.losses.video_emotion_loss.video_network_folder) / cfg.losses.video_emotion_loss.video_network_name)
        cfg.losses.video_emotion_loss.from_source = False
        cfg.losses.video_emotion_loss.active = True
        cfg.losses.video_emotion_loss.feature_extractor = "no"
        cfg.losses.video_emotion_loss.metric = "mse"
        # cfg.losses.video_emotion_loss.weight = 10.0
        # cfg.losses.video_emotion_loss.weight = 5.0
        cfg.losses.video_emotion_loss.weight = 2.5
        
        # video_emotion_loss_cfg.feat_extractor_cfg = "no"

        cfg.losses.lip_reading_loss = munchify(OmegaConf.to_container(helper_config.learning.losses.lip_reading_loss))
        cfg.losses.lip_reading_loss.from_source = True
        cfg.losses.lip_reading_loss.active = True
        # cfg.losses.lip_reading_loss.weight = 100.00
        cfg.losses.lip_reading_loss.weight = 0
        cfg.losses.expression_reg = Munch()
        # cfg.losses.expression_reg.weight = 1.0
        cfg.losses.expression_reg.weight = 1e-3
        cfg.losses.expression_reg.active = True

        cfg.settings = Munch()
        cfg.settings.optimize_exp = True
        cfg.settings.optimize_jaw_pose = True

        if cfg.settings.optimize_jaw_pose:
            cfg.losses.jaw_pose_reg = Munch()
            cfg.losses.jaw_pose_reg.weight = 1.0
            # cfg.losses.jaw_pose_reg.active = True
            cfg.losses.jaw_pose_reg.active = False
            cfg.losses.jaw_pose_reg.from_source = True
            cfg.losses.jaw_pose_reg.input_space = 'aa'
            cfg.losses.jaw_pose_reg.output_space = '6d'

        cfg.settings.flame_cfg = munchify({
            "type": "flame",
            "flame_model_path": "/ps/scratch/rdanecek/data/FLAME/geometry/generic_model.pkl",
            "n_shape": 100 ,
            # n_exp: 100,
            "n_exp": 50,
            "flame_lmk_embedding_path": "/ps/scratch/rdanecek/data/FLAME/geometry/landmark_embedding.npy" ,
            "tex_type": "BFM",
            "tex_path": "/ps/scratch/rdanecek/data/FLAME/texture/FLAME_albedo_from_BFM.npz",
            "n_tex": 50,
        })

        cfg.settings.renderer = munchify(OmegaConf.to_container(helper_config.model.renderer))

        cfg.optimizer = Munch()
        # cfg.optimizer.type = "adam"
        cfg.optimizer.type = "sgd"
        # cfg.optimizer.type = "lbfgs"
        # cfg.optimizer.lr = 1e-4
        cfg.optimizer.lr = 1e-6
        # cfg.optimizer.lr = 1e-2
        # cfg.optimizer.lr = 1e-1
        cfg.optimizer.n_iter = 1000
        cfg.optimizer.patience = 50
        
        cfg.init = Munch()
        cfg.init.source_sample_idx = 61 # 'M003/video/front/neutral/level_1/008.mp4'
        cfg.init.target_sample_idx = 58 # 'M003/video/front/happy/level_3/024.mp4'
        cfg.init.geometry_type = 'emoca'
        # cfg.init.geometry_type = 'spectre'

        cfg.inout = Munch()
        cfg.inout.result_root = "/is/cluster/work/rdanecek/talkinghead/optimization_results"

    optimize(cfg)


if __name__ == "__main__":    
    main()
