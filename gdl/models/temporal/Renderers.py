from gdl.models.temporal.Bases import Renderer
from gdl.models.Renderer import *
from gdl.utils.lbs import batch_rodrigues, batch_rigid_transform
import torchvision.transforms.functional as F_v


class FlameRenderer(Renderer):

    def __init__(self, cfg):
        super().__init__() 
        mask = imread(cfg.face_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_mask = F.interpolate(mask, [cfg.uv_size, cfg.uv_size])
        self.register_buffer('uv_face_mask', uv_face_mask)
        mask = imread(cfg.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [cfg.uv_size, cfg.uv_size])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)

        # TODO: detail part rendering not implemented 

        self.render = SRenderY(cfg.image_size, obj_filename=cfg.topology_path,
                               uv_size=cfg.uv_size)  
        
        self.project_landmarks = cfg.get("project_landmarks", True)

    def forward(self, sample): 
        verts = sample["verts"]
        albedo = sample["albedo"]
        if self.project_landmarks:
            landmarks2d = None 
            landmarks2d_mediapipe = None
            if "predicted_landmarks2d_flame_space" in sample.keys():
                landmarks2d = sample["predicted_landmarks2d_flame_space"]
            if "predicted_landmarks2d_mediapipe_flame_space" in sample.keys():
                landmarks2d_mediapipe = sample["predicted_landmarks2d_mediapipe_flame_space"]
        cam = sample["cam"]
        lightcode = sample["lightcode"]

        B = verts.shape[0]
        T = verts.shape[1]

        # batch temporal squeeze
        verts = verts.view(B*T, *verts.shape[2:])
        # albedo = albedo.view(B*T, *albedo.shape[2:])
        albedo = albedo.reshape(B*T, *albedo.shape[2:])
        if self.project_landmarks:
            if landmarks2d is not None:
                landmarks2d = landmarks2d.view(B*T, *landmarks2d.shape[2:])
            if landmarks2d_mediapipe is not None:
                landmarks2d_mediapipe = landmarks2d_mediapipe.view(B*T, *landmarks2d_mediapipe.shape[2:])
        cam = cam.view(B*T, *cam.shape[2:])

        # 9 and 3 are the spherical harmonics things
        lightcode = lightcode.view(B*T, 9, 3)

        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        if self.project_landmarks:
            if landmarks2d is not None:
                predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
            if landmarks2d_mediapipe is not None:
                predicted_landmarks_mediapipe = util.batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]

        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        if self.project_landmarks:
            if landmarks2d is not None:
                predicted_landmarks[:, :, 1:] = -predicted_landmarks[:, :, 1:]
            if landmarks2d_mediapipe is not None:
                predicted_landmarks_mediapipe[:, :, 1:] = -predicted_landmarks_mediapipe[:, :, 1:]

        outputs = self.render(verts, trans_verts, albedo, lightcode)
        effective_batch_size = verts.shape[0]

        # mask
        mask_face_eye = F.grid_sample(self.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                      outputs['grid'].detach(), align_corners=False)
        predicted_images = outputs['images']
        predicted_mask = mask_face_eye * outputs['alpha_images']

        # batch temporal unsqueeze
        predicted_images = predicted_images.view(B, T, *predicted_images.shape[1:])
        predicted_mask = predicted_mask.view(B, T, *predicted_mask.shape[1:])
        if self.project_landmarks:
            if landmarks2d is not None:
                predicted_landmarks = predicted_landmarks.view(B, T, *predicted_landmarks.shape[1:])
            if landmarks2d_mediapipe is not None:
                predicted_landmarks_mediapipe = predicted_landmarks_mediapipe.view(B, T, *predicted_landmarks_mediapipe.shape[1:])
        predicted_trans_verts = trans_verts.view(B,T, *trans_verts.shape[1:])

        sample["predicted_video"] = predicted_images 
        sample["predicted_mask"] = predicted_mask
        if self.project_landmarks:
            if landmarks2d is not None:
                sample["predicted_landmarks"] = predicted_landmarks
            if landmarks2d_mediapipe is not None:
                sample["predicted_landmarks_mediapipe"] = predicted_landmarks_mediapipe
        sample["trans_verts"] = predicted_trans_verts
        return sample


    def render_coarse_shape(self, sample, indices=None, **kwargs):
        if indices is None: 
            indices = np.arange(sample["verts"].shape[0], dtype=np.int64)
        verts = sample["verts"][indices]
        trans_verts = sample["trans_verts"][indices]
        coarse_shape_rendering = self.render.render_shape(verts, trans_verts, **kwargs)
        return coarse_shape_rendering


class FixedViewFlameRenderer(FlameRenderer):

    def __init__(self, cfg):
        super().__init__(cfg) 
        
        # shape [#num cams, cam params]
        fixed_cams = torch.tensor(cfg.fixed_cams) 
        self.register_buffer('fixed_cams', fixed_cams)
        # self.register_buffer('fixed_cams', -fixed_cams)

        # shape [#num cams, 3]
        fixed_poses_aa = torch.tensor(cfg.fixed_poses)
        self.register_buffer('fixed_poses_aa', fixed_poses_aa)
        # convert aa to rotation matrix
        # fixed_poses = batch_rodrigues(
        #     fixed_poses.view(-1, 3), dtype=fixed_poses.dtype).view([-1, 3, 3]).transpose(1, 2)
        fixed_poses = batch_rodrigues(
            fixed_poses_aa.view(-1, 3), dtype=fixed_poses_aa.dtype).view([-1, 3, 3])

        self.register_buffer('fixed_poses', fixed_poses)

        fixed_lightcode = torch.tensor(cfg.fixed_light)
        self.register_buffer('fixed_lightcode', fixed_lightcode)
        
        self.cam_names = cfg.cam_names

        self.cut_out_mouth = cfg.get("cut_out_mouth", False)
        if self.cut_out_mouth: 
            self.mouth_crop_width = cfg.get("mouth_crop_width", 96) # the default of SPECTRE
            self.mouth_crop_height = cfg.get("mouth_crop_height", 96)
            self.mouth_window_margin = cfg.get("mouth_crop_height", 12)
            self.mouth_landmark_start_idx = 48
            self.mouth_landmark_stop_idx = 68

    def set_shape_model(self, shape_model):
        self.shape_model = shape_model

    def forward(self, sample, train=False, input_key_prefix="gt_", output_prefix="predicted_", **kwargs):
        verts = sample[input_key_prefix + "vertices"]
        jaw = sample[input_key_prefix +  "jaw"]
        
        # create an extra dimension for the fixed views
        B, T, = verts.shape[:2]
        C = self.fixed_cams.shape[0]
        _other_dims_v = verts.shape[2:]
        _other_dims_v_repeat = len(_other_dims_v) * [1]


        # rec_types = []
        # if 'gt_exp' in sample:
        #     rec_types += [None]
        # else: 
        #     rec_types += sample["reconstruction"].keys()

        albedo = sample["gt_albedo"]
        if albedo.ndim == 4: 
            # add temporal dimension
            # albedo = albedo.unsqueeze(1).repeat(1, T, 1, 1, 1)
            albedo = albedo.unsqueeze(1).expand(B, T, *albedo.shape[1:])
    
        _other_dims_a = albedo.shape[2:]
        _other_dims_a_repeat = len(_other_dims_a) * [1]

        # shape [B, T, #num cams, ...]
        verts = verts.unsqueeze(2)
        jaw = jaw.unsqueeze(2)
        albedo = albedo.unsqueeze(2)

        # repeat the fixed views
        # verts = verts.repeat(1, 1, C, *_other_dims_v_repeat)
        verts = verts.expand(B, T, C, *_other_dims_v)
        verts = verts.view(B, T, C, -1, 3)
        # albedo = albedo.repeat(1, 1, C, *_other_dims_a_repeat)
        albedo = albedo.expand(B, T, C, *_other_dims_a)
        jaw = jaw.expand(B, T, C, jaw.shape[-1])

        # collapse the cam dimension 
        # shape [B, T, #num cams, ...] -> [B, T * #num cams, ...]
        # verts = verts.view(B, T * C, *_other_dims)
        # verts = verts.view(B, T * C, -1, 3)
        # albedo = albedo.view(B, T * C, *_other_dims)

        # cams 
        # shape [B, T * #num cams, 3]
        # cams = self.fixed_cams.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        cams = self.fixed_cams.unsqueeze(0).unsqueeze(0).expand(B, T, C, self.fixed_cams.shape[-1])

        # light 
        # shape [B, T * #num cams, 9, 3]
        # lightcode = self.fixed_lightcode.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1, 1)
        lightcode = self.fixed_lightcode.unsqueeze(0).unsqueeze(0).expand(B, T, C, *self.fixed_lightcode.shape[-2:])

        # pose the vertices 
        verts_posed  = verts @ self.fixed_poses
        # verts = verts @ self.fixed_poses.transpose(1, 2)

        if self.project_landmarks:
            # self.fixed_poses_aa is [C, 3]. expand to [B, T, C, 3]
            poses_aa = self.fixed_poses_aa.unsqueeze(0).unsqueeze(0)
            neck_aa = torch.zeros_like(poses_aa)
            eyes_aa = torch.zeros_like(poses_aa)
            poses_aa = poses_aa.expand(B, T, C, self.fixed_poses_aa.shape[-1])
            neck_aa = neck_aa.expand(B, T, C, self.fixed_poses_aa.shape[-1])
            eyes_aa = eyes_aa.expand(B, T, C, self.fixed_poses_aa.shape[-1])

            full_pose = torch.cat([poses_aa, neck_aa, jaw, eyes_aa], dim=-1)

            landmarks_2d_posed = self.shape_model._vertices2landmarks2d(verts_posed.view(B * T * C, *verts_posed.shape[3:]), full_pose.view(B * T * C, *full_pose.shape[3:]) )
            landmarks_2d_posed = landmarks_2d_posed.view(B, T * C, *landmarks_2d_posed.shape[1:])

        rendering_sample = {}
        rendering_sample["verts"] = verts_posed
        rendering_sample["albedo"] = albedo
        rendering_sample["cam"] = cams
        rendering_sample["lightcode"] = lightcode

        for key in rendering_sample:
            # collapse the cam dimension 
            # shape [B, T, #num cams, ...] -> [B, T * #num cams, ...]
            dims_after_cam = rendering_sample[key].shape[3:]
            # rendering_sample[key] = rendering_sample[key].view(B, T * C, *dims_after_cam)
            rendering_sample[key] = rendering_sample[key].reshape(B, T * C, *dims_after_cam)

        if self.project_landmarks:
            rendering_sample["predicted_landmarks2d_flame_space"] = landmarks_2d_posed

        rendering_sample = super().forward(rendering_sample)
        
        out_vid_name = output_prefix + "video"
        out_landmark_name = output_prefix + "landmarks_2d"
        # out_verts_name = output_prefix + "trans_verts"
        assert out_vid_name not in sample, f"Key '{out_vid_name}' already exists in sample. Please choose a different output_prefix to not overwrite and existing value"
        assert out_landmark_name not in sample, f"Key '{out_landmark_name}' already exists in sample. Please choose a different output_prefix to not overwrite and existing value"
        sample[out_vid_name] = {}
        sample[out_landmark_name] = {}
        if self.cut_out_mouth: 
            out_mouth_vid_name = output_prefix + "mouth_video"
            sample[out_mouth_vid_name] = {}
        # sample[out_verts_name] = {}
        for ci, cam_name in enumerate(self.cam_names):
            sample[out_vid_name][cam_name] = rendering_sample["predicted_video"][:, ci::C, ...]
            # sample[out_name][cam_name] = sample[out_name][cam_name].view(B, T, *sample[out_name][cam_name].shape[1:])
            # sample[out_verts_name][cam_name] = rendering_sample["trans_verts"][:, ci::C, ...]
            if self.project_landmarks:
                sample[out_landmark_name][cam_name] = rendering_sample["predicted_landmarks"][:, ci::C, ...]

            if self.cut_out_mouth: 
                sample[out_mouth_vid_name][cam_name] = []
                for bi in range(B):
                    sample[out_mouth_vid_name][cam_name] += [self.cut_mouth(sample[out_vid_name][cam_name][bi], sample[out_landmark_name][cam_name][bi])]
                sample[out_mouth_vid_name][cam_name] = torch.stack(sample[out_mouth_vid_name][cam_name], dim=0)

        # ## plot the landmakrs over the video for debugging and sanity checking
        # import matplotlib.pyplot as plt 
        # plt.figure()
        # image = sample[out_vid_name][self.cam_names[0]][0].permute( 2, 0, 3, 1).cpu().numpy()
        # mouth_image = sample[out_vid_name + "_mouth"][self.cam_names[0] + ][0].unsqueeze(1).permute( 2, 0, 3, 1).cpu().numpy()
        # if self.project_landmarks:
        #     # trans_verts = sample[out_verts_name][self.cam_names[0]][0].cpu().numpy()[..., :2]  * image.shape[-2] / 2 + image.shape[-2] / 2
        #     landmarks = sample[out_landmark_name][self.cam_names[0]][0].cpu().numpy() * image.shape[-2] / 2 + image.shape[-2] / 2
        # # image = image.view()
        # frame_num = 2
        # plt.imshow(image[:,frame_num,...])
        # if self.project_landmarks:
        #     # plt.scatter(trans_verts[frame_num,0], trans_verts[frame_num,1], s=3)
        #     plt.scatter(landmarks[frame_num, :,0], landmarks[frame_num, :,1], s=3)
        # plt.show()
        # plt.figure()
        # plt.imshow(mouth_image[:,frame_num,...])
        # plt.show()
        return sample


    def cut_mouth(self, images, landmarks, convert_grayscale=True):
        """ function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""

        mouth_sequence = []

        image_size = images.shape[-1] / 2

        landmarks = landmarks * image_size + image_size
        for frame_idx, frame in enumerate(images):
            window_margin = min(self.mouth_window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = landmarks[frame_idx-window_margin:frame_idx + window_margin + 1].mean(dim=0)
            smoothed_landmarks += landmarks[frame_idx].mean(dim=0) - smoothed_landmarks.mean(dim=0)

            center_x, center_y = torch.mean(smoothed_landmarks[self.mouth_landmark_start_idx:self.mouth_landmark_stop_idx], dim=0)

            center_x = center_x.round()
            center_y = center_y.round()

            height = self.mouth_crop_height//2
            width = self.mouth_crop_width//2

            threshold = 5

            if convert_grayscale:
                img = F_v.rgb_to_grayscale(frame).squeeze()
            else:
                img = frame

            if center_y - height < 0:
                center_y = height
            if center_y - height < 0 - threshold:
                raise Exception('too much bias in height')
            if center_x - width < 0:
                center_x = width
            if center_x - width < 0 - threshold:
                raise Exception('too much bias in width')

            if center_y + height > img.shape[-2]:
                center_y = img.shape[-2] - height
            if center_y + height > img.shape[-2] + threshold:
                raise Exception('too much bias in height')
            if center_x + width > img.shape[-1]:
                center_x = img.shape[-1] - width
            if center_x + width > img.shape[-1] + threshold:
                raise Exception('too much bias in width')

            mouth = img[...,int(center_y - height): int(center_y + height),
                                 int(center_x - width): int(center_x + round(width))]

            mouth_sequence.append(mouth.unsqueeze(0))

        mouth_sequence = torch.stack(mouth_sequence,dim=0)
        return mouth_sequence
