from gdl.models.temporal.Bases import Renderer
from gdl.models.Renderer import *
from gdl.utils.lbs import batch_rodrigues, batch_rigid_transform


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
            landmarks2d = sample["predicted_landmarks2d_flame_space"]
            landmarks2d_mediapipe = sample["predicted_landmarks2d_mediapipe_flame_space"]
        cam = sample["cam"]
        lightcode = sample["lightcode"]

        B = verts.shape[0]
        T = verts.shape[1]

        # batch temporal squeeze
        verts = verts.view(B*T, *verts.shape[2:])
        albedo = albedo.view(B*T, *albedo.shape[2:])
        if self.project_landmarks:
            landmarks2d = landmarks2d.view(B*T, *landmarks2d.shape[2:])
            landmarks2d_mediapipe = landmarks2d_mediapipe.view(B*T, *landmarks2d_mediapipe.shape[2:])
        cam = cam.view(B*T, *cam.shape[2:])

        # 9 and 3 are the spherical harmonics things
        lightcode = lightcode.view(B*T, 9, 3)

        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        if self.project_landmarks:
            predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
            predicted_landmarks_mediapipe = util.batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]

        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        if self.project_landmarks:
            predicted_landmarks[:, :, 1:] = -predicted_landmarks[:, :, 1:]
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
            predicted_landmarks = predicted_landmarks.view(B, T, *predicted_landmarks.shape[1:])
            predicted_landmarks_mediapipe = predicted_landmarks_mediapipe.view(B, T, *predicted_landmarks_mediapipe.shape[1:])
        predicted_trans_verts = trans_verts.view(B,T, *trans_verts.shape[1:])

        sample["predicted_video"] = predicted_images 
        sample["predicted_mask"] = predicted_mask
        if self.project_landmarks:
            sample["predicted_landmarks"] = predicted_landmarks
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
        fixed_poses = torch.tensor(cfg.fixed_poses)
        # convert aa to rotation matrix
        # fixed_poses = batch_rodrigues(
        #     fixed_poses.view(-1, 3), dtype=fixed_poses.dtype).view([-1, 3, 3]).transpose(1, 2)
        fixed_poses = batch_rodrigues(
            fixed_poses.view(-1, 3), dtype=fixed_poses.dtype).view([-1, 3, 3])

        self.register_buffer('fixed_poses', fixed_poses)

        fixed_lightcode = torch.tensor(cfg.fixed_light)
        self.register_buffer('fixed_lightcode', fixed_lightcode)
        
        self.cam_names = cfg.cam_names

    
    def forward(self, sample, train=False, input_key="verts", output_prefix="predicted_", **kwargs):
        verts = sample[input_key]
        
        # create an extra dimension for the fixed views
        B, T, = verts.shape[:2]
        C = self.fixed_cams.shape[0]
        _other_dims_v = verts.shape[2:]
        _other_dims_v_repeat = len(_other_dims_v) * [1]

        albedo = sample["albedo"]
        if albedo.ndim == 4: 
            # add temporal dimension
            # albedo = albedo.unsqueeze(1).repeat(1, T, 1, 1, 1)
            albedo = albedo.unsqueeze(1).expand(B, T, *albedo.shape[1:])
    
        _other_dims_a = albedo.shape[2:]
        _other_dims_a_repeat = len(_other_dims_a) * [1]

        # shape [B, T, #num cams, ...]
        verts = verts.unsqueeze(2)
        albedo = albedo.unsqueeze(2)

        # repeat the fixed views
        # verts = verts.repeat(1, 1, C, *_other_dims_v_repeat)
        verts = verts.expand(B, T, C, *_other_dims_v)
        verts = verts.view(B, T, C, -1, 3)
        # albedo = albedo.repeat(1, 1, C, *_other_dims_a_repeat)
        albedo = albedo.expand(B, T, C, *_other_dims_a)

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

        rendering_sample = super().forward(rendering_sample)
        
        out_name = output_prefix + "video"
        assert out_name not in sample, f"Key '{out_name}' already exists in sample. Please choose a different output_prefix to not overwrite and existing value"
        sample[out_name] = {}
        for ci, cam_name in enumerate(self.cam_names):
            sample[out_name][cam_name] = rendering_sample["predicted_video"][:, ci::C, ...]
        
        # import matplotlib.pyplot as plt 
        # plt.figure()
        # image = sample[out_name][self.cam_names[0]][0].permute( 2, 0, 3, 1).cpu().numpy()
        # # image = image.view()
        
        # plt.imshow(image[:,0,...])
        # plt.show()

        return sample

