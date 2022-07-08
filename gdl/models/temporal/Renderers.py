from gdl.models.temporal.Bases import Renderer
from gdl.models.Renderer import *


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

    def forward(self, sample): 
        verts = sample["verts"]
        albedo = sample["albedo"]
        landmarks2d = sample["predicted_landmarks2d_flame_space"]
        landmarks2d_mediapipe = sample["predicted_landmarks2d_mediapipe_flame_space"]
        cam = sample["cam"]
        lightcode = sample["lightcode"]

        B = verts.shape[0]
        T = verts.shape[1]

        # batch temporal squeeze
        verts = verts.view(B*T, *verts.shape[2:])
        albedo = albedo.view(B*T, *albedo.shape[2:])
        landmarks2d = landmarks2d.view(B*T, *landmarks2d.shape[2:])
        landmarks2d_mediapipe = landmarks2d_mediapipe.view(B*T, *landmarks2d_mediapipe.shape[2:])
        cam = cam.view(B*T, *cam.shape[2:])

        # 9 and 3 are the spherical harmonics things
        lightcode = lightcode.view(B*T, 9, 3)

        # world to camera
        trans_verts = util.batch_orth_proj(verts, cam)
        predicted_landmarks = util.batch_orth_proj(landmarks2d, cam)[:, :, :2]
        predicted_landmarks_mediapipe = util.batch_orth_proj(landmarks2d_mediapipe, cam)[:, :, :2]

        # camera to image space
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
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
        predicted_landmarks = predicted_landmarks.view(B, T, *predicted_landmarks.shape[1:])
        predicted_landmarks_mediapipe = predicted_landmarks_mediapipe.view(B, T, *predicted_landmarks_mediapipe.shape[1:])
        predicted_trans_verts = trans_verts.view(B,T, *trans_verts.shape[1:])

        sample["predicted_video"] = predicted_images 
        sample["predicted_mask"] = predicted_mask
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

