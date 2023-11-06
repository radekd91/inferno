import torch 
import inferno.layers.losses.MediaPipeLandmarkLosses as mp_loss


def masking(tensor, mask):
    # make mask have the same ndim as tensor by adding dimensions that are missing (if any)
    while mask.ndim < tensor.ndim:
        mask = mask[..., None]
    return tensor * mask


class LandmarkLoss(object):

    def __init__(self, device, loss_type='l2', dim='2D'):
        self.device = device
        self.loss_type = loss_type
        self.dim = dim
        if self.dim == '2D':
            self.dim_ = 2
        elif self.dim == '3D':
            self.dim_ = 3
        else:
            raise ValueError('Dimension not supported: {}'.format(self.dim))

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        pred = pred[..., :self.dim_]
        target = target[..., :self.dim_]
        if mask is not None:
            pred = self.mask(pred, mask)
            target = self.mask(target, mask)
        if self.loss_type == 'l2':
            loss = torch.mean(torch.norm(pred - target, dim=-1))
        elif self.loss_type == 'l1':
            loss = torch.mean(torch.abs(pred - target))
        else:
            raise ValueError('Loss type not supported: {}'.format(self.loss_type))
        return loss
    
    def mask(self, tensor, mask):
        # make mask have the same ndim as tensor by adding dimensions that are missing (if any)
        return masking(tensor, mask)

    def to(self, device, *args, **kwargs):
        self.device = device
        return self


class SubsetLandmarkLoss(LandmarkLoss):

    def __init__(self, indices_to_keep, device, loss_type='l2'):
        super().__init__(device, loss_type)
        self.indices_to_keep = indices_to_keep

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        pred = pred[..., self.indices_to_keep, :]
        target = target[..., self.indices_to_keep, :]
        return super().__call__(pred, target, mask=mask)
           
    def to(self, device, *args, **kwargs):
        self.device = device
        self.indices_to_keep = self.indices_to_keep.to(device)
        return self
    

class FanContourLandmarkLoss(SubsetLandmarkLoss):

    def __init__(self, device, loss_type='l2'):
        # we only want the face landmarks of the face contour and the nose (no eyes, no lips, no eyebrows)
        contour_landmarks = torch.arange(0, 17, dtype=torch.long)
        nose_landmarks = torch.arange(27, 36, dtype=torch.long)
        indices_to_keep = torch.cat([contour_landmarks, nose_landmarks])
        super().__init__(indices_to_keep, device, loss_type)
        self.to(self.device)

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        return super().__call__(pred, target, mask=mask)


class MediaPipeLandmarkLoss(LandmarkLoss):

    def __init__(self, device, loss_type='l1'):
        assert loss_type == "l1"
        super().__init__(device, loss_type)

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        # this landmark loss only corresponds to a subset of mediapipe landmarks, see mp_loss.EMBEDDING_INDICES
        if mask is not None:
            pred = self.mask(pred, mask)
            target = self.mask(target, mask)
        return mp_loss.landmark_loss(pred, target)


class MediaPipeRelativeLoss(object):

    def __init__(self, device, loss_type='l1'):
        assert loss_type == "l1"
        super().__init__() 
        self.device = device
        self.loss_type = loss_type


    def to(self, device, *args, **kwargs):
        self.device = device
        return self

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        raise NotImplementedError("MediaPipeRelativeLoss not implemented yet")


class MediaPipeMouthCornerLoss(MediaPipeRelativeLoss):

    def __init__(self, device, loss_type='l1'):
        super().__init__(device, loss_type)

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        # this landmark loss only corresponds to a subset of mediapipe landmarks, see mp_loss.EMBEDDING_INDICES
        if mask is not None:
            pred = masking(pred, mask)
            target = masking(target, mask)
        return mp_loss.mouth_corner_loss_v2(pred, target)
        # return mp_loss.mouth_corner_loss(pred, target)
    
    def to(self, device, *args, **kwargs):
        self.device = device
        return self


class MediaPipleEyeDistanceLoss(MediaPipeRelativeLoss):
    
    def __init__(self, device, loss_type='l1'):
        super().__init__(device, loss_type)

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        if mask is not None:
            pred = masking(pred, mask)
            target = masking(target, mask)
        # this landmark loss only corresponds to a subset of mediapipe landmarks, see mp_loss.EMBEDDING_INDICES
        return mp_loss.eyed_loss_v2(pred, target)
        # return mp_loss.eyed_loss(pred, target)

    def to(self, device, *args, **kwargs):
        self.device = device
        return self


class MediaPipeLipDistanceLoss(MediaPipeRelativeLoss):

    def __init__(self, device, loss_type='l1'):
        super().__init__(device, loss_type)

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        if mask is not None:
            pred = masking(pred, mask)
            target = masking(target, mask)
        # this landmark loss only corresponds to a subset of mediapipe landmarks, see mp_loss.EMBEDDING_INDICES
        return mp_loss.lipd_loss_v2(pred, target)
        # return mp_loss.lipd_loss(pred, target)
    
    def to(self, device, *args, **kwargs):
        self.device = device
        return self


class PhotometricLoss(object):

    def __init__(self, device, loss_type='l2'):
        super().__init__() 
        self.device = device
        self.loss_type = loss_type 

    def to(self, device, *args, **kwargs):
        self.device = device
        return self


    def __call__(self, pred, target, *args, mask=None, **kwargs):
        if mask is not None:
            pred = masking(pred, mask)
            target = masking(target, mask)
        if self.loss_type == 'l2':
            loss = torch.mean(torch.norm(pred - target, dim=-1))
        elif self.loss_type == 'l1':
            loss = torch.mean(torch.abs(pred - target))
        else:
            raise ValueError('Loss type not supported: {}'.format(self.loss_type))
        return loss
    

class VertexLoss(object):

    def __init__(self, device, loss_type='l2'):
        super().__init__() 
        self.device = device
        self.loss_type = loss_type 

    def to(self, device, *args, **kwargs):
        self.device = device
        return self

    def __call__(self, pred, target, *args, mask=None, **kwargs):
        ## pred and target are both B x N x 3 or B x T x N x 3
        if mask is not None:
            pred = masking(pred, mask)
            target = masking(target, mask)
        if self.loss_type == 'l2':
            loss = torch.mean(torch.norm(pred - target, dim=-1))
        elif self.loss_type == 'l1':
            loss = torch.mean(torch.abs(pred - target))
        else:
            raise ValueError('Loss type not supported: {}'.format(self.loss_type))
        return loss


class GaussianRegLoss(object): 

    def __init__(self) -> None:
        super().__init__() 
        pass

    def __call__(self, pred, *args, **kwargs):
        return torch.sum(pred ** 2) / 2
    
    def to(self, *args, **kwargs):
        return self
    

class LightRegLoss(object): 

    def __init__(self) -> None:
        pass

    def __call__(self, lightcode, *args, **kwargs):
        if lightcode.ndim == 2:
            B_ = lightcode.shape[0]
        elif lightcode.ndim == 3: 
            B, T = lightcode.shape[:2]
            B_ = B * T
        else: 
            raise ValueError("lightcode must have ndim 2 or 3")
        lightcode_ = lightcode.view(B_,  -1 ,3)
        return ((torch.mean(lightcode_, dim=2, keepdim=True) - lightcode_) ** 2).mean()
    
    def to(self, *args, **kwargs):
        return self