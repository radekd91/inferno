import numpy as np
import torch

from inferno.utils.MediaPipeLandmarkLists import left_eye_landmark_indices, right_eye_landmark_indices, mouth_landmark_indices

## 68 LANDMARK DESCRIPTIONS 

# face contour 
CONTOUR_INDICES = torch.arange(0, 17).long()

# nose 
NOSE_INDICES = torch.arange(27, 36).long()


def landmark_loss(predicted_landmarks, landmarks_gt, weights=None):
    assert predicted_landmarks[..., :2].isnan().sum() == 0
    assert landmarks_gt[..., :2].isnan().sum() == 0
    loss_lmk_2d = (predicted_landmarks[..., :2] - landmarks_gt[..., :2]).abs()
    if loss_lmk_2d.ndim == 3:
        loss_lmk_2d= loss_lmk_2d.mean(dim=2)
    elif loss_lmk_2d.ndim == 4: 
        loss_lmk_2d = loss_lmk_2d.mean(dim=(2,3))
    else: 
        raise ValueError(f"Wrong dimension of loss_lmk_2d: { loss_lmk_2d.ndim}")
    if weights is None: 
        return loss_lmk_2d.mean()
    if weights.sum().abs() < 1e-8:
        return torch.tensor(0)
    if weights is not None:
        w = weights / torch.sum(weights)
        loss_lmk_2d = w * loss_lmk_2d
        return loss_lmk_2d.sum()
    return loss_lmk_2d 


def landmark_loss_indices(predicted_landmarks, landmarks_gt, indices, weights=None):
    return landmark_loss(predicted_landmarks[..., indices, :2], landmarks_gt[..., indices, :2], weights=weights)

def landmark_loss_contour(predicted_landmarks, landmarks_gt, weights=None):
    if CONTOUR_INDICES.device != predicted_landmarks.device:
        CONTOUR_INDICES.to(device=predicted_landmarks.device)
    return landmark_loss_indices(predicted_landmarks, landmarks_gt, CONTOUR_INDICES, weights=weights)

def landmark_loss_nose(predicted_landmarks, landmarks_gt, weights=None):
    if NOSE_INDICES.device != predicted_landmarks.device:
        NOSE_INDICES.to(device=predicted_landmarks.device)
    return landmark_loss_indices(predicted_landmarks, landmarks_gt, NOSE_INDICES, weights=weights)


