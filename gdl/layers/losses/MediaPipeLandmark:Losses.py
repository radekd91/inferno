import numpy as np
import torch

from gdl.utils.MediaPipeFaceOccluder import left_eye_landmark_indices, right_eye_landmark_indices, mouth_landmark_indices

## MEDIAPIPE LANDMARK DESCRIPTIONS 

# LEFT EYE
# perspective of the landmarked person
LEFT_EYE_LEFT_CORNER = 263
LEFT_EYE_RIGHT_CORNER = 362 
# the upper and lower eyelid points are in correspondences, ordered from right to left (perspective of the landmarked person)
LEFT_UPPER_EYELID_INDICES = [398, 384, 385, 386, 387, 388, 466]
LEFT_LOWER_EYELID_INDICES = [382, 381, 380, 374, 373, 390, 249]

LEFT_UPPER_EYEBROW_INDICES = [336, 296, 334, 293, 300]
LEFT_LOWER_EYEBROW_INDICES = [285, 295, 282, 283, 276]

# RIGHT EYE
# perspective of the landmarked person
RIGHT_EYE_LEFT_CORNER = 133
RIGHT_EYE_RIGHT_CORNER = 33 
# the upper and lower eyelid points are in correspondences, ordered from right to left (perspective of the landmarked person)
RIGHT_UPPER_EYELID_INDICES = [246, 161, 160, 159, 158, 157, 173]
RIGHT_LOWER_EYELID_INDICES = [7  , 163, 144, 145, 153, 154, 155]

RIGHT_UPPER_EYEBROW_INDICES = [ 70,  63, 105,  66, 107]
RIGHT_LOWER_EYEBROW_INDICES = [ 46,  53,  52,  65,  55]

# MOUTH
LEFT_INNER_LIP_CORNER = 308 
LEFT_OUTTER_LIP_CORNER = 291 
RIGHT_INNER_LIP_CORNER = 78
RIGHT_OUTTER_LIP_CORNER = 61 
# from right to left, the upper and lower are in correspondence
UPPER_INNER_LIP_LINE = [191,  80, 81 , 82 , 13 , 312, 311, 310, 415]
LOWER_INNER_LIP_LINE = [ 95,  88, 178, 87 , 14 , 317, 402, 318, 324]
# from right to left, the upper and lower are in correspondence
UPPER_OUTTER_LIP_LINE = [185,  40,  39,  37,   0, 267, 269, 270, 409]
LOWER_OUTTER_LIP_LINE = [146,  91, 181,  84,  17, 314, 405, 321, 375]

# NOSE
# from up (between the eyes) downards (nose tip)
VERTICAL_NOSE_LINE = [168, 6, 197, 195, 5, 4]
# from right (next to the right nostril, just under the right nostril , under the nose) to left (landmarked person perspective)
HORIZONTAL_NOSE_LINE = [129,  98, 97,  2, 326, 327, 358]


def get_mediapipe_indices():
    # This index array contains indices of mediapipe landmarks that are selected by Timo. 
    # These include the eyes, eyebrows, nose, and mouth. Not the face contour and others. 
    # Loaded from mediapipe_landmark_embedding.npz by Timo.
    indices = np.array([276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
        55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
        381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
        144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
        168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
          0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
        87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
        308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
        415])
    return indices


def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[:, :, 2] = weights[None, :] * real_2d_kp[:, :, 2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k


def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt) #.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1))#.cuda()
                             ], dim=-1)
    # real_2d = torch.cat(landmarks_gt).cuda()

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight



def lip_dis(landmarks):
    lip_up = landmarks[:, UPPER_OUTTER_LIP_LINE + UPPER_INNER_LIP_LINE, :]
    lip_down = landmarks[:, LOWER_OUTTER_LIP_LINE + LOWER_INNER_LIP_LINE, :]
    dis = torch.sqrt(((lip_up - lip_down) ** 2).sum(2))  # [bz, 4]
    return dis


def mouth_corner_dis(landmarks):
    lip_right = landmarks[:, [LEFT_INNER_LIP_CORNER, LEFT_OUTTER_LIP_CORNER], :]
    lip_left = landmarks[:,  [RIGHT_INNER_LIP_CORNER, RIGHT_OUTTER_LIP_CORNER], :]
    dis = torch.sqrt(((lip_right - lip_left) ** 2).sum(2))  # [bz, 4]
    return dis


def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)#.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
                             ], dim=-1)
    pred_lipd = lip_dis(predicted_landmarks[:, :, :2])
    gt_lipd = lip_dis(real_2d[:, :, :2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss


def mouth_corner_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)#.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
                             ], dim=-1)
    pred_corner_d = mouth_corner_dis(predicted_landmarks[:, :, :2])
    gt_corner_d = mouth_corner_dis(real_2d[:, :, :2])

    loss = (pred_corner_d - gt_corner_d).abs().mean()
    return loss