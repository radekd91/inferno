import numpy as np

from gdl.datasets.ImageDatasetHelpers import bbox2point, bbpoint_warp


def align_face(image, landmarks, landmark_type, scale_adjustment, target_size_height, target_size_width=None,):
    """
    Returns an image with the face aligned to the center of the image.
    :param image: The full resolution image in which to align the face. 
    :param landmarks: The landmarks of the face in the image (in the original image coordinates).
    :param landmark_type: The type of landmarks. Such as 'kpt68' or 'bbox' or 'mediapipe'.
    :param scale_adjustment: The scale adjustment to apply to the image.
    :param target_size_height: The height of the output image.
    :param target_size_width: The width of the output image. If not provided, it is assumed to be the same as target_size_height.
    :return: The aligned face image. The image will be in range [0,1].
    """
    # landmarks_for_alignment = "mediapipe"
    left = landmarks[:,0].min()
    top =  landmarks[:,1].min()
    right =  landmarks[:,0].max()
    bottom = landmarks[:,1].max()

    old_size, center = bbox2point(left, right, top, bottom, type=landmark_type)
    size = (old_size * scale_adjustment).astype(np.int32)

    img_warped, lmk_warped = bbpoint_warp(image, center, size, target_size_height, target_size_width, landmarks=landmarks)

    return img_warped

