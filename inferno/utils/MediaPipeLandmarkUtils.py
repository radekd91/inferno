import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList, NormalizedLandmark
from inferno.layers.losses.MediaPipeLandmarkLosses import EMBEDDING_INDICES_NP


def mediapipe2np(landmarks): 
    # d = protobuf_to_dict(landmarks)
    array = np.zeros(shape=(len(landmarks), 3))
    for i in range(len(landmarks)):
        array[i, 0] = landmarks[i].x
        array[i, 1] = landmarks[i].y
        array[i, 2] = landmarks[i].z
    return array


def np2mediapipe(array): 
    # from munch import Munch
    landmarks = NormalizedLandmarkList()
    for i in range(len(array)):
        # landmarks += [ Munch(landmark=Munch(x=array[i, 0], y=array[i, 1], z=array[i, 2]))]
        # landmarks += [Munch(x=array[i, 0], y=array[i, 1], z=array[i, 2])]
        if array.shape[1] == 3:
            lmk = NormalizedLandmark(x=array[i, 0], y=array[i, 1], z=array[i, 2])
        else: 
            lmk = NormalizedLandmark(x=array[i, 0], y=array[i, 1], z=0.)
        landmarks.landmark.append(lmk)
    return landmarks


def draw_mediapipe_landmarks(image, landmarks_mp, normalized_coords=True, subset_to_draw=None): 
    
    image_size = image.shape[0]
    if normalized_coords: # coords are from -1 to 1, scale up to 0 to image_size
        landmarks_mp = landmarks_mp * (image_size/2) + (image_size/2)


    # # # T, C, W, H to T, W, H, C 
    # # image = image.permute(0, 2, 3, 1)
    
    # # C, W, H to W, H, C
    # image = image.permute(1, 2, 0)

    # plot the video frames with plotly
    # horizontally concatenate the frames
    # landmarks_mp_list = [] 
    # for i in range(landmarks_mp.shape[0]):
    landmarks_mp_proto = np2mediapipe(landmarks_mp / image_size)
    # landmarks_mp_list.append(landmarks_mp_proto)

    # Load drawing_utils and drawing_styles
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles
    if subset_to_draw is None:
        subset_to_draw = mp.solutions.face_mesh.FACEMESH_CONTOURS

    image_with_landmarks_mp = np.ascontiguousarray( (np.copy(image)*255).astype(np.uint8))
    mp_drawing.draw_landmarks(
        image=image_with_landmarks_mp,
        landmark_list=landmarks_mp_proto,
        connections=subset_to_draw,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style()
        )
    return image_with_landmarks_mp


def draw_mediapipe_landmark_flame_subset(image, landmarks_mp_subset, normalized_coords=True):

    # get the full landmarks_array with the correct ordering 
    landmarks_mp = -1 * np.zeros(shape=(468, 2)) 
    landmarks_mp[EMBEDDING_INDICES_NP, :] = landmarks_mp_subset

    return draw_mediapipe_landmarks(image, landmarks_mp, normalized_coords=normalized_coords,)