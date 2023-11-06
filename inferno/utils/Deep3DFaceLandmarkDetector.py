import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import torch
import pickle as pkl
from inferno.utils.FaceDetector import FaceDetector, MTCNN
import os, sys
from inferno.utils.other import get_path_to_externals 
from pathlib import Path




path_to_deep3dface = (Path(get_path_to_externals()) / "Deep3DFaceRecon_pytorch").absolute()

if str(path_to_deep3dface) not in sys.path:
    sys.path.insert(0, str(path_to_deep3dface))

# from util.detect_lm68 import load_lm_graph
from util.preprocess import crop

INPUT_SIZE = 224


# create tensorflow graph for landmark detector
def load_lm_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='net')
        img_224 = graph.get_tensor_by_name('net/input_imgs:0')
        output_lm = graph.get_tensor_by_name('net/lm:0')
        lm_sess = tf.Session(graph=graph)

    return lm_sess,img_224,output_lm


class Deep3DFaceLandmarkDetector(FaceDetector):
    """
    The facial landmark detector used by Deep3DFaceRecon_pytorch. It predicts 68 landmarks, the 3D version of them. 
    Runs on thensorflow (so problematic wrt the rest of the codebase). So probably not super useful.
    """

    def __init__(self, device = 'cuda', instantiate_detector='mtcnn'):
        self.sess, self.img_224, self.output_lm = load_lm_graph(str(path_to_deep3dface / 'checkpoints/lm_model/68lm_detector.pb'))
        self.mean_face = np.loadtxt(str(path_to_deep3dface /'util/test_mean_face.txt'))
        self.mean_face = self.mean_face.reshape([68, 2])
        self.detector = None
        if instantiate_detector == 'mtcnn':
            self.detector = MTCNN()
        elif instantiate_detector is not None: 
            raise ValueError("Invalid value for instantiate_detector: {}".format(instantiate_detector))
        

    # @profile
    @torch.no_grad()
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        if detected_faces is None: 
            bboxes, _ = self.detector.run(image)
        else:
            bboxes = [[0,0,image.shape[1],image.shape[0]]]

        kpts = []
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(image)
        for bbox in bboxes:
            # crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

            # if crop.shape[0] != crop.shape[1] and crop.shape[0] != INPUT_SIZE:
            #     crop = np.array(Image.fromarray(crop).resize((INPUT_SIZE, INPUT_SIZE)))

            cropped, scale = crop(image, bbox)

            # detect landmarks
            input_img = np.reshape(
                cropped, [1, INPUT_SIZE, INPUT_SIZE, 3]).astype(np.float32)
            landmark = self.sess.run(
                self.output_lm, feed_dict={self.img_224: input_img})

            # transform back to original image coordinate
            landmark = landmark.reshape([68, 2]) + self.mean_face
            # landmark[:, 1] = 223 - landmark[:, 1]
            landmark = landmark / scale
            landmark[:, 0] = landmark[:, 0] + bbox[0]
            landmark[:, 1] = landmark[:, 1] + bbox[1]
            landmark[:, 1] = image.shape[0] - 1 - landmark[:, 1]

            kpts += [landmark]
            plt.plot(landmark[:, 0], landmark[:, 1], 'ro')
        plt.show()

        # del lms # attempt to prevent memory leaks
        if with_landmarks:
            return bboxes, 'kpt68', kpts
        else:
            return bboxes, 'kpt68'

    @torch.no_grad()
    def detect_in_crop(self, crop):
        with torch.no_grad():
            X_recon, lms_in_crop, X_lm_hm = self.net.detect_landmarks(crop)
        lms_in_crop = utils.nn.to_numpy(lms_in_crop.reshape(1, -1, 2))
        return X_recon, lms_in_crop, X_lm_hm


