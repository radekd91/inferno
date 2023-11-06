####################################################################################################
## The following code is copied over from the original insightface.model_zoo.model_zoo.py file 
## and adapted to work with torch instead of onnxruntime.
###################################################################################################

# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 

import os
import os.path as osp
import glob
# import onnxruntime
# from .arcface_onnx import *
# from .retinaface import *
#from .scrfd import *
# from insightface.model_zoo.landmark import *
# from insightface.model_zoo.attribute import Attribute
from insightface.utils import download_onnx
from .RetinaFaceTorch import RetinaFace
import onnx2torch

__all__ = ['get_model']


# class PickableInferenceSession(onnxruntime.InferenceSession): 
#     # This is a wrapper to make the current InferenceSession class pickable.
#     def __init__(self, model_path, **kwargs):
#         super().__init__(model_path, **kwargs)
#         self.model_path = model_path

#     def __getstate__(self):
#         return {'model_path': self.model_path}

#     def __setstate__(self, values):
#         model_path = values['model_path']
#         self.__init__(model_path)

class ModelRouter:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file

    def get_model(self, **kwargs):
        # session = PickableInferenceSession(self.onnx_file, **kwargs)
        # print(f'Applied providers: {session._providers}, with options: {session._provider_options}')
        # input_cfg = session.get_inputs()[0]
        # input_shape = input_cfg.shape
        # outputs = session.get_outputs() ### THIS OUTPUTS: ['None', 3, 192, 192] (for the retinaface model antelopev2/1k3d68.onnx )
        # onnx_model = onnx2torch.ImportModel(self.onnx_file)
        # pytorch_model = onnx_model.convert_to_pytorch()
        pytorch_model = onnx2torch.convert(self.onnx_file)

        assert 'antelopev2/1k3d68.onnx' in self.onnx_file, "Only antelopev2/1k3d68.onnx is supported (the retinaface model)"
        return RetinaFace(model=pytorch_model) 

        # # if len(outputs)>=5:
        #     # return RetinaFace(model_file=self.onnx_file) 
        # elif input_shape[2]==112 and input_shape[3]==112:
        #     raise NotImplementedError("ArcFaceONNX is not implemented")
        #     return ArcFaceONNX(model_file=self.onnx_file, session=session)
        # elif input_shape[2]==192 and input_shape[3]==192:
        #     raise NotImplementedError("Landmark is not implemented")
        #     return Landmark(model_file=self.onnx_file, session=session)
        # elif input_shape[2]==96 and input_shape[3]==96:
        #     raise NotImplementedError("Attribute is not implemented")
        #     return Attribute(model_file=self.onnx_file, session=session)
        # else:
        #     #raise RuntimeError('error on model routing')
        #     return None

def find_onnx_file(dir_path):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob("%s/*.onnx" % dir_path)
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]

def get_model(name, **kwargs):
    root = kwargs.get('root', '~/.insightface')
    root = os.path.expanduser(root)
    model_root = osp.join(root, 'models')
    allow_download = kwargs.get('download', False)
    if not name.endswith('.onnx'):
        model_dir = os.path.join(model_root, name)
        model_file = find_onnx_file(model_dir)
        if model_file is None:
            return None
    else:
        model_file = name
    if not osp.exists(model_file) and allow_download:
        model_file = download_onnx('models', model_file, root=root)
    assert osp.exists(model_file), 'model_file should exist'
    assert osp.isfile(model_file), 'model_file should be file'
    router = ModelRouter(model_file)
    model = router.get_model()
    return model

