from __future__ import print_function
import os
print(os.getcwd())
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAA')
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time

from Face_Detect.models.retinaface import RetinaFace
from Face_Detect.models.net_slim import Slim
from Face_Detect.models.net_rfb import RFB

from Face_Detect.data import cfg_mnet, cfg_slim, cfg_rfb
from Face_Detect.layers.functions.prior_box import PriorBox
from Face_Detect.utils.nms.py_cpu_nms import py_cpu_nms
import cv2


from Face_Detect.utils.box_utils import decode, decode_landm
from Face_Detect.utils.timer import Timer
import onnxruntime
import onnx
import time
from easydict import EasyDict as edict
import os 
cur_path = os.getcwd()
args = edict({
    'trained_model': 'Face_Detect/weights/RBF_Final.pth',
    'network': 'RFB',
    'origin_size': True,
    'long_side': 640,
    'save_folder': 'Face_Detect/widerface_evaluate/widerface_txt/',
    'cpu': False,
    'confidence_threshold': 0.9,
    'top_k': 500,
    'nms_threshold': 0.4,
    'keep_top_k':50,
    'save_image': True,
    'vis_thres': 0.6
})

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}
def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# if __name__ == '__main__':

def load_net():
    torch.set_grad_enabled(False)
    # paths  = os.listdir(args.i)
    # paths  = sorted(paths)
    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)
    #v1
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    #v2
    # net = load_model(net, args.trained_model, args.cpu).eval().

    return net, cfg

