from __future__ import print_function
import os
print(os.getcwd())
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAA')
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2

# import sys
# sys.path.insert(0, "/home/cyber/Dat/Face-Detector-1MB-with-landmark")
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import onnxruntime
import onnx
import time
from easydict import EasyDict as edict
import os 
from torch2trt import torch2trt

cur_path = os.getcwd()
args = edict({
    'trained_model': 'Face_Detect/weights/RBF_Final.pth',
    'network': 'RFB',
    'origin_size': True,
    'long_side': 640,
    'save_folder': 'Face_Detect/widerface_evaluate/widerface_txt/',
    'cpu': False,
    'confidence_threshold': 0.9,
    'top_k': 5000,
    'nms_threshold': 0.4,
    'keep_top_k':750,
    'save_image': True,
    'vis_thres': 0.6
})
# parser = argparse.ArgumentParser(description='Test')
# parser.add_argument('-m', '--trained_model', default='./weights/RBF_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
# parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
# parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
# parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
# parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
# parser.add_argument('--i')
# parser.add_argument('--o')

# args = parser.parse_args()


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
    # trained_model = "./weights/RBF_Final.pth"
    # device = torch.device("cuda")
    # model = net = load_model(net, args.trained_model, args.cpu).eval().to(device)

    # sample = torch.ones((1, 3, args.long_side, args.long_side)).cuda()
    # model_trt = torch2trt(model, [sample])


    # trained_model = "./weights/RBF_Final.pth"
    device = torch.device("cuda")
    model = load_model(net, args.trained_model, args.cpu).eval().to(device)

    sample = torch.ones((1, 3, args.long_side, args.long_side)).cuda()
    model_trt = torch2trt(model, [sample])
    # return model_trt, cfg

    return model_trt, cfg

