# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 Fran�ois Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os
from random import randint
import cv2
import numpy as np
import requests

# import tensorflow as tf
from scipy import misc
import time
import math
import uuid
import face_model
# from mtcnn.mtcnn import MTCNN
import json
import helper
from numpy import dot
from numpy.linalg import norm
import threading
import pymongo
import time
from datetime import datetime as dt
import copy
import torch
import torch.backends.cudnn as cudnn
from Face_Detect import load_model as detect_face
from Face_Detect.layers.functions.prior_box import PriorBox
from Face_Detect.utils.box_utils import decode, decode_landm
from Face_Detect.utils.nms.py_cpu_nms import py_cpu_nms


root_path = os.environ('ROOT_PATH')
gpu_memory_fraction = 0.7
debug = False

vu_debug_facial_tracing = False
vu_tracking_number = 0

trackingConfidence = {}
avg_trackingConfidence = {}
class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None

class Recognition:
    def __init__(self, args):
        self.detect = Detection(args)
        self.encoder = Encoder(args)
        self.identifier = Identifier(args)
        self.args = args
        self.net, self.cfg = detect_face.load_net()
        self.dict_faces = {}

    def identify(self, image):
        faces = self.detect.find_faces(image, self.net, self.cfg)
        return faces

    def uuid(self):
        millis = int(round(time.time() * 1000))
        return str(millis)

    def save_unknown(self, faces):
        count = 0
        list_insert = []

        for face in faces:
            time =dt.now().strftime('%Y-%m-%d %H:%M:%S')
            user_name = face.name
            confidence = face.confidence
            saveBGRImageS = cv2.cvtColor(face.image, cv2.COLOR_RGB2BGR)
            nname = self.uuid()+ '_'+ str(count) + '_'+ str(user_name) + '_'+ '.jpg'

            cv2.imwrite(root_path + '/00Unknown/' + nname, saveBGRImageS,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # save crop image 112x112
            count = count + 1
            image_name = nname
            mydict = { "user_name": user_name, "confidence":confidence, 'image': image_name, 'time': time}
            list_insert.append(mydict)

    def recognition(self, faces):
        s = time.time()
        new_faces = []
        new_faces_unknow = []
        name_in_frame = []
        for i, face in enumerate(faces):
            face.embedding = self.encoder.generate_embedding(face)
            rslt = self.identifier.identify(face, self.encoder)
            name, conf, obj = rslt[0:3]
            name_in_frame.append(name)
            if name not in self.dict_faces:
                self.dict_faces[name] = {}
                self.dict_faces[name]['conf'] = [conf]
                self.dict_faces[name]['obj'] = [obj]
                self.dict_faces[name]['start_time'] = time.time()
            else:
                self.dict_faces[name]['conf'].append(conf)
                self.dict_faces[name]['obj'].append(obj)

        for name in list(self.dict_faces):
            list_conf = self.dict_faces[name]['conf']
            list_obj = self.dict_faces[name]['obj']
            start_time =  self.dict_faces[name]['start_time']
            check_condition = sum(i >= 99  for i in list_conf) >= 1 or sum(i >= 98  for i in list_conf) >= 2 or sum(i >= 97 for i in list_conf) >= 3 or sum(i >= 96  for i in list_conf) >= 4
            if check_condition:
                idx = np.argmax(np.array(list_conf))
                print('Success: ',name, list_conf)
                face = list_obj[idx]
                face.name = name
                face.confidence = list_conf[idx]
                new_faces.append(copy.deepcopy(face))
                del self.dict_faces[name]
            elif time.time() - start_time > 30:
                print('Fail: ', name, list_conf)
                for idx, value in enumerate(list_obj):
                    list_obj[idx].confidence = list_conf[idx]
                    list_obj[idx].name = name
                    new_faces_unknow.append(list_obj[idx])
                del self.dict_faces[name]
        if len(new_faces_unknow):
            print('save to unknow ...')
            s_save_unknow = time.time()
            self.save_unknown(new_faces_unknow)
            print('Time save unknow image: ', time.time() - s_save_unknow)
        return new_faces

class Identifier:
    def __init__(self, args):
        self.args = args
        self.em = None
        if os.path.exists(args.data_dir + '.json'):
            self.em = helper.load_embedding(args.data_dir + '.json')
            print('Loading embedding files done')

        if self.em != None:
            classes = list(self.em)
            classes.sort()
        else:
            classes = os.listdir(args.data_dir)
            classes = [f for f in classes if f != '.DS_Store' and f != '00Unknown'].sort()
            # add code
            folder_empty = []
            print(args.data_dir)
            for folder in classes:
                files = os.listdir(args.data_dir + '/' + folder)
                files = [f for f in files if f != '.DS_Store']
                if len(files) == 0:
                    folder_empty.append(folder)
            for folder in folder_empty:
                classes.remove(folder)
            
        self.dict = {}
        self.classlist = []
        for index in range(len(classes)):
            self.classlist.append(index)
            self.dict[str(index)] = classes[index]
        print('dataset classes %s' % self.classlist)
        print('dataset dict %s' % self.dict)
        # load model
        self.model = pickle.load(open(args.classifier, 'rb'))
        self.result = ['NA', 0, 0, 0]

    def identify(self, face, encoder):
        global avgTracking
        if face.embedding is not None:
            # classify check
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(
                len(best_class_indices)), best_class_indices]
            confidence = int(best_class_probabilities[0] * 100)
            className = self.dict[str(best_class_indices[0])]
            self.result[0] = className
            self.result[1] = confidence
            self.result[2] = face
            return self.result


class Encoder:
    def __init__(self, args):
        self.model = face_model.FaceModel(args)

    def generate_embedding(self, face):
        embedding = self.model.get_feature(face.image)
        return embedding

    def generate_embedding_cv2(self, image):
        img = cv2.resize(image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        embedding = self.model.get_feature(img)
        return embedding

class Detection:
    def __init__(self, args):
        
        self.args = args
        self.check_size = args.check_size
        self.statictic_find_face = []
 
    def find_faces(self, img_raw, net,cfg,origin_size=True, confidence_threshold=0.97, top_k=500, nms_threshold=0.01, keep_top_k=50, vis_thres=0.97):
        s_find_faces = time.time()
        resize = 1
        img = np.float32(img_raw)
        im_shape = img.shape
        im_height, im_width, _ = img.shape
        
        resize_ = 0
        if im_height > im_width:
            resize_ = round(256.0/im_width, 1)
        else:
            resize_ = round(256.0/im_height, 1)
        img = cv2.resize(img_raw, (0, 0), fx=resize_, fy=resize_)
        img = np.float32(img)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        device = torch.device("cuda")
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        # net = load_net()
        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > float(confidence_threshold))[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        faces = []
        for b in dets:
            if b[4] < vis_thres or min(b[0:4]) <0:
                continue
            b = list(map(int, b))
            x,y,w,h, c = int(b[0] / resize_), int(b[1] / resize_), int(b[2] / resize_), int(b[3] / resize_), b[4]
            if y < h and x < w:
                face = Face()
                face_img = img_raw[y:h, x:w]
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                min_value = min(x,y, img_raw.shape[0] - h, img_raw.shape[1] - w)
                if min_value > 50:
                    min_value = 50
                if min_value > 0 and y-min_value >= 0  and h+min_value <= img_raw.shape[0] and x-min_value >=0 and w+min_value <= img_raw.shape[1]:
                    face.vimage = cv2.cvtColor(img_raw[y-min_value:h+min_value, x-min_value:w+min_value], cv2.COLOR_BGR2RGB)
                else:
                    face.vimage = face_img
                face_img = cv2.resize(face_img, (112,112))  
                face.image = face_img
                faces.append(face)
        if len(faces):
            diff = time.time() -s_find_faces
            self.statictic_find_face.append(diff)
            if len(self.statictic_find_face) > 10000:
                print("AVG TIME FIND FACE RETINA",sum( self.statictic_find_face)/len( self.statictic_find_face) )
                self.statictic_find_face = [] 
        return faces
