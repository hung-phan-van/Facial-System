#!/usr/bin/env python
import os
from threading import Thread
import socket
import struct  # to send `int` as  `4 bytes`
import time   # for test
import cv2
from imutils.video import VideoStream
import face
import datetime
import requests
import json
import argparse
import base64
import numpy as np
import imagezmq
os.environ["CUDA_VISIBLE_DEVICES"]="0"
SERVER_ADDR   = os.environ('SERVER_ADDR')

def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # parameter
    parser.add_argument('--data-dir', default='',
                        help='dataset or avatar directory')
    parser.add_argument('--model', default='',
                        help='path to deep learning model')
    parser.add_argument('--model-epoch', default='', help='epoch')
    # rtsp link rtsp://admin:dou123789@10.0.0.15:554/cam/realmonitor?channel=1&subtype=0
    parser.add_argument('--rtsp', default='', help='rtsp link')
    parser.add_argument('--classifier', default='', help='trained classifier')
    parser.add_argument('--gpu', default='', help='using gpu=1 or cpu')
    parser.add_argument('--trackingDuration', type=float,
                        help='time to tracking')
    parser.add_argument('--check-size', type=bool,
                        default=False, help='check size of face')
    parser.add_argument('--socket-server', default='',
                        help='socket server for handle result response')
    parser.add_argument('--crop-camera', default='400,900,470,1150',
                        help='crop camera height and width')
    parser.add_argument('--type-in-out', default='in', help='camera in or out')
    parser.add_argument('--client_id', default='0', help='client id')
    parser.add_argument('--face_detect_type', default='MTCNN', help='MTCNN/RETINA')

    args = parser.parse_args()
    return args

class Streaming(Thread):
    def __init__(self, args):
        Thread.__init__(self)
        self.face_recognition = face.Recognition(args)
        self.args = args
        self.frame_interval = 1

    def processingImage(self, frame_count, image_hub):
        result = []
        name , frame = image_hub.recv_image()
        regcozieArr = []
        if (frame_count % self.frame_interval) == 0:
            faces = self.face_recognition.identify(frame)
            if len(faces):
                regcozieArr = self.face_recognition.recognition(faces)
        for face in regcozieArr:
            _, buffer = cv2.imencode(
                '.jpg', cv2.cvtColor(face.vimage, cv2.COLOR_RGB2BGR))
            image = base64.b64encode(buffer).decode("utf-8")
            if face.name != 'Unknown':
                result.append({
                    'image': image,
                    'name': face.name,
                    'conf': face.confidence
                })
        return result

    def run(self):
        frame_count = 0
        image_hub = imagezmq.ImageHub(SERVER_ADDR)
        while True:
            start_reg_time = time.time()
            rslt = self.processingImage(frame_count, image_hub)

            if len(rslt):
                print('Result: %s' % len(rslt))
                array = []
                for item in rslt:
                    payload = {
                        'image': item['image'],
                        'userId': item['name'],
                        'conf': item['conf']
                    }
                    array.append(payload)

                info_detection = {"location": str(
                    args.client_id), "array": array}
                jsonPayload = json.dumps(info_detection)
                headers = {'Content-Type': 'application/json'}
                
                r = requests.post(self.args.socket_server, 
                                data=jsonPayload, headers=headers)
                print(r.status_code, r.reason)
                el_start_reg_time = time.time() - start_reg_time
                print('Processing  (Internet) time %s' % el_start_reg_time)
                print('---------------------------------')
                
            frame_count += 1
            if frame_count == 10000:
                frame_count = 0
            image_hub.send_reply(b'OK')
       
if __name__ == '__main__':
    args = None
    args = parse_args()
    Streaming(args).run()
