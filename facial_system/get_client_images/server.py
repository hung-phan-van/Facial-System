import os
#import pygame
from threading import Thread
import socket
import struct  # to send `int` as  `4 bytes`
import time   # for test
import cv2
# from imutils.video import VideoStream
import datetime
import requests
import json
import argparse
import base64
import numpy as np
import imagezmq
SERVER_ADDR  = os.environ("SERVER_ADDR")
jpeg_quality = 95

def run():
    frame_count = 0
    infor_hub = imagezmq.ImageHub(SERVER_ADDR, REQ_REP = True)
    while True:
        start_reg_time = time.time()
        _, jpg_buffer = infor_hub.recv_jpg()
        frame = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
        print(frame.shape)

if __name__ == '__main__':

    run()
