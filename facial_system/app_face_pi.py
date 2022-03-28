from flask import Flask, current_app, request, render_template
import requests
import cv2
import numpy as np
import base64
import datetime
import os
import face
import argparse
import shutil
from collections import Counter
import json
from flask_socketio import SocketIO
import queue
from threading import Thread
import datetime
import random


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


app = Flask(__name__)
socketio = SocketIO(app)
face_recognition = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # parameter
    parser.add_argument('--data-dir', default='',
                        help='dataset or avatar directory')
    parser.add_argument('--model', default='',
                        help='path to deep learning model')
    parser.add_argument('--model-epoch', default='', help='epoch')
    parser.add_argument('--classifier', default='', help='trained classifier')
    parser.add_argument('--gpu', default='', help='using gpu=1 or cpu')
    parser.add_argument('--trackingDuration', type=float,
                        help='time to tracking')
    parser.add_argument('--check-size', type=bool,
                        default=False, help='check size of face')
    parser.add_argument('--socket-server', default='',
                        help='socket server for handle result response')
    parser.add_argument('--image', default='', help='input image')
    args = parser.parse_args()
    return args


def recreate_folder(folder):
    if os.path.isdir(folder) == True:
        shutil.rmtree(folder)
    os.makedirs(folder)


def post_ui(rslt):

    str_in_out = "in"
    array = []
    for item in rslt:
        payload = {
            'image': item['image'],
            'userId': item['name'],
            'conf': item['conf'],
            'type': str_in_out,
        }
        array.append(payload)

    client_id = 4
    socket_server = os.environ('SOCKET_SERVER')

    info_detection = {"location": str(client_id), "array": array}
    jsonPayload = json.dumps(info_detection)
    headers = {'Content-Type': 'application/json'}
    r = requests.post(socket_server,
                      data=jsonPayload, headers=headers)
    print(r.status_code, r.reason)


def save_frame(frame):

    now = datetime.datetime.now()
    image_name = str(now.hour) + str(now.minute) + \
        str(now.second) + "_" + str(random.randint(0, 9999))
    print(image_name)
    # cv2.imwrite("/home/ml/tuan/images_namhai/" + image_name + ".jpg", frame)


@socketio.on('face_message')
def face_message(message):

    global face_recognition

    frame = readb64(message)
    # save_frame(frame)

    faces = face_recognition.identify(frame)
    regcozieArr = face_recognition.recognition(faces)

    result = []
    for face in regcozieArr:
        ret, buffer = cv2.imencode(
            '.jpg', cv2.cvtColor(face.vimage, cv2.COLOR_RGB2BGR))
        image = base64.b64encode(buffer).decode("utf-8")
        if face.name != 'Unknown':
            result.append({
                "image": image,
                "name": face.name,
                "conf": face.confidence
            })

    if len(result) > 0:
        post_ui(result)
    socketio.emit('ProcessFrame', {'data': 'Yes'})


if __name__ == '__main__':
    args = None
    args = parse_args()
    face_recognition = face.Recognition(args)
    recreate_folder("/home/ml/tuan/images_namhai/")
    print("init server")
    socketio.run(app, host='10.0.11.144', port=8888)
    print("server running")
