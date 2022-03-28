import argparse
import json
import requests

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

    parser.add_argument('--socket-fake', default='',
                        help='socket server for handle result response')

    parser.add_argument('--socket-mask', default='',
                        help='socket server for handle result response')

    parser.add_argument('--face_detect_type', default='MTCNN', help='MTCNN / RETINA / RETINATENSORT')
    parser.add_argument('--face-mask', type=bool,
                        default=False, help='warning face mask')
    parser.add_argument('--trace-data', default='',
                        help='socket server for handle result response')
    parser.add_argument('--socket-mask-option', default='',
                        help='socket server for handle result response')
    args = parser.parse_args()
    return args

class SubFunction:
    def __init__(self, args):
        self.args = args
    def sent_fake_information(self, image, userId, conf, type_, fake_mask=False):
        array = []
        payload = {
            'image': image,
            'userId': userId,
            'conf': conf,
            'type': type_,
        }
        array.append(payload)
        print(payload['userId'])
        print(payload['conf'])
        info_detection = {"location": str(
            self.args.client_id), "array": array}
        jsonPayload = json.dumps(info_detection)
        headers = {'Content-Type': 'application/json'}
        if fake_mask == True:
            r = requests.post(self.args.socket_mask, 
            data=jsonPayload, headers=headers)
        else:
            r = requests.post(self.args.socket_fake, 
            data=jsonPayload, headers=headers)
        print(r.status_code)
    def sent_result_recognition(self, rslt):
        array = []
        for item in rslt:
            payload = {
                'image': item['image'],
                'userId': item['name'],
                'conf': item['conf'],
                'type': self.args.type_in_out,
            }
            array.append(payload)

        info_detection = {"location": str(
            self.args.client_id), "array": array}
        jsonPayload = json.dumps(info_detection)
        headers = {'Content-Type': 'application/json'}
        r = requests.post(self.args.socket_server, 
                        data=jsonPayload, headers=headers)
        print(r.status_code, r.reason)  


