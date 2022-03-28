import argparse
import cv2
import sys
import numpy as np
import face_model
import os
import pickle
import time
import arrow
import json
import helper

args = None
def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # parameter
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--model-epoch', default='', help='epoch')
  parser.add_argument('--model', default='', help='path to deep learning model')
  parser.add_argument('--gpu', default='', help='using gpu=1 or cpu')
  parser.add_argument('--em-file', default=True, type=bool,help='Allow/Not allow embedding file')

  args = parser.parse_args()
  return args

def extract(args):
    # 3D model points.
    print('____GET EMBEDDING____')
    print(args.data_dir, )
    model_points = np.array([
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (0.0, 0.0, 0.0),  # Nose tip
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    dist_coeffs = np.zeros((4, 1))
    size = [112, 112]  # 1000, 1600
    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype=float
    )
    # extract
    em_file = args.em_file
    if em_file == True:
        if os.path.exists(args.data_dir + '.json'):
            data = helper.load_embedding(args.data_dir + '.json')
            print('Loading embedding files done')
        else:
            data = {}
    else:
        data = {}
    model = face_model.FaceModel(args)
    train_folder = args.data_dir
    # get class
    folders = os.listdir(train_folder)
    folders = [f for f in folders if f != '.DS_Store' and f != '00Unknown']

    # count total file
    total = 0
    file_change = 'file_change'
    count_change = 0
    folder_change = []
    folder_empty = []
    for folder in folders:
        files = os.listdir(train_folder + '/' + folder)
        files = [f for f in files if f != '.DS_Store']
        if len(files) == 0:
            folder_empty.append(folder)
        else:
            total = total + len(files)

    print('Total %s' % total)
    for folder in folder_empty:
      folders.remove(folder)

    if len(data) > 0:
        for folder in data:
            if folder not in folders:
                count_change += 1
                folder_change.append(folder)
        for folder in folder_change:
            data.pop(folder, None)
    
    data[file_change] = {}
    dataset = {}
    data_delete = {}
    for folder in folders:
        dataset[folder] = []
        files = os.listdir(train_folder + '/' + folder)
        files = [f for f in files if f != '.DS_Store']
        for f in files:
            filename = f.split('.')[0]
            dataset[folder].append(filename)

    for folder in data:
        data_delete[folder] = []
        for filename in data[folder]:
            if filename not in dataset[folder]:
                count_change += 1
                data_delete[folder].append(filename)

    for folder in data_delete:
        for filename in data_delete[folder]:
            data[folder].pop(filename, None)

    for folder in folders: 
        if folder not in data:
            data[folder] = {}
        files = os.listdir(train_folder + '/' + folder)
        files = [f for f in files if f != '.DS_Store']
        
        for f in files:
            filename = f.split('.')[0]
            ext = f.split('.')[1]
            if filename not in data[folder]:
                count_change += 1
                data[folder][filename] = {}
                img = cv2.imread(train_folder + '/' + folder + '/' + f)
            # extract embedding
                img = cv2.resize(img, (112,112))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
                if img is not None:
                    embedding = model.get_feature(img)
                    data[folder][filename]['features'] = embedding
                    print('Extract embedding',folder, filename)
                else:
                    data[folder].pop(filename, None)
                    data.pop(folder, None)
    if count_change == 0:
        data.pop(file_change, None)
    with open(train_folder + '.json', 'w') as outfile:
        json.dump(data, outfile)
    print('Dump features file done.')
    time.sleep(5)
    
def main():
    global args
    args = parse_args()
    extract(args)
if __name__ == '__main__':
    main()

