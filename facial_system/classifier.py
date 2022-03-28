import argparse
import cv2
import sys
import numpy as np
import face_model
import os
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.svm import SVC
import time
import arrow
import helper
import json

args = None
def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # parameter
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--model-type', default='', help='deep learning model')
  parser.add_argument('--model-epoch', default='', help='epoch')
  parser.add_argument('--model', default='', help='path to deep learning model')
  parser.add_argument('--gpu', default='', help='using gpu=1 or cpu')
  parser.add_argument('--name', default='', help='trained model name')
  parser.add_argument('--em-file', default=True, type=bool,help='Allow/Not allow embedding file')
  parser.add_argument('--max-iter', default=1000, type=int,help='Max loop ANN model')
  args = parser.parse_args()
  return args

def train_net(args):
    model = face_model.FaceModel(args)
    em_file = args.em_file
    file_change = 'file_change'

    em = None
    if em_file == True:
        if os.path.exists(args.data_dir + '.json'):
            em = helper.load_embedding(args.data_dir + '.json')
            print('Loading embedding files done')
    # get train folder and class
    train_folder = args.data_dir
    modelName = args.model_type
    if args.name:
        name = args.name + '.sav'
    else:
        name = arrow.utcnow().format('YYYYMMDDHHmmss') + '.sav'
    X = []
    Y = []
    if em != None and  file_change in em:
        em.pop(file_change, None)
        with open(args.data_dir + '.json', 'w') as outfile:
            json.dump(em, outfile)
        classes = list(em)
        classes.sort()
        dict = {}
        classlist = []
        for index in range(len(classes)):
            classlist.append(index)
            dict[str(index)] = classes[index]
        
        for classIndex in classlist:
            folder = dict[str(classIndex)]
            for filename in list(em[folder]):
                embedding = np.asarray(em[folder][filename]['features'])
                X.append(embedding)
                Y.append(classIndex)
        print('Train X %s' % len(X))
        print('Train Y %s' % len(Y))
        if modelName == 'svm':
            clf = SVC(kernel="linear", C=10, probability=True, verbose=True)
        if modelName == 'ann':
            print('MAX iteration', args.max_iter)
            clf = MLPClassifier(verbose=True, validation_fraction=0.0, hidden_layer_sizes=(len(classlist)*2, ), tol=0.00055, n_iter_no_change=25, max_iter = args.max_iter, batch_size = 1024)
        if modelName == 'rf':
            clf =  RandomForestClassifier(n_estimators=2000, verbose=2)

        clf.fit(X,Y)
        pickle.dump(clf, open('classifiers/' + name, 'wb'))
        print('----- Training complete -----')

def main():
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

