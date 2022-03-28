import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import json
import models as model_md
from torchvision import transforms as tf

def read_json(filename):
    with open(filename, 'r') as fp:
        content = json.load(fp)

    return content
def get_model():
  models_cfg = read_json(os.environ('MODEL_CFG'))
  embedding_cfg = models_cfg['encoder']
  print('name', embedding_cfg['name'])
  print(embedding_cfg['args'])
  emb_model = getattr(model_md, embedding_cfg['name'])(**embedding_cfg['args'])
  emb_model.load_checkpoint(('cuda:0'))
  return emb_model

class FaceModel:
  def __init__(self, args):
    self.model = None
    self.model = get_model()
    transforms_default = tf.Compose([
                          tf.ToTensor(),
                          tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])

    self.inference_batch = {'transforms':transforms_default}
  
  def get_feature(self, list_image):

    embedding = self.model.inference_batch([list_image], **self.inference_batch)
    return embedding.reshape(512).tolist()
