import json
import numpy as np

def load_embedding(source):
    with open(source) as json_data:
        obj = json.load(json_data)
        return obj

def get_embedding_from_image(folder, filename, em):
    return {
        'features': em[folder][filename]['features'], # type nparray.tolist
        'directionL': em[folder][filename]['directionL']
    }

def get_embedding_from_folder(folder, em):
    return em[folder]