from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np
from PIL import Image 
import cv2

def normalize_pixels(input_image):
    input_image_norm = input_image.astype('float32')
    input_image_norm = input_image_norm / 255.0
    return input_image_norm

def load_model_face_mask():
    model = load_model(os.environ('MASK_MODEL'))
    return model

def prepare_image(img, IMG_DIM):
    im_pil = Image.fromarray(img)
    resample = Image.NEAREST
    im_resized = im_pil.resize(IMG_DIM, resample)
    img_array = img_to_array(im_resized)
    return [img_array]
