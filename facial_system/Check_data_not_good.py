import json
import os
import helper
import pickle
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
if os.path.exists('../../dataset_2/allstaff' + '.json'):
    em = helper.load_embedding('../../dataset_2/allstaff.json')
    print('Loading embedding files done')

if em != None:
    classes = list(em)
    classes.sort()
dict = {}
classlist = []
for index in range(len(classes)):
    classlist.append(index)
    dict[str(index)] = classes[index]

# load model
model = pickle.load(open('classifiers/all_ann_ds2.sav', 'rb'))

def predict_confidence(em):
    dict_results = {}
    for i in em:
        if i not in dict_results:
            dict_results[i] = []
        for image in list(em[i]):
            predictions = model.predict_proba([em[i][image]['features']])
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(
                len(best_class_indices)), best_class_indices]
            confidence = int(best_class_probabilities[0] * 100)
            className = dict[str(best_class_indices[0])]
            if i != className:
                print("Error ", i ,"to", className, confidence)
            dict_results[i].append(confidence)
    print(dict_results)

def pair_index_distance(list_images):
    list_two_images = []
    for i in range(len(list_images) - 1):
        for j in range(i+1, len(list_images)):
            list_two_images.append([list_images[i], list_images[j]])
    return list_two_images

def distanct_between_images_same_class(em):
    dict_results = {}
    list_avg = []
    for i in em:
        if i not in dict_results:
            dict_results[i] = []
            list_images = list(em[i])
        list_two_images = pair_index_distance(list_images)
        list_distance = []
        list_cosin = []
        for two_image in list_two_images:
            em_a = em[i][two_image[0]]['features']
            em_b = em[i][two_image[1]]['features']
            dst = distance.euclidean(em_a[0],em_b[0])
            a= np.array(em_a)
            b= np.array(em_b)
            c = np.sqrt((sum(a**2)))
            d = np.sqrt((sum(b**2)))
            cosin = a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))
            list_cosin.append((cosin))
            list_distance.append(round(dst,6))

        if list_distance==[]:
            continue
        a, b = selectionsort(list_distance, list_two_images)

        list_avg.append([(sum(list_distance)/len(list_distance)), i])

def selectionsort(list_distance, list_two_images):
    for i in range(len(list_distance)-1):
        min_idx = i
        for j in range(i+1, len(list_distance)):
            if list_distance[j] < list_distance[min_idx]:
                min_idx = j
        list_distance[min_idx], list_distance[i] = list_distance[i], list_distance[min_idx]
        list_two_images[min_idx], list_two_images[i] = list_two_images[i], list_two_images[min_idx]
    return list_distance, list_two_images
distanct_between_images_same_class(em)