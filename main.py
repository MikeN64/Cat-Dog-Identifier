import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm
from config import TRAIN_DIR, TEST_DIR, IMG_SIZE, MODEL_NAME


# Labelling dataset
def label_img(img):
    word_label = img.split(".")[-3]

    if word_label == "cat": return [1,0]
    elif word_label == "dog": return [0,1]

def create_train_data():
    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
        
        training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data) 

    return training_data


create_train_data()