import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Định nghĩa hàm load_data
def load_data(data_dir):
    images = []
    labels = []

    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')

    if not os.path.exists(images_dir) or not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Either '{images_dir}' or '{annotations_dir}' does not exist.")

    for img_file in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image '{img_path}'")
            continue
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)

        json_file = os.path.splitext(img_file)[0] + '.json'
        json_path = os.path.join(annotations_dir, json_file)

        if not os.path.exists(json_path):
            print(f"Warning: JSON file '{json_path}' does not exist for image '{img_file}'")
            continue

        with open(json_path, 'r') as f:
            annotation = json.load(f)
     
        labels.append(annotation)

    return np.array(images), labels

# Gọi hàm load_data sau khi nó đã được định nghĩa
images, _ = load_data('data')
