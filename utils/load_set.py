import os
from PIL import Image

def load_set(data_path, set_type):
    abs_path = os.path.join(data_path, set_type)
    i = 0
    X_set = []
    Y_set = []
    for label_folder in os.listdir(abs_path):
        label_folder_path = os.path.join(abs_path, label_folder)
        for image in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, image)
            print(f"Set : {set_type}, Class: {label_folder}, Image: {image}", end="\r")
            X_set.append(image_path)
            Y_set.append(label_folder)
        print()
    return X_set, Y_set



