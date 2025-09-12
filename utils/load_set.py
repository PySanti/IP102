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
            X_set.append(image_path)
            Y_set.append(int(label_folder))
    return X_set, Y_set



