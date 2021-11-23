import os
import glob
import numpy as np
from sklearn.utils import Bunch
import Food
import cv2

def fetch_data(data_path):
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.jpg'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # intitialize data matrix with correct number of features
    feature_names = ['area', 'perimeter', 'prominent_hue', 'shape']
    data = np.empty((0,len(feature_names)), float)
    target = []

    # loop over the image paths
    for filename in file_list:
        food_image = cv2.imread(filename)


        label = filename.split(os.path.sep)[-2]

        food_item = 0

        # Extracting features in food class.
        if label == "Meatball":
            print("Meatball class, shape circle.")
            food_item = Food.Meatball(food_image)
        elif label == "Pasta":
            print("Pasta class, shape rectangle")
            food_item = Food.Pasta(food_image)
        elif label == "Bean":
            print("Bean class, shape rectangle")
            food_item = Food.Bean(food_image)
        
        target.append(label)

        # retrieve features from class
        features = np.array((food_item.getArea(), food_item.getPerimeter(), food_item.getProminentHue(), food_item.getShape()))
        features = (features)
        print("[INFO] contour features: {}".format(features))

        # append features to data matrix
        data = np.append(data, np.array([features]), axis=0)

    unique_targets = np.unique(target)
    print("[INFO] targets found: {}".format(unique_targets))

    dataset = Bunch(data = data,
                    target = target,
                    unique_targets = unique_targets,
                    feature_names = feature_names)

    return dataset

