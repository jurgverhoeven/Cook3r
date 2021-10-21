import os
import glob
import numpy as np
from sklearn.utils import Bunch
import Food
import cv2
import Bean_detect
import Meatball_detect
import Pasta_detect
import Pan

def fetch_data(data_path):
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.j*'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # intitialize data matrix with correct number of features
    feature_names = ['area', 'perimeter', 'prominent_hue', 'shape']
    data = np.empty((0,len(feature_names)), float)
    target = []

    # loop over the image paths
    for filename in file_list:
        food_image = cv2.imread(filename)
        pan = Pan.Pan(food_image)
        pan_image = pan.getMasked()
        label = filename.split(os.path.sep)[-2]

        food_items = []

        # Extracting features in food class.
        if label == "Meatballs":
            print("Meatball class, shape circle.")
            # food_item = Food.Meatball(food_image)
            meatball_detect = Meatball_detect.Meatball_detect()
            food_items.extend(meatball_detect.getMeatballs(pan_image))
        elif label == "Pasta":
            print("Pasta class, shape rectangle")
            # food_item = Food.Pasta(food_image)
            pasta_detect = Pasta_detect.Pasta_detect()
            food_items.extend(pasta_detect.getPasta(pan_image))
        elif label == "Beans":
            print("Bean class, shape rectangle")
            # food_item = Food.Bean(food_image)
            bean_detect = Bean_detect.Bean_detect()
            food_items.extend(bean_detect.getBeans(pan_image))
        
        for food_item in food_items:
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

