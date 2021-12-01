from Food_recognition import Food_recognition
import Pan
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import Food
import os
from Fetch_data import fetch_data
from sklearn import tree
import graphviz 
from joblib import dump, load
from sklearn.metrics import plot_confusion_matrix, classification_report
import Food_recognition
from sklearn.utils import Bunch
import glob

test_data_path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/TestDataset/Beans"

if __name__ == "__main__":
    clf = load("decision_tree_model.joblib")
    food_recognition = Food_recognition.Food_recognition()
    print("[INFO] loading images...")
    p = os.path.sep.join([test_data_path, '**', '*.j*'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))
    for filename in file_list:
        image = cv2.imread(filename)
        pan = Pan.Pan(image)
        panImage = pan.getMasked()
        foods = food_recognition.recognize(panImage)


        if foods != 0:
            totalprobabilities = [0, 0, 0]
            for food in foods:
                # retrieve features from class
                features = np.array((food.getArea(), food.getPerimeter(), food.getProminentHue(), food.getShape()))
                features = features.reshape(-1, 4)

                probability = clf.predict_proba(features)[0]

                totalprobabilities[0] += probability[0]*100
                totalprobabilities[1] += probability[1]*100
                totalprobabilities[2] += probability[2]*100

            if(totalprobabilities[0] > 0):
                totalprobabilities[0] /= len(foods)
            if(totalprobabilities[1] > 0):
                totalprobabilities[1] /= len(foods)
            if(totalprobabilities[2] > 0):
                totalprobabilities[2] /= len(foods)

        print("The image: "+filename+" contains "+str(totalprobabilities[0])+"% beans, "+str(totalprobabilities[1])+"% meatballs and "+str(totalprobabilities[2])+"% pasta")



