from Food_recognition import Food_recognition
import Pan
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
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
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
from videoCapture import videoCapture

if __name__ == "__main__":
    cap = videoCapture("C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Filmpjes/beans2.mov")
    frame = cap.getFrame()
    clf = load("decision_tree_model.joblib")
    food_recognition = Food_recognition.Food_recognition()

    while(cap.isSucces):
        cv2.waitKey(1)
        pan = Pan.Pan(frame)
        panImage = pan.getMasked()
        cv2.imshow("Masked pan", panImage);
        foods = food_recognition.recognize(panImage)

        if foods != 0:
            totalprobabilities = [0, 0, 0, 0, 0, 0, 0]
            for food in foods:
                features = np.array((food.getArea(), food.getPerimeter(), food.getProminentHue(), food.getShape()))
                features = features.reshape(-1, 4)
                probability = clf.predict_proba(features)[0]

                totalprobabilities[0] += probability[0]*100
                totalprobabilities[1] += probability[1]*100
                totalprobabilities[2] += probability[2]*100
                totalprobabilities[3] += probability[3]*100
                totalprobabilities[4] += probability[4]*100
                totalprobabilities[5] += probability[5]*100

            if(totalprobabilities[0] > 0):
                totalprobabilities[0] /= len(foods)
            if(totalprobabilities[1] > 0):
                totalprobabilities[1] /= len(foods)
            if(totalprobabilities[2] > 0):
                totalprobabilities[2] /= len(foods)
            if(totalprobabilities[3] > 0):
                totalprobabilities[3] /= len(foods)
            if(totalprobabilities[4] > 0):
                totalprobabilities[4] /= len(foods)
            if(totalprobabilities[5] > 0):
                totalprobabilities[5] /= len(foods)

            print("The frame contains "+str(totalprobabilities[0])+"% beans, "+str(totalprobabilities[1])+"% carrots_with_water and "+str(totalprobabilities[2])+"% Fish_sticks"+str(totalprobabilities[3])+"% Meatballs, "+str(totalprobabilities[4])+"% Pasta and "+str(totalprobabilities[5])+"% Potatoes_with_water")

        frame = cap.getFrame()
        frame = cap.getFrame()
        frame = cap.getFrame()
        frame = cap.getFrame()
        frame = cap.getFrame()
        frame = cap.getFrame()
        frame = cap.getFrame()
        frame = cap.getFrame()
        frame = cap.getFrame()



