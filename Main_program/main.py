from Food_recognition import Food_recognition
import Pan
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import Food
import os

food_path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans"


if __name__ == "__main__":
    foodItemsList = []
    food_recognition = Food_recognition()
    for root, dirs, files in os.walk(food_path):
        for dir in dirs:
            for root1, dirs1, files1 in os.walk(food_path+"/"+dir):
                for filename in files1:
                    print("Filename: "+filename)
                    food_image = cv2.imread(food_path+"/"+dir+"/"+filename)
                    # # cv2.imshow("Food image", food_image)
                    pan = Pan.Pan(food_image)
                    # cv2.imshow("Masked pan", pan.getMasked())
                    food_items = food_recognition.recognize(pan.getMasked())
                    foodItemsList.append(food_items)
    print(foodItemsList)
    cv2.waitKey(0)




