from Food_recognition import Food_recognition
import Pan
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import Food

if __name__ == "__main__":
    food_path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_masked_warped/Meatballs/IMG_3571.jpg"

    food_image = cv2.imread(food_path)

    cv2.imshow("Food image", food_image)

    pan = Pan.Pan(food_image)
    cv2.imshow("Masked pan", pan.getMasked())

    food_recognition = Food_recognition()

    food_items = food_recognition.recognize(pan.getMasked())

    cv2.waitKey(0)




