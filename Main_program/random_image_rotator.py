import numpy as np
import cv2
import os
from random import randrange

path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans/Meatballs"

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

if __name__ == "__main__":
    for root, dirs, files in os.walk(path):
        for filename in files:
            image = cv2.imread(path+"/"+filename)
            random_int = randrange(1, 360)
            rotated = rotate_image(image, random_int)
            cv2.imwrite(path+"/rotated/"+str(random_int)+filename, rotated)