import cv2
import numpy as np

path = "C:/Users/Jurg Verhoeven/Documents/Cook3r/Git/Cook3r/BEANS_MASKED"
imageName = "IMG_4027.jpg"

original = cv2.imread(path+"/"+imageName)

print(original)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 500, 500)
cv2.imshow("Original", original)