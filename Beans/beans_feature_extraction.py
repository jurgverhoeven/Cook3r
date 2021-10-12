import cv2
import numpy as np

path = "C:\\Users\\Jurg Verhoeven\\Documents\\Cook3r\\Git\\Cook3r\\Beans\\BEANS_MASKED"
imageName = "IMG_4036.bmp"

original = cv2.imread("C:\\Users\\Jurg Verhoeven\\Documents\\Cook3r\\Git\\Cook3r\\Beans\\BEANS_MASKED\\IMG_4036.bmp", 1)

print(original)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 500, 500)
cv2.imshow("Original", original)