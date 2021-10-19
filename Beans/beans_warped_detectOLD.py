import cv2
from skimage.feature import peak_local_max
from skimage.segmentation._watershed import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import os
import random

path = "C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_masked_warped/Beans2" 
MINIMAL_HSV = np.array([31, 110, 165], np.uint8)
MAXIMAL_HSV = np.array([140, 255, 255], np.uint8)
kernelLap = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

def findMeatballs(image):
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsvBlur = cv2.GaussianBlur(hsvImage,(25, 25),cv2.BORDER_DEFAULT)
    hsvMask = cv2.inRange(hsvBlur, MINIMAL_HSV,MAXIMAL_HSV)
    maskedMeatballs = cv2.bitwise_and(image, image, mask=hsvMask)   

    gray = cv2.cvtColor(maskedMeatballs, cv2.COLOR_BGR2GRAY)

    (ret, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map

    D = ndimage.distance_transform_edt(thresh)
    
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(D, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)

# loop over the unique labels returned by the Watershed
# algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        # detect contours in the mask and grab the largest one
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create the marker image for the watershed algorithm
        beansList = []
        
        for i in range(len(contours)):
            cv2.drawContours(mask, contours, i, (i+1), -1)
            tempImage_8u = (mask * 10).astype('uint8')
            # Create a new image containing the contents of a single contour
            cnt = cv2.bitwise_and(image, image, mask=tempImage_8u)
            # Crop the new image to the minimal size
            area = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            cropped_image = cnt[y:y+h, x:x+w]
            beansList.append(cropped_image)
        return beansList

for root, dirs, files in os.walk(path):
    beans = []
    for filename in files:
        image = cv2.imread(path+"/"+filename)
        # meatballs = findMeatballs(image)
        beans.extend(findMeatballs(image))

    for bean in range(len(beans)):
        cv2.imwrite("C:/Users/Lou/Cook3r/Beans/BEANS_MASKED/"+str(bean)+".jpg", beans[bean])

cv2.waitKey(0)