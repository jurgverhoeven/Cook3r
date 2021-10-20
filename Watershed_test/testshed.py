from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils
import cv2
import numpy as np

original  = cv2.imread("IMG_5803.jpeg")
# cv2.imshow("Original", original)

hsvImage = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

MINIMAL_HSV = np.array([0, 60, 0], np.uint8)
MAXIMAL_HSV = np.array([179, 255, 255], np.uint8)

hsvMask = cv2.inRange(hsvImage, (0, 100, 0), (179, 255, 255))
maskedBeans = cv2.bitwise_and(original, original, mask=hsvMask)

# cv2.imshow("Masked beans", maskedBeans)

shifted = cv2.pyrMeanShiftFiltering(maskedBeans, 21, 51)
cv2.imshow("Input", maskedBeans)

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
	labels=thresh)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
labels = ndimage.binary_fill_holes(labels - 1)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

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
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	rect = cv2.minAreaRect(c)
	box = cv2.boxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(maskedBeans,[box],0,(0,0,255),2)

# show the output image
cv2.imshow("Output", maskedBeans)

cv2.waitKey(0)