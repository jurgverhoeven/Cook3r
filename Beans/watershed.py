from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2

image = cv2.imread("C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_masked_warped/Beans2/IMG_5849.jpeg")
shifted = cv2.pyrMeanShiftFiltering(image, 21, 21)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 500, 500)
cv2.imshow("Original", image)

MINIMAL_HSV = np.array([31, 120, 185], np.uint8)
MAXIMAL_HSV = np.array([179, 255, 255], np.uint8)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsvMask = cv2.inRange(hsvImage, MINIMAL_HSV,MAXIMAL_HSV)
maskedMeatballs = cv2.bitwise_and(image, image, mask=hsvMask)   

gray = cv2.cvtColor(maskedMeatballs, cv2.COLOR_BGR2GRAY)

(ret, thresh) = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
#D = ndimage.distance_transform_edt(thresh)
im_floodfill = thresh.copy()
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_out = thresh | im_floodfill_inv
cv2.imshow("thresh",thresh)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(im_out, structure=np.ones((3, 3)))[0]
labels = watershed(im_out, markers, mask=thresh)
cv2.imshow("Dimg",im_out)
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
	
	for i in range(len(contours)):
		cv2.drawContours(mask, contours, i, (i+1), -1)
		tempImage_8u = (mask * 10).astype('uint8')
		# Create a new image containing the contents of a single contour
		cnt = cv2.bitwise_and(image, image, mask=tempImage_8u)
		# Crop the new image to the minimal size
		area = cv2.contourArea(contours[i])
		x,y,w,h = cv2.boundingRect(contours[i])
		cropped_image = cnt[y:y+h, x:x+w]
		im_floodfill = thresh.copy()
		h, w = thresh.shape[:2]
		mask = np.zeros((h+2, w+2), np.uint8)
		# Floodfill from point (0, 0)
		cv2.floodFill(im_floodfill, mask, (0,0), 255)
		im_floodfill_inv = cv2.bitwise_not(im_floodfill)
		im_out = thresh | im_floodfill_inv
		cv2.namedWindow("image"+str(i), cv2.WINDOW_NORMAL)
		cv2.resizeWindow("image"+str(i), 500, 500)
		cv2.imshow("image"+str(i), im_out)
cv2.waitKey(0)