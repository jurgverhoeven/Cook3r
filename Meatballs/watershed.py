from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2

image = cv2.imread("C:\\Users\\Lou\\OneDrive - HAN\\Cook3r\\Meatballs\\Meatball.jpeg")
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 500, 500)
cv2.imshow("Original", image)

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

MINIMAL_HSV = np.array([0, 60, 0], np.uint8)
MAXIMAL_HSV = np.array([179, 255, 255], np.uint8)

hsvMask = cv2.inRange(hsvImage, (0, 100 , 0), (179, 255, 255))
maskedMeatballs = cv2.bitwise_and(image, image, mask=hsvMask)

cv2.namedWindow("HSV Masked Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Masked Image", 500, 500)
cv2.imshow("HSV Masked Image", maskedMeatballs)

gray = cv2.cvtColor(maskedMeatballs, cv2.COLOR_BGR2GRAY)

(ret, thresh) = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thresholded", 500, 500)
cv2.imshow("Thresholded", thresh)

cv2.namedWindow("Blurred", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blurred", 500, 500)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=50,
	labels=thresh)
# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
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
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
	cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# show the output image
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Output", 500, 500)
cv2.imshow("Output", image)
cv2.waitKey(0)