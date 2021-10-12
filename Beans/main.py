import cv2
import numpy as np

original = cv2.imread("C:\\Users\\Lou\\OneDrive - HAN\\Cook3r\\Meatballs\\Meatball.jpeg")

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 500, 500)
cv2.imshow("Original", original)

hsvImage = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

MINIMAL_HSV = np.array([0, 60, 0], np.uint8)
MAXIMAL_HSV = np.array([179, 255, 255], np.uint8)

hsvMask = cv2.inRange(hsvImage, (0, 100 , 0), (179, 255, 255))
maskedMeatballs = cv2.bitwise_and(original, original, mask=hsvMask)

cv2.namedWindow("HSV Masked Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Masked Image", 500, 500)
cv2.imshow("HSV Masked Image", maskedMeatballs)

gray = cv2.cvtColor(maskedMeatballs, cv2.COLOR_BGR2GRAY)

(ret, thresh) = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Thresholded", 500, 500)
cv2.imshow("Thresholded", thresh)

blurred = cv2.GaussianBlur(thresh, (23, 23), 0)

cv2.namedWindow("Blurred", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Blurred", 500, 500)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(101,101))

eroding = cv2.erode(blurred, kernel)

eroding = cv2.dilate(eroding, kernel)

cv2.imshow("Blurred", eroding)

circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=20,maxRadius=5000)

circles = np.uint16(np.around(circles))

cv2.imshow("Circles", circles)

print(len(circles))









cv2.waitKey(0)