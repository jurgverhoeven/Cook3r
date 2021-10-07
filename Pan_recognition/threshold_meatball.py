import cv2
import numpy as np

path = "C:/Users/Jurg Verhoeven/Documents/Cook3r/Git/Cook3r/Pan_recognition/balls"
imageName = "IMG_3580.jpg"

original = cv2.imread(path+"/"+imageName)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 500, 500)
cv2.imshow("Original", original)

# 52 183 67
# 70 230 110

ORANGE_MAX = np.array([0, 60, 0],np.uint8)
ORANGE_MIN = np.array([179, 255, 255],np.uint8)

hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 100 , 0), (179, 255, 255))
result = cv2.bitwise_and(original, original, mask=mask)

cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test", 500, 500)
cv2.imshow("test", result)

circled = result.copy()

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

blurred = cv2.blur(gray, (5, 5))

cv2.namedWindow("threshold", cv2.WINDOW_NORMAL)
cv2.resizeWindow("threshold", 500, 500)

(ret, thresh) = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

kernel = np.ones((19,19), np.uint8)

eroded = cv2.erode(thresh, kernel, iterations=1)
eroded = cv2.dilate(eroded, kernel, iterations=1)

cv2.imshow("threshold", eroded)

contours, hierarchy  = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img = cv2.drawContours(circled, contours, -1, (0,255,75), 2)

print(len(contours))

area = 0

for meatballContour in contours:
    meatballArea = cv2.contourArea(meatballContour)
    print(meatballArea)
    area += meatballArea

print(area)

cv2.namedWindow("Circles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Circles", 500, 500)
cv2.imshow("Circles", circled)

cv2.waitKey(0)