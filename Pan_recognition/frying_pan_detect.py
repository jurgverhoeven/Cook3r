import cv2
import numpy as np

class Circle:
    def __init__(self, x = 0, y = 0, r = 0) -> None:
        self.x = x
        self.y = y
        self.r = r

path = "C:/Users/Jurg Verhoeven/Documents/Cook3r/Git/Cook3r/Pan_recognition"
imageName = "IMG_3571.jpg"

original = cv2.imread(path+"/"+imageName)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original", 500, 500)
cv2.imshow("Original", original)

cv2.namedWindow("Masked", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Masked", 500, 500)

height, width, channels = original.shape

heightPart = int(height/100*15)
widthPart = int(width/100*15)

crop = original[widthPart:(width-widthPart),heightPart:(height-heightPart)]

gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (23, 23), 0)
blurred = cv2.blur(gray, (13, 13))

circlesInBlurred = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.3, 100)

biggestCircle = Circle()
mask = np.zeros(crop.shape[:2], dtype="uint8")

if circlesInBlurred is not None:
    circlesInBlurred = np.round(circlesInBlurred[0, :].astype("int"))
    for (x1, y1, r1) in circlesInBlurred:
        if r1 > biggestCircle.r:
            biggestCircle = Circle(x = x1, y = y1, r = r1)
cv2.circle(mask, (biggestCircle.x, biggestCircle.y), biggestCircle.r, 255, -1)
masked = cv2.bitwise_and(crop, crop, mask=mask)

cv2.imshow("Masked", masked)


cv2.waitKey(0)