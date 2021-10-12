import cv2
import numpy as np

image = cv2.imread("IMG_3318.jpg")


output = image.copy()

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 100, minRadius=500, maxRadius=3000)

if circles is not None:
    circles = np.round(circles[0, :].astype("int"))
    print(circles)
    for (x, y, r) in circles:
        print(x)
        print(y)
        print(r)
        cv2.circle(output, (x, y), r, (0, 0, 0), 10)
else:
    print("No circle found")
cv2.namedWindow("circle", cv2.WINDOW_NORMAL)
cv2.resizeWindow("circle", 500, 500)
cv2.imshow("circle", output)
cv2.waitKey(0)
