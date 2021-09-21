import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

mypath = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Simple_dataset/Pasta"
topath = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Masked/Pasta"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(onlyfiles)

class Circle:
    def __init__(self, x = 0, y = 0, r = 0) -> None:
        self.x = x
        self.y = y
        self.r = r


for file in onlyfiles:
    readImage = cv2.imread(mypath+"/"+file)
    gray = cv2.cvtColor(readImage, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), cv2.BORDER_DEFAULT)
    circlesInBlurred = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.3, 100)

    biggestCircle = Circle()
    mask = np.zeros(readImage.shape[:2], dtype="uint8")

    if circlesInBlurred is not None:
        circlesInBlurred = np.round(circlesInBlurred[0, :].astype("int"))
        for (x1, y1, r1) in circlesInBlurred:
            if r1 > biggestCircle.r:
                biggestCircle = Circle(x = x1, y = y1, r = r1)
    cv2.circle(mask, (biggestCircle.x, biggestCircle.y), biggestCircle.r, 255, -1)
    masked = cv2.bitwise_and(readImage, readImage, mask=mask)
    print("processed")
    cv2.imwrite(topath+"/"+file, masked)
