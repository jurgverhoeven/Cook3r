import cv2
from enum import Enum
import numpy as np
import random as rng
import colorsys
rng.seed(12345)

class Shape(Enum):
    Circle = 0
    Rectangle = 1

class Food:
    def __init__(self, image, shape):
        self.image = image
        self.shape = shape
        self.area, self.perimeter = self.determineArea(image)
        self.prominentColor = self.determineProminentColor(image)

    def getImage(self):
        return self.image

    def getShape(self):
        if self.shape == Shape.Circle:
            return 0
        else:
            return 1

    def getArea(self):
        return self.area

    def getPerimeter(self):
        return self.perimeter

    def getProminentColorBlue(self):
        return self.prominentColor[0]
    def getProminentColorGreen(self):
        return self.prominentColor[1]
    def getProminentColorRed(self):
        return self.prominentColor[2]

    def getProminentHue(self):
        return int(colorsys.rgb_to_hsv(1/self.getProminentColorRed(), 1/self.getProminentColorGreen(), 1/self.getProminentColorBlue())[0]*255)
        

    def determineArea(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (ret, binaryImage) = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours
        drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        highest = 0
        countervalue = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > highest:
                highest = area
                countervalue = i
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours, countervalue, color, 2, cv2.LINE_8, hierarchy, 0)
        # Show in a window
        cv2.imshow('Contours', drawing)

        return int(highest), int(cv2.arcLength(contours[countervalue],True))

    def determineProminentColor(self, img):
        Z = img.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 1
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2[0][0]

class Meatball(Food):
    def __init__(self, image):
        super(Meatball, self).__init__(image, Shape.Circle)

class Pasta(Food):
    def __init__(self, image):
        super(Pasta, self).__init__(image, Shape.Rectangle)

class Bean(Food):
    def __init__(self, image):
        super(Bean, self).__init__(image, Shape.Rectangle)