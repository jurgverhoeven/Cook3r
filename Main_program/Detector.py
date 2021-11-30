import cv2
import numpy as np
import Food
from enum import Enum

class TargetFood(Enum):
    Meatball = 0
    Pasta = 1
    Bean = 2
    Fishstick = 3
    Potato = 4
    Carrot = 5

class Detector:
    def __init__(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal, targetFood):
        self.image = image
        self.maskedImage = self.maskImage(image, minH, minS, minV, maxH, maxS, maxV, blurVal)
        self.targetFood = targetFood

    def maskImage(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv,(blurVal, blurVal),cv2.BORDER_DEFAULT)
        mask = cv2.inRange(blur, (minH, minS, minV), (maxH, maxS, maxV)) 
        filteredImage = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("Mask", filteredImage)
        return filteredImage

    def findFood(self, minSize, maxSize):
        kernel = np.array([[1, 1, 1], [1, -13, 1], [1, 1, 1]], dtype=np.float32)
        # Use laplacian filtering and convert it to CV_32F
        # This is because something deeper than CV_8U is needed because of negative values in the kernel
        imgLaplacian = cv2.filter2D(self.getMaskedImage(), cv2.CV_32F, kernel)
        sharp = np.float32(self.getMaskedImage())
        imgResult = sharp - imgLaplacian

        # Convert the image back to 8bits gray scale
        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype('uint8')
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)

        bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 20, 250, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY) # Change min value to optimize

        dist_8u = dist.astype('uint8')

        # Find total markers
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create the marker image for the watershed algorithm
        markers = np.zeros(dist.shape, dtype=np.int32)
        tempImage = np.zeros(dist.shape, dtype=np.int32)

        foodList = []
        for i in range(len(contours)):
            # Draw the foreground markers
            cv2.drawContours(markers, contours, i, (i+1), -1)  
            # Draw the individual markers on seperate images
            cv2.drawContours(tempImage, contours, i, (i+1), -1)
            tempImage_8u = (tempImage * 10).astype('uint8')
            # Create a new image containing the contents of a single contour
            cnt = cv2.bitwise_and(self.getImage(), self.getImage(), mask=tempImage_8u)
            # Crop the new image to the minimal size
            if cv2.contourArea(contours[i]) > minSize and cv2.contourArea(contours[i]) < maxSize: 
                print(cv2.contourArea(contours[i]))
                food = self.createCorrectFood(contours, cnt, i)
                foodList.append(food)
            tempImage = np.zeros(dist.shape, dtype=np.int32)
        markers_8u = (markers * 10).astype('uint8')

        cv2.namedWindow('Markers', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Markers', 500, 500)
        cv2.imshow('Markers', markers_8u)
        return foodList
    
    def createCorrectFood(self, contours, contour, contourIndex):
        x,y,w,h = cv2.boundingRect(contours[contourIndex])
        cropped_image = contour[y:y+h, x:x+w]
        if(self.getTargetFood() == TargetFood.Meatball): 
            return Food.Meatball(image=cropped_image, x=x, y=y, width=w, height=h)
        elif(self.getTargetFood() == TargetFood.Pasta):
            return Food.Pasta(image=cropped_image, x=x, y=y, width=w, height=h)
        elif(self.getTargetFood() == TargetFood.Bean):
            return Food.Bean(image=cropped_image, x=x, y=y, width=w, height=h)
        elif(self.getTargetFood() == TargetFood.Fishstick):
            return Food.Fishstick(image=cropped_image, x=x, y=y, width=w, height=h)
        elif(self.getTargetFood() == TargetFood.Potato):
            return Food.Potato(image=cropped_image, x=x, y=y, width=w, height=h)
        elif(self.getTargetFood() == TargetFood.Carrot):
            return Food.Carrot(image=cropped_image, x=x, y=y, width=w, height=h)

    def getImage(self):
        return self.image

    def getMaskedImage(self):
        return self.maskedImage

    def getTargetFood(self):
        return self.targetFood

class Meatball_detect(Detector):
    def __init__(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal):
        super(Meatball_detect, self).__init__(image, minH, minS, minV, maxH, maxS, maxV, blurVal, TargetFood.Meatball)

    def findFood(self):
        maskedMeatballs = self.getMaskedImage()
        gray = cv2.cvtColor(maskedMeatballs, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=26, minRadius=40, maxRadius=75)
        meatballList = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            count = 0
            for i in circles[0, :]:
                meatballImage, x, y, r = self.meatballCrop(maskedMeatballs, i[0], i[1], i[2])
                meatball = Food.Meatball(image=meatballImage, x=x, y=y, width=r, height=r)

                meatballList.append(meatball)
                count += 1
        return meatballList

    def meatballCrop(self, image, x, y, r):
        pt_A = [x - r, y - r]
        pt_B = [x - r, y + r]
        pt_C = [x + r, y + r]
        pt_D = [x + r, y - r]

        width_AD = r*2
        width_BC = r*2
        maxWidth = max(int(width_AD), int(width_BC))
        height_AB = r*2
        height_CD = r*2
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])

        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts, output_pts)

        out = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
        return out, x, y, r

class Pasta_detect(Detector):
    def __init__(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal):
        super(Pasta_detect, self).__init__(image, minH, minS, minV, maxH, maxS, maxV, blurVal, TargetFood.Pasta)

class Bean_detect(Detector):
    def __init__(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal):
        super(Bean_detect, self).__init__(image, minH, minS, minV, maxH, maxS, maxV, blurVal, TargetFood.Bean)

class Carrot_detect(Detector):
    def __init__(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal):
        super(Carrot_detect, self).__init__(image, minH, minS, minV, maxH, maxS, maxV, blurVal, TargetFood.Carrot)

class Fishstick_detect(Detector):
    def __init__(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal):
        super(Fishstick_detect, self).__init__(image, minH, minS, minV, maxH, maxS, maxV, blurVal, TargetFood.Fishstick)

class Potato_detect(Detector):
    def __init__(self, image, minH, minS, minV, maxH, maxS, maxV, blurVal):
        super(Potato_detect, self).__init__(image, minH, minS, minV, maxH, maxS, maxV, blurVal, TargetFood.Potato)
