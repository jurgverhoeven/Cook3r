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
    def __init__(self, minH, minS, minV, maxH, maxS, maxV, blurVal, targetFood):
        self.minH = minH
        self.minS = minS
        self.minV = minV
        self.maxH = maxH
        self.maxS = maxS
        self.maxV = maxV
        self.blurVal = blurVal
        self.targetFood = targetFood


    def maskImage(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv,(self.blurVal, self.blurVal),cv2.BORDER_DEFAULT)
        mask = cv2.inRange(blur, (self.minH, self.minS, self.minV), (self.maxH, self.maxS, self.maxV)) 
        filteredImage = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("Mask", filteredImage)
        return filteredImage

    def findFood(self, image, minSize, maxSize):
        kernel = np.array([[1, 1, 1], [1, -13, 1], [1, 1, 1]], dtype=np.float32)
        # Use laplacian filtering and convert it to CV_32F
        # This is because something deeper than CV_8U is needed because of negative values in the kernel
        imgLaplacian = cv2.filter2D(self.getMaskedImage(image), cv2.CV_32F, kernel)
        sharp = np.float32(self.getMaskedImage(image))
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
            cnt = cv2.bitwise_and(image, image, mask=tempImage_8u)
            # Crop the new image to the minimal size
            if cv2.contourArea(contours[i]) > minSize and cv2.contourArea(contours[i]) < maxSize: 
                # print(cv2.contourArea(contours[i]))
                food = self.createCorrectFood(contours, cnt, i)
                foodList.append(food)
            tempImage = np.zeros(dist.shape, dtype=np.int32)
        markers_8u = (markers * 10).astype('uint8')

        # cv2.namedWindow('Markers', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Markers', 500, 500)
        # cv2.imshow('Markers', markers_8u)
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


    def getMaskedImage(self, image):
        return self.maskImage(image)

    def getTargetFood(self):
        return self.targetFood

class Meatball_detect(Detector):
    def __init__(self):
        super(Meatball_detect, self).__init__(minH=0, minS=60, minV=0, maxH=179, maxS=255, maxV=255, blurVal=5, targetFood=TargetFood.Meatball)

    def findFood(self, image):
        maskedMeatballs = self.getMaskedImage(image)
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
    def __init__(self):
        super(Pasta_detect, self).__init__(minH=20, minS=105, minV=145, maxH=30, maxS=210, maxV=245, blurVal=25, targetFood=TargetFood.Pasta)

class Bean_detect(Detector):
    def __init__(self):
        super(Bean_detect, self).__init__(minH=29, minS=90, minV=198, maxH=55, maxS=250, maxV=255, blurVal=25, targetFood=TargetFood.Bean)
        # super(Bean_detect, self).__init__(minH=15, minS=15, minV=120, maxH=45, maxS=150, maxV=255, blurVal=25, targetFood=TargetFood.Bean)

class Carrot_detect(Detector):
    def __init__(self):
        super(Carrot_detect, self).__init__(minH=5, minS=90, minV=120, maxH=20, maxS=220, maxV=255, blurVal=3, targetFood=TargetFood.Carrot)

class Fishstick_detect(Detector):
    def __init__(self):
        super(Fishstick_detect, self).__init__(minH=10, minS=150, minV=65, maxH=20, maxS=250, maxV=250, blurVal=3, targetFood=TargetFood.Fishstick)

class Potato_detect(Detector):
    def __init__(self):
        super(Potato_detect, self).__init__(minH=20, minS=55, minV=145, maxH=30, maxS=165, maxV=250, blurVal=15, targetFood=TargetFood.Potato)
