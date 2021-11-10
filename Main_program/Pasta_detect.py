import cv2
import numpy as np
import os
import random
import Food

class Pasta_detect:
    def getPasta(self, image):
        return self.__findPasta(image)

    def __findPasta(self, image):

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Filter out the object by finding the correct color mask
        blur = cv2.GaussianBlur(hsv,(25, 25),cv2.BORDER_DEFAULT)
        mask = cv2.inRange(blur, (20, 105, 145), (30, 210, 245)) 
        # Find the contours of the object
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a new image showing only the filtered out object
        filteredImage = cv2.bitwise_and(image, image, mask=mask)
        cv2.namedWindow("Filtered Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Filtered Image", 500, 500)
        # cv2.imshow("Filtered Image", filteredImage)

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        # Use laplacian filtering and convert it to CV_32F
        # This is because something deeper than CV_8U is needed because of negative values in the kernel
        imgLaplacian = cv2.filter2D(filteredImage, cv2.CV_32F, kernel)
        sharp = np.float32(filteredImage)
        imgResult = sharp - imgLaplacian

        # Convert the image back to 8bits gray scale
        imgResult = np.clip(imgResult, 0, 255)
        imgResult = imgResult.astype('uint8')
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)

        bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

        cv2.namedWindow('Distance Transform Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Distance Transform Image', 500, 500)
        # cv2.imshow('Distance Transform Image', dist)

        _, dist = cv2.threshold(dist, 0.4, 1.0, cv2.THRESH_BINARY)

        # Dilate a bit the dist image
        kernel1 = np.ones((12, 12), dtype=np.uint8)
        kernelErode = np.ones((3,3), dtype=np.uint8)
        dist = cv2.dilate(dist, kernel1)
        dist = cv2.erode(dist,kernelErode)
        cv2.namedWindow('Peaks', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Peaks', 500, 500)
        # cv2.imshow('Peaks', dist)
            
        dist_8u = dist.astype('uint8')

        # Find total markers
        contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create the marker image for the watershed algorithm
        markers = np.zeros(dist.shape, dtype=np.int32)
        tempImage = np.zeros(dist.shape, dtype=np.int32)

        pastaList = []
        for i in range(len(contours)):
            # Draw the foreground markers
            cv2.drawContours(markers, contours, i, (i+1), -1)
            
            # Draw the individual markers on seperate images
            cv2.drawContours(tempImage, contours, i, (i+1), -1)
            windowname = "Contour" + str(i)
            tempImage_8u = (tempImage * 10).astype('uint8')
            # Create a new image containing the contents of a single contour
            cnt = cv2.bitwise_and(image, image, mask=tempImage_8u)
            area = cv2.contourArea(contours[i])
            # Crop the new image to the minimal size
            x,y,w,h = cv2.boundingRect(contours[i])
            cropped_image = cnt[y:y+h, x:x+w]
            # Uncomment the line below to display a window for each individual cropped image
            # # cv2.imshow(windowname, cropped_image)
            # Save the image and filter out the outliers that are too small
            if(area > 300 and area < 3000):
                pastaPiece = Food.Pasta(image=cropped_image, x=x, y=y, width=w, height=h)
                pastaList.append(pastaPiece)
            # Reset the tempImage after each loop so the new image only contains an individual contour
            tempImage = np.zeros(dist.shape, dtype=np.int32)

        # Draw the background marker
        cv2.circle(markers, (5,5), 3, (255,255,255), -1)
        markers_8u = (markers * 10).astype('uint8')

        cv2.namedWindow('Markers', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Markers', 500, 500)
        # cv2.imshow('Markers', markers_8u)

        cv2.watershed(imgResult, markers)
        mark = markers.astype('uint8')
        mark = cv2.bitwise_not(mark)

        cv2.namedWindow('Markers_v2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Markers_v2', 500, 500)
        # cv2.imshow('Markers_v2', mark)

        return pastaList