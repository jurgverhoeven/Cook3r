import cv2
import numpy as np
import os

piss = 0
path = "C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_masked_warped/Beans" 
MINIMAL_HSV = np.array([29, 90, 198], np.uint8)
MAXIMAL_HSV = np.array([55, 250, 255], np.uint8)

def findBeans(image):
    global piss
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Filter out the object by finding the correct color mask
    blur = cv2.GaussianBlur(hsv,(25, 25),cv2.BORDER_DEFAULT)
    mask = cv2.inRange(blur,MINIMAL_HSV, MAXIMAL_HSV) # mask pasta no water
    im_floodfill = mask.copy()
    h, w = mask.shape[:2]
    test = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, test, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = mask | im_floodfill_inv
    
    # Create a new image showing only the filtered out object
    filteredImage = cv2.bitwise_and(image, image, mask=im_out)
   
    kernel = np.array([[1, 1, 1], [1, -13, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(filteredImage, cv2.CV_32F, kernel)
    sharp = np.float32(filteredImage)
    imgResult = sharp - imgLaplacian

    # Convert the image back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    
    _, dist = cv2.threshold(dist, 0, 1.0, cv2.THRESH_BINARY)
    
        
    dist_8u = dist.astype('uint8')
    
    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    tempImage = np.zeros(dist.shape, dtype=np.int32)
    beansList = []
    for i in range(len(contours)):
        # Draw the foreground markers
        cv2.drawContours(markers, contours, i, (i+1), -1)
        
        # Draw the individual markers on seperate images
        cv2.drawContours(tempImage, contours, i, (i+1), -1)
        windowname = "Contour" + str(i)
        tempImage_8u = (tempImage * 10).astype('uint8')
        # Create a new image containing the contents of a single contour
        cnt = cv2.bitwise_and(image, image, mask=tempImage_8u)
        # Crop the new image to the minimal size
        if cv2.contourArea(contours[i]) > 200 and cv2.contourArea(contours[i]) < 12000: 
            x,y,w,h = cv2.boundingRect(contours[i])
            cropped_image = cnt[y:y+h, x:x+w]
            piss = piss + 1
            cv2.imwrite("C:/Users/Lou/Cook3r/Beans/BEANS_MASKED/"+str(piss)+".jpg", cropped_image)
            
            
        tempImage = np.zeros(dist.shape, dtype=np.int32)
    return beansList

for root, dirs, files in os.walk(path):
    beans = []
    for filename in files:
        image = cv2.imread(path+"/"+filename)
        findBeans(image)
    print("done") 

cv2.waitKey(0)