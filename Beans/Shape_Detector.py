import cv2
import numpy as np
import random as rng
rng.seed(12345)

def nothing(x):
    pass

path = "C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_masked_warped/Beans2" 
#imageName = "PASTA_NO_WATER2.jpg"
imageName = "IMG_5849.jpeg"
# Load image
image = cv2.imread(path+"/"+imageName)
if image is None:
    print('Could not open or find the image')
    exit(0)

height, width, channels = image.shape

heightPart = int(height)
widthPart = int(width)
MINIMAL_HSV = np.array([31, 135, 170], np.uint8)
MAXIMAL_HSV = np.array([90, 255, 255], np.uint8)
# Create a window
cv2.namedWindow('image')
cv2.resizeWindow('image', widthPart, heightPart)
while(1):
    # Show original image
    cv2.imshow('image', image)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Filter out the object by finding the correct color mask
    blur = cv2.GaussianBlur(hsv,(31, 31),cv2.BORDER_DEFAULT)
    mask = cv2.inRange(blur,MINIMAL_HSV, MAXIMAL_HSV) # mask pasta no water
    # mask = cv2.inRange(blur, (20, 85, 120), (179, 255, 255)) # mask beans
    # Find the contours of the object
    
    # Create a new image showing only the filtered out object
    filteredImage = cv2.bitwise_and(image, image, mask=mask)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Filtered Image", filteredImage)

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
    _, bw = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    cv2.imshow('Distance Transform Image', dist)

    _, dist = cv2.threshold(dist, 0, 1.0, cv2.THRESH_BINARY)

    # Dilate a bit the dist image
    #kernel1 = np.ones((5,5), dtype=np.uint8)
    #dist = cv2.erode(dist, kernel1)
    #cv2.GaussianBlur(dist,(11, 11),cv2.BORDER_DEFAULT)
    cv2.imshow('Peaks', dist)
        
    dist_8u = dist.astype('uint8')

    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    tempImage = np.zeros(dist.shape, dtype=np.int32)

    
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
        if cv2.contourArea(contours[i]) > 500 and cv2.contourArea(contours[i]) < 8500: 
            x,y,w,h = cv2.boundingRect(contours[i])
            cropped_image = cnt[y:y+h, x:x+w]
            imagePathName = "C:/Users/Lou/Cook3r/Beans/BEANS_MASKED/" + str(i+53) + ".jpg"
            cv2.imwrite(imagePathName, cropped_image)
        # Uncomment the line below to display a window for each individual cropped image
        # cv2.imshow(windowname, cropped_image)
        # Save the image
        #imagePathName = "C:/Users/Lou/Cook3r/Beans/BEANS_MASKED/" + str(i) + ".jpg"
        #cv2.imwrite(imagePathName, cropped_image)
        # Reset the tempImage after each loop so the new image only contains an individual contour
        tempImage = np.zeros(dist.shape, dtype=np.int32)

    # Draw the background marker
    cv2.circle(markers, (5,5), 3, (255,255,255), -1)
    markers_8u = (markers * 10).astype('uint8')
    cv2.imshow('Markers', markers_8u)
    cv2.watershed(imgResult, markers)
    mark = markers.astype('uint8')
    mark = cv2.bitwise_not(mark)
    kernel1 = np.ones((11,11), dtype=np.uint8)
    mark = cv2.erode(mark, kernel1)
    cv2.imshow('Markers_v2', mark)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
