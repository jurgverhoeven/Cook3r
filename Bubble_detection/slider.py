import cv2
import numpy as np

def nothing(x):
    pass

# Load image
image = cv2.imread("C:/Users/Lou/Cook3r/Bubble_detection/masked/18.jpg")
testR = image.copy()
height, width, channels = image.shape

# Create a window
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.namedWindow('sliders',cv2.WINDOW_NORMAL)
cv2.resizeWindow('sliders', 500, 500)

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('Canny 1', 'sliders', 0, 500, nothing)
cv2.createTrackbar('Canny 2', 'sliders', 0, 500, nothing)
cv2.createTrackbar('Blur', 'sliders', 1, 90, nothing)
cv2.createTrackbar('HMin', 'sliders', 0, 179, nothing)
cv2.createTrackbar('SMin', 'sliders', 0, 255, nothing)
cv2.createTrackbar('VMin', 'sliders', 0, 255, nothing)
cv2.createTrackbar('HMax', 'sliders', 0, 179, nothing)
cv2.createTrackbar('SMax', 'sliders', 0, 255, nothing)
cv2.createTrackbar('VMax', 'sliders', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('HMax', 'sliders', 179)
cv2.setTrackbarPos('SMax', 'sliders', 255)
cv2.setTrackbarPos('VMax', 'sliders', 255)


# Initialize HSV min/max values
Canny1Val = Canny2Val = pCanny1Val = pCanny2Val = 1
blurVal = pBlurVal = 1
hMin = sMin = vMin = hMax = sMax = vMax = 1
phMin = psMin = pvMin = phMax = psMax = pvMax = 1
rMin = gMin = bMin = rMax = gMax = bMax = 1
prMin = pgMin = pbMin = prMax = pgMax = pbMax = 1
while(1):
    # Get current positions of all trackbars
    Canny1Val = cv2.getTrackbarPos('Canny 1', 'sliders')
    Canny2Val = cv2.getTrackbarPos('Canny 2', 'sliders')
    blurVal = cv2.getTrackbarPos('Blur', 'sliders')
    hMin = cv2.getTrackbarPos('HMin', 'sliders')
    sMin = cv2.getTrackbarPos('SMin', 'sliders')
    vMin = cv2.getTrackbarPos('VMin', 'sliders')
    hMax = cv2.getTrackbarPos('HMax', 'sliders')
    sMax = cv2.getTrackbarPos('SMax', 'sliders')
    vMax = cv2.getTrackbarPos('VMax', 'sliders')
    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    rlower = np.array([rMin, gMin, bMin])
    rupper = np.array([rMax, gMax, bMax])
    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image2 = image.copy()
    if(blurVal % 2 ==0):
        blurVal = (blurVal + 1)
        cv2.setTrackbarPos('Blur', 'sliders', blurVal)
    blur = cv2.GaussianBlur(hsv,(blurVal, blurVal),cv2.BORDER_DEFAULT)
    mask = cv2.inRange(blur, lower, upper)
    im_floodfill = mask.copy()
    h, w = mask.shape[:2]
    test = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, test, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = mask | im_floodfill_inv
    circles = cv2.HoughCircles(im_out, cv2.HOUGH_GRADIENT, 1, 20, param1 = 25, param2 = 10, minRadius = 20, maxRadius = 50)

    # Draw circles on the original image.
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle( image2, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # Find circles by the Hough transfermation.

        #cv2.imshow("Marked", image)
    # Display result image
    
    result = cv2.bitwise_and(image, image, mask=im_out)
    cv2.imshow('image', result)
    cv2.imshow('flood', im_out)
    cv2.imshow('image2', image2)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()