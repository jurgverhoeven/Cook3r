import cv2
import numpy as np

class Circle:
    def __init__(self, x = 0, y = 0, r = 0) -> None:
        self.x = x
        self.y = y
        self.r = r

class Pan:
    def __init__(self, image) -> None:
        self.image = image

    def getMasked(self):
        cv2.namedWindow("Original pan image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original pan image", 500, 500)
        cv2.imshow("Original pan image", self.image)

        height, width, channels = self.image.shape
        heightPart = int(height/100*1)
        widthPart = int(width/100*1)
        crop = self.image[heightPart:(height-heightPart), widthPart:(width-widthPart)]

        cv2.namedWindow("Cropped pan image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Cropped pan image", 500, 500)
        cv2.imshow("Cropped pan image", crop)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), cv2.BORDER_DEFAULT)
        # blurred = cv2.blur(gray, (15, 15))

        cv2.namedWindow("Blurred pan image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Blurred pan image", 500, 500)
        cv2.imshow("Blurred pan image", blurred)

        # circlesInBlurred = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.3, 100, minRadius=500, maxRadius=750)
        circlesInBlurred = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.3, 100, minRadius=500, maxRadius=750)

        # circlesInBlurred = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.3, 100)
        print(len(circlesInBlurred))

        biggestCircle = Circle()
        mask = np.zeros(crop.shape[:2], dtype="uint8")

        if circlesInBlurred is not None:
            circlesInBlurred = np.round(circlesInBlurred[0, :].astype("int"))
            for (x1, y1, r1) in circlesInBlurred:
                if r1 > biggestCircle.r:
                    biggestCircle = Circle(x = x1, y = y1, r = r1)
        cv2.circle(mask, (biggestCircle.x, biggestCircle.y), biggestCircle.r, 255, -1)
        masked = cv2.bitwise_and(crop, crop, mask=mask)
        cv2.imshow("Original", masked)
        print("Biggest circle diameter: ")
        print(biggestCircle.r)

        newImage = np.zeros((1000,1000,3), np.uint8)

        pt_A = [biggestCircle.x-biggestCircle.r, biggestCircle.y-biggestCircle.r]
        pt_B = [biggestCircle.x-biggestCircle.r, biggestCircle.y+biggestCircle.r]
        pt_C = [biggestCircle.x+biggestCircle.r, biggestCircle.y+biggestCircle.r]
        pt_D = [biggestCircle.x+biggestCircle.r, biggestCircle.y-biggestCircle.r]

        # print(pt_A)
        # print(pt_B)
        # print(pt_C)
        # print(pt_D)


        width_AD = 1000
        width_BC = 1000
        maxWidth = max(int(width_AD), int(width_BC))
        height_AB = 1000
        height_CD = 1000
        maxHeight = max(int(height_AB), int(height_CD))

        input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])

        # Compute the perspective transform M
        M = cv2.getPerspectiveTransform(input_pts,output_pts)

        out = cv2.warpPerspective(masked,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)

        cv2.imshow("Warped", out)
        return out
