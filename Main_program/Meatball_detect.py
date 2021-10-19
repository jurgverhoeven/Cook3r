import cv2
import numpy as np
import Food

class Meatball_detect:
    def getMeatballs(self, image):
        return self.__findMeatballs(image)

    def __meatballCrop(self, image, x, y, r):
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

    def __findMeatballs(self, image):
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original", 500, 500)
        # cv2.imshow("Original", image)

        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        MINIMAL_HSV = np.array([0, 60, 0], np.uint8)
        MAXIMAL_HSV = np.array([179, 255, 255], np.uint8)

        hsvMask = cv2.inRange(hsvImage, (0, 100, 0), (179, 255, 255))
        maskedMeatballs = cv2.bitwise_and(image, image, mask=hsvMask)

        cv2.namedWindow("HSV Masked Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HSV Masked Image", 500, 500)
        # cv2.imshow("HSV Masked Image", maskedMeatballs)

        gray = cv2.cvtColor(maskedMeatballs, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        cv2.namedWindow("Blurred", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Blurred", 500, 500)
        # cv2.imshow("Blurred", blurred)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100,
                                param1=100, param2=26, minRadius=40, maxRadius=75)
        amount = 0

        meatballList = []

        if circles is not None:
            amount = len(circles[0])
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                meatballImage, x, y, r = self.__meatballCrop(maskedMeatballs, i[0], i[1], i[2])
                meatball = Food.Meatball(image=meatballImage, x=x, y=y, width=r, height=r)

                meatballList.append(meatball)
        return meatballList