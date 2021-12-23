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
        blurred = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
        # blurred = cv2.blur(gray, (15, 15))

        cv2.namedWindow("Blurred pan image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Blurred pan image", 500, 500)
        cv2.imshow("Blurred pan image", blurred)

        # circlesInBlurred = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.3, 100, minRadius=500, maxRadius=750)
        circlesInBlurred = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 100, minRadius=100, maxRadius=200)

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


if __name__ == "__main__":

    import os
    import glob

    path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Gezamenlijk/Test fotos/Joost en Thijs/wortelwebcam/wortel (1).jpg"
    
    image = cv2.imread(path)
    pan = Pan(image)
    pan_image = pan.getMasked()
    cv2.imshow("Masked pan image", pan_image)
    cv2.imwrite("plaatje voor Lou.jpg", pan_image)
    
    # path = "C:/Users/Jurg Verhoeven/OneDrive - HAN/Bureaublad/PanDetect"

    # saveloc = "C:/Users/Jurg Verhoeven/Documents/temp"
    
    # print("[INFO] loading images...")
    # p = os.path.sep.join([path, '**', '*.j*'])

    # file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    # print("[INFO] images found: {}".format(len(file_list)))

    # # loop over the image paths
    # for filename in file_list:
    #     label = filename.split(os.path.sep)[-2]
    #     food_image = cv2.imread(filename)
    #     print((os.path.basename(filename)))
    #     pan = Pan(food_image)
    #     pan_image = pan.getMasked()
    #     cv2.imwrite(saveloc+"/"+label+"/"+(os.path.basename(filename))+".jpg", pan_image)
    cv2.waitKey(0)
