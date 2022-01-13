import cv2
import numpy as np

class CookerDetect:
    def __init__(self, image) -> None:
        maxWidth = 1500
        maxHeight = 1500
        input_pts = np.float32([[59,222], [0,1540], [1500,1540], [1409,250]])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])
        M = cv2.getPerspectiveTransform(input_pts,output_pts)
        self.image = cv2.warpPerspective(image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
        
        
    def pit(self, pointA, pointB, pointC, pointD):
        maxWidth = 1500
        maxHeight = 1500

        input_pts = np.float32([pointA, pointB, pointC, pointD])
        output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])
        M = cv2.getPerspectiveTransform(input_pts,output_pts)

        out = cv2.warpPerspective(self.image,M,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
        return out
    def getMasked(self):
        
        cv2.namedWindow("Original pan image", cv2.WINDOW_NORMAL)
        cv2.imshow("Original pan image", self.image)
        pit1 = self.pit([0,0],[0,750],[750,750],[750,0])
        pit2 = self.pit([750,0],[750,750],[1500,750],[1500,0])
        pit3 = self.pit([375,750],[375,1500],[1125,1500],[1125,750])
    
        return [    pit1,pit2,pit3]

if __name__ == "__main__":
    filename = "C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/cooker.jpeg"
    image = cv2.imread(filename)
    cooker = CookerDetect(image)
    pits = cooker.getMasked()
    for pit in range(len(pits)):
        cv2.imwrite("C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/"+"pit"+str(pit)+".jpg",pits[pit])

    cv2.waitKey(0)