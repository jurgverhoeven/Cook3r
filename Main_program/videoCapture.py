import cv2
class videoCapture:
    def __init__(self,video):
        self.vidcap = cv2.VideoCapture(video)
    
    def getFrame(self):
        self.success,self.image = self.vidcap.read()
        return self.image
        
    def isSucces(self):
        return self.success


# if __name__ == "__main__":
#     cap = videoCapture("C:/Users/Lou-J/Cook3r/Main_program/output.avi")
#     frame = cap.getFrame()
#     while(cap.isSucces):
#         cv2.imshow("Frame",frame)
#         cv2.waitKey(1)
#         frame = cap.getFrame()