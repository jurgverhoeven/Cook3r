import Food
import Meatball_detect
import cv2

class Food_recognition:
    def __init__(self) -> None:
        self.meatballDetector = Meatball_detect.Meatball_detect()

    def recognize(self, panImage):
        meatballs = self.meatballDetector.getMeatballs(panImage)

        print("Amount of meatballs: "+str(len(meatballs)))

        for meatball in meatballs:
            print(meatball.x)
            print(meatball.y)
            print(meatball.radius)
            cv2.circle(panImage, (meatball.x, meatball.y), meatball.radius, (0, 255, 0), 5)

        cv2.imshow("Circles", panImage)