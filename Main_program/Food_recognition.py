import Food
import Meatball_detect
import Pasta_detect
import Bean_detect
import cv2


class Food_recognition:
    def __init__(self) -> None:
        self.meatballDetector = Meatball_detect.Meatball_detect()
        self.pastaDetector = Pasta_detect.Pasta_detect()
        self.beanDetector = Bean_detect.Bean_detect()

    def recognize(self, panImage):
        meatballs = self.meatballDetector.getMeatballs(panImage)
        pasta = self.pastaDetector.getPasta(panImage)
        beans = self.beanDetector.getBeans(panImage)

        print("Amount of meatballs: "+str(len(meatballs)))
        print("Amount of Pasta: "+str(len(pasta)))
        print("Amount of Beans: "+str(len(beans)))

        if len(meatballs) > len(pasta) and len(meatballs) > len(beans):
            return meatballs
        elif len(pasta) > len(meatballs) and len(pasta) > len(beans):
            return pasta
        elif len(beans) > len(meatballs) and len(beans) > len(pasta):
            return beans
        else:
            return 0
