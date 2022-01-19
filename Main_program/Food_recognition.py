import Food
import Detector
import cv2


class Food_recognition:
    def __init__(self) -> None:
        self.meatballDetector = Detector.Meatball_detect()
        self.pastaDetector = Detector.Pasta_detect()
        self.beanDetector = Detector.Bean_detect()
        self.carrotDetector = Detector.Carrot_detect()
        self.fishStickDetector = Detector.Fishstick_detect()
        self.potatoDetector = Detector.Potato_detect()

    def recognize(self, panImage):
        meatballs = self.meatballDetector.findFood(panImage)
        pasta = self.pastaDetector.findFood(panImage, minSize=200, maxSize=20000)
        beans = self.beanDetector.findFood(panImage, minSize=200, maxSize=20000)
        carrots = self.carrotDetector.findFood(panImage, minSize=200, maxSize=20000)
        fish_sticks = self.fishStickDetector.findFood(panImage, minSize=200, maxSize=20000)
        potatoes = self.potatoDetector.findFood(panImage, minSize=200, maxSize=20000)

        # print("Amount of meatballs: "+str(len(meatballs)))
        # print("Amount of Pasta: "+str(len(pasta)))
        # print("Amount of Beans: "+str(len(beans)))

        foods = []
        foods.extend(meatballs)
        foods.extend(pasta)
        foods.extend(beans)
        foods.extend(carrots)
        foods.extend(fish_sticks)
        foods.extend(potatoes)

        return foods

        # if len(meatballs) > len(pasta) and len(meatballs) > len(beans):
        #     return meatballs
        # elif len(pasta) > len(meatballs) and len(pasta) > len(beans):
        #     return pasta
        # elif len(beans) > len(meatballs) and len(beans) > len(pasta):
        #     return beans
        # else:
        #     return 0
