import Pan
import cv2
import numpy as np
import os
import glob

data_path = "C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/07102021/Water/"

if __name__ == "__main__":
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.j*'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))
    target = []
    i = 0
    # loop over the image paths
    for filename in file_list:
        image = cv2.imread(filename)
        pan = Pan.Pan(image)
        panImage = pan.getMasked()
        cv2.imwrite("C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/07102021/Water_masked/"+str(i)+".jpeg", panImage)
        i = i+1



