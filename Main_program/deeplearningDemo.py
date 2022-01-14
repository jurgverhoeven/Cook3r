from ctypes import resize
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img
# Helper libraries
import numpy as np
import cv2
from videoCapture import videoCapture
import Pan
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
waterPath = 'C:/Users/Lou-J/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_v2_masked_warped_v2/Black_pans_v2_masked_warped'

training = tf.keras.utils.image_dataset_from_directory(
    waterPath,image_size=(1,
    1),
    seed=123
)
class_names = training.class_names 
num_rows = 1
num_cols = 1
num_images = num_rows*num_cols
def plot_image(predictions_array, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img/255.0, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(predictions_array, true_label):
  plt.grid(False)
  plt.xticks(range(9))
  plt.yticks([])
  thisplot = plt.bar(range(9), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

if __name__ == "__main__":
  cap = videoCapture("C:\\Users\\Lou-J\\OneDrive - HAN\\EVML Cook3r 2021-2022\\Lou, Tim, Jurg\\Dataset\\Filmpjes\\beans.MOV")
  frame = cap.getFrame()
  model = keras.models.load_model('C:/Users/Lou-J/Cook3r/DeepLearning/saved_model')
  probability_model = tf.keras.Sequential([model, 
                                          tf.keras.layers.Softmax()])
  while(cap.isSucces):
      k = cv2.waitKey(1)
      pan = Pan.Pan(frame)
      panImage = pan.getMasked()
      dim = (256,256)
      resizePan = cv2.resize(panImage,(256,256))
      cv2.imshow("resizePan",resizePan)
      cv2.imwrite("resizePan.jpg", resizePan)
      test_image = image.load_img("resizePan.jpg", target_size=(256, 256))
      test_image = image.img_to_array(test_image)
      test_image = np.expand_dims(test_image, axis=0)
      predict = probability_model.predict(test_image, batch_size=1)
      if k == 27:
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for i in range(1):  
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(predict[i],1, test_image[0])
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(predict[i], 1)
        plt.tight_layout()
        plt.show()
      
      print("Classnames: ",class_names)
      print("All predictions: ",np.round(100*predict[0]))
      predicted_label = np.argmax(predict[0])
      print("Most likely predict: ",class_names[predicted_label])
      print("prediction accuracy:",100*np.max(predict[0]),"\r\n\r\n")
      frame = cap.getFrame()
      frame = cap.getFrame()
      frame = cap.getFrame()
      frame = cap.getFrame()
      frame = cap.getFrame()
      frame = cap.getFrame()
      frame = cap.getFrame()
      frame = cap.getFrame()
      frame = cap.getFrame()