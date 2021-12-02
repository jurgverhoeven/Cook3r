# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

waterPath = 'C:/Users/Lou/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Machinelearning/Bubbles/rotated'
batchSize = 32
# Fetch the data
training = tf.keras.utils.image_dataset_from_directory(
    waterPath, validation_split=0.2,
    subset="training",
    seed=123
)

testing = tf.keras.utils.image_dataset_from_directory(
    waterPath, validation_split=0.2,
    subset="validation",
    seed=123
)

class_names = training.class_names

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(training, validation_data=testing, epochs=10)

# Show history
print(history.history.keys())

# Plot learning curves
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='loss (training data)')
ax.plot(history.history['val_loss'], label='loss (validation data)')
ax.plot(history.history['accuracy'], label='accuracy (training data)')
ax.plot(history.history['val_accuracy'], label='accuracy (validation data)')
ax.set_title('Learning curves')
ax.set_ylabel('value')
ax.set_xlabel('No. epoch')
ax.grid(True)
ax.set_ylim(0,1)
ax.legend(loc="lower right")
plt.show()

test_loss, test_acc = model.evaluate(testing, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

def plot_image(predictions_array, true_label, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img.numpy()/255.0, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label.numpy()]),
                                color=color)

def plot_value_array(predictions_array, true_label):
  plt.grid(False)
  plt.xticks(range(2))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label.numpy()].set_color('blue')
  
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 8
num_cols = 4
num_images = num_rows*num_cols
for test_images, test_labels in testing:
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        predict = probability_model.predict(test_images)
        for i in range(batchSize):  
            plt.subplot(num_rows, 2*num_cols, 2*i+1)
            plot_image(predict[i],test_labels[i], test_images[i])
            plt.subplot(num_rows, 2*num_cols, 2*i+2)
            plot_value_array(predict[i], test_labels[i])
        plt.tight_layout()
        plt.show()
        break

from sklearn.metrics import confusion_matrix

for test_images, test_labels in testing:
    predictions = model.predict(test_images)
    most_probable_predictions = np.argmax(predictions, axis=1)
    cm = np.round(confusion_matrix(test_labels, most_probable_predictions, normalize='true'),1)
    break

# Plot confusion matrix
import seaborn as sns

plt.figure()
ax4 = sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()