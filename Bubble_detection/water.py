# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
waterPath = 'C:/Users/Jurg Verhoeven/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Carrots-Fish_sticks-potatoes'
# Fetch the data
training = tf.keras.utils.image_dataset_from_directory(
    waterPath, 
    validation_split=0.2,
    subset="training",
    color_mode="rgb",
    image_size=(256,256),
    seed=123
)
testing = tf.keras.utils.image_dataset_from_directory(
    waterPath, 
    validation_split=0.2,
    subset="validation",
    color_mode="rgb",
    image_size=(256,256),
    seed=123
)
class_names = training.class_names

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(7)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Feed the model
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

# Check performance on the test set
test_images, test_labels = next(iter(testing))
test_loss, test_acc = model.evaluate(testing, verbose=2)
print('\nTest accuracy:', test_acc)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix

predictions = model.predict(test_images)
most_probable_predictions = np.argmax(predictions, axis=1)
cm = np.round(confusion_matrix(test_labels, most_probable_predictions, normalize='true'),1)

# Plot confusion matrix
import seaborn as sns

plt.figure()
ax4 = sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.show()
