# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)

waterPath = 'C:/Users/Lou-J/OneDrive - HAN/EVML Cook3r 2021-2022/Lou, Tim, Jurg/Dataset/Black_pans_v2_masked_warped'
batchSize = 32
labelSize = 9
# Fetch the data
training = tf.keras.utils.image_dataset_from_directory(
    waterPath, validation_split=0.3,
    subset="training",
    seed=123
)

testing = tf.keras.utils.image_dataset_from_directory(
    waterPath, validation_split=0.3,
    subset="validation",
    seed=123
)

class_names = training.class_names

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3,activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(filters=10,kernel_size=(3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(300, activation='relu'),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dense(labelSize)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(training, validation_data=testing, epochs=5)

model.summary()

for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

# Example of visualizing some filters
# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
plt.figure()
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		ax.imshow(f[:, :, j], cmap='gray')
		ix += 1

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
  plt.xticks(range(9))
  plt.yticks([])
  thisplot = plt.bar(range(9), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label.numpy()].set_color('blue')
  
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 8
num_cols = 4
num_images = num_rows*num_cols
# get feature map for first hidden layer
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

        # Define a new Model, Input= image 
        # Output= intermediate representations for all layers in the  
        # previous model after the first.
        successive_outputs = [layer.output for layer in model.layers[1:]]
        #visualization_model = Model(img_input, successive_outputs)
        visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
        #Load the input image
        for i in range(3):
                img = test_images[i].numpy()/255.0
                x   = img_to_array(img)                           
                x   = x.reshape((1,) + x.shape)
                # Rescale by 1/255
                # Let's run input image through our vislauization network
                # to obtain all intermediate representations for the image.
                successive_feature_maps = visualization_model.predict(x)
                # Retrieve are the names of the layers, so can have them as part of our plot
                layer_names = [layer.name for layer in model.layers]
                for layer_name, feature_map in zip(layer_names, successive_feature_maps):
                  print(feature_map.shape)
                  if len(feature_map.shape) == 4:
                    
                    # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
                   
                    n_features = feature_map.shape[-1]  # number of features in the feature map
                    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
                    
                    # We will tile our images in this matrix
                    display_grid = np.zeros((size, size * n_features))
                    
                    # Postprocess the feature to be visually palatable
                    for i in range(n_features):
                      x  = feature_map[0, :, :, i]
                      x -= x.mean()
                      x /= x.std ()
                      x *=  64
                      x += 128
                      x  = np.clip(x, 0, 255).astype('uint8')
                      # Tile each filter into a horizontal grid
                      display_grid[:, i * size : (i + 1) * size] = x
                # Display the grid
                    scale = 20. / n_features
                    plt.figure( figsize=(scale * n_features, scale) )
                    plt.title ( layer_name )
                    plt.grid  ( False )
                    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
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
