# fashion MNIST

# import tensorflow and keras
from statistics import mode

import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

'''
Fashion MNIST is intended as a drop-in replacement for the classic MNIST dataset
often used as the "Hello, World" of machine learning programs for computer 
vision. 

The MNIST dataset contains images of handwritten digits (0, 1, 2, etc) 
in an identical format to the articles of clothing we'll use here.

This guide uses Fashion MNIST for variety, and because it's a slightly more 
challenging problem than regular MNIST. Both datasets are relatively small and 
are used to verify that an algorithm works as expected. They're good starting 
points to test and debug code.

We will use 60,000 images to train the network and 10,000 images to evaluate how
accurately the network learned to classify images. You can access the Fashion 
MNIST directly from TensorFlow, just import and load the data:
'''

fashion_mnist = keras.datasets.fashion_mnist

# loading the dataset returns four numpy arrays.
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

'''
The labels are an array of integers, ranging from 0 to 9. These correspond to
the class of clothing the image represents. Since the class names are not 
included with the dataset, store them here to use later when plotting the images:
'''
class_names = ['T-Shirt/Top', 'Trouser', 'Pullover','Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


# preprocessing the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# scaling using min-max scaling technique.
train_images = train_images/255.0
test_images = test_images/255.0

'''
Display the first 25 images from the training set and display the class name 
below each image. Verify that the data is in the correct format and we're ready
to build and train the network.
'''
plt.figure(figsize= (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# building the model

'''
Basic bldg. block = layer. most deep learning models are seq. chains of layers.
most layers like tf.keras.Dense have parameters, that are learnt during training.

    The first layer in this network, tf.keras.layers. Flatten, transforms the 
    format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 
    28 * 28 = 784 pixels. Think of this layer as unstacking rows of pixels in 
    the image and lining them up. This layer has no parameters to learn; it only
    reformats the data. 
    
    After the pixels are flattened, the network consists of a sequence of two 
    tf.keras.layers.Dense layers. These are densely-connected, or fully-
    connected, neural layers. The first Dense layer has 128 nodes (or neurons). 
    
    The second (and last) layer is a 10-node softmax layer—this returns an array
    of 10 probability scores that sum to 1. Each node contains a score that 
    indicates the probability that the current image belongs to one of the 10 
    classes.
'''
# I. define a n/w
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation= tf.nn.softmax)
])

# II. compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# III. fit the n/w
'''
- during training, data is fed to the existing compiled NN
- model learns on its own to associate labels with the data(here, it's images)
- to start training, the model is 'fit' to the training data.
'''
model.fit(train_images, train_labels, epochs=5)
# we see that model reaches an accuracy of 89% on training data.

# IV. evaluate the network.
test_loss, test_acc = model.evaluate(test_images, test_labels)

'''
it's evident that test_acc = 87%, is a bit lower than train_loss. This is due
to over fitting. This happens when machine learning model performs worse 
on new data than on training data.
'''

# V. make predictions
# ask the model to make predictions about a test set.

predictions = model.predict(test_images)

'''
the predictions is a numpy array of 10 because we set it to 10, when we
defined the n/w.

predictions is an o/p of model's predictions, and gives labels.
np.argmax(predictions[0]) => max confidence that for test_image[0], the label is 
np.argmax(predictions[0].

This can be verified by checking 
>> test_labels[0]

---
Predictions can also work on single images as follows:-
>> img = test_images[0]
>> predictions_single = model.predict(img)

>> np.argmax(predictions_single[0])

Now checking for multiple labels 
'''
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = \
        predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap= plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color = "#777777")

    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')



i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)




