# Thank you to the following resources:
# https://www.tensorflow.org/tutorials/keras/classification
    # start of the tutorial from tensorflow's website
    # this is the tutorial that I'm following throughout the code

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__) 2.1.0

fashion_mnist = keras.datasets.fashion_mnist
# mnist is Modified National Institute of Standards and Technology database

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# train_images and train_labels are training set -> data model uses to learn
# test_images and test_labels are test set -> data model is tested against

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print(train_images.shape) (60000, 28, 28)
# 28x28 pixel images, 60000 of them
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
