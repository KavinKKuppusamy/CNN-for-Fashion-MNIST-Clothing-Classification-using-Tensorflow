# based on code from https://www.tensorflow.org/tutorials

import tensorflow as tf
import numpy as np

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)

tf.random.set_seed(1)

# specify path to training data and testing data


folderbig = "big"
foldersmall = "small"

train_x_location = foldersmall + "/" + "train_x.csv"
train_y_location = foldersmall + "/" + "train_y.csv"
test_x_location = folderbig + "/" + "test_x.csv"
test_y_location = folderbig + "/" + "test_y.csv"

print("Reading training data")
x_train_2d = np.loadtxt(train_x_location, dtype="uint8")
x_train_3d = x_train_2d.reshape(-1,28,28,1)
x_train = x_train_3d
y_train = np.loadtxt(train_y_location, dtype="uint8")

print("Pre processing x of training data")
x_train = x_train / 255.0

#define the training model

model = tf.keras.models.Sequential([
    tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(1, 1),
    tf.keras.layers.Conv2D(20, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(1, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer = tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()

optimizer=tf.keras.optimizers.Nadam(learning_rate=0.003)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print("train")
model.fit(x_train, y_train, epochs=15)

print("Reading testing data")
x_test_2d = np.loadtxt(test_x_location, dtype="uint8")
x_test_3d = x_test_2d.reshape(-1,28,28,1)
x_test = x_test_3d
y_test = np.loadtxt(test_y_location, dtype="uint8")

print("Pre processing testing data")
x_test = x_test / 255.0

print("evaluate")
model.evaluate(x_test, y_test)
