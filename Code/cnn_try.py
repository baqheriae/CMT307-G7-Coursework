# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

with open('data_2.pickle', 'rb') as f:
    X = pickle.load(f)
with open('data_y_2.pickle', 'rb') as f:
    y = pickle.load(f)

#Split into training set test set(80% 20%)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Normalized
x_train = X_train / 255.0
x_test = X_test / 255.0

#Convolutional Neural Network Example
#activation function = Rectified Linear Unit, ReLU
#ReLU activation for hidden layers
#Means of Early stopping and Dopout are showed on the slide CMT307_Session12.pdf
#some Arguments means of model can be found at https://keras.io/models/sequential/
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                        input_shape=[200, 200, 3]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# reshape: map data to 4D, with the last dimension of 1 channel (grayscale)
history = model.fit(x_train.reshape((x_train.shape[0], 200, 200, 3)), y_train, epochs=60, validation_split=0.1,
                    callbacks=[early_stopping_cb])
model.evaluate(x_test.reshape(x_test.shape[0], 200, 200, 3), y_test)

#Test

model.evaluate(x_test.reshape(x_test.shape[0], 200, 200, 3), y_test)

#save

model.save("model-cnn.hdf5")


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc = 'upper left')

plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc = 'upper left')

plt.show()
