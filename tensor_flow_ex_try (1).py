import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist



#loading

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train

X_train[0]

y_train

y_train[0]

#Normalizing the Images

X_train = X_train / 255.0

X_test = X_test / 255.0

X_train[0]

#Database Reshaping

X_train.shape

# As the dimension of each image is 28x28, we changed the entire database to the format [-1 (all elements), height * width]
X_train = X_train.reshape(-1, 28*28)

X_train.shape

X_train[0]

# We also changed the test base dimension
X_test = X_test.reshape(-1, 28*28)

X_test.shape

#Defining the model

model = tf.keras.models.Sequential()

model

#Adding the first dense (fully-connected) layer Layer hyper-parameters:

model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))

#Adding droupout

model.add(tf.keras.layers.Dropout(0.2))

#Adding the output layer

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

#Compiling the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.summary()

#Training the model
model.fit(X_train, y_train, epochs=5)

#Model evaluation and forecast
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy: {}".format(test_accuracy))

test_loss
