import tensorflow as tf
import numpy as np
from tensorflow import keras

mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

permutation = np.random.permutation(len(X_train_full))
X_train_full = X_train_full[permutation]
y_train_full = y_train_full[permutation]

fashion_mnist_valid_X , fashion_mnist_train_X = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
fashion_mnist_valid_y , fashion_mnist_train_y = y_train_full[:5000], y_train_full[5000:]


model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=7, strides = 1,
input_shape = [28, 28, 1], activation='relu', padding='SAME'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1,
padding="SAME", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.Conv2D(filters=128, kernel_size=3, strides=1,
padding="SAME", activation="relu"))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(rate=0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(60, activation="relu"))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(20, activation="relu"))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(11, activation="softmax"))

model.summary()

model.compile(keras.optimizers.Adam(learning_rate=0.001),
loss="SparseCategoricalCrossentropy",
metrics=["accuracy"])

history = model.fit(fashion_mnist_train_X, fashion_mnist_train_y, epochs=30,
    validation_data=(fashion_mnist_valid_X, fashion_mnist_valid_y))


model.evaluate(X_test, y_test)

model.save('cnn_fashion_mnist.h5')
