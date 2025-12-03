import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')


for i in range(10):
  print(f"Number{i} : X_train shape is {X_train[i].shape} and it has the value of {y_train[i]}")
  print(f"This X_train has the minimum value of {X_train.min()} and maximum value of {X_train.max()}")
  print()

X_train = X_train.astype('float32')/255.0
X_test  = X_test.astype('float32')/255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

print(f"New shape of X_train is {X_train.shape}")
print(f"New shape of X_test is {X_test.shape}")

from tensorflow.keras import models, layers

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.summary()