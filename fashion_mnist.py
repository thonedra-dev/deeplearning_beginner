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

