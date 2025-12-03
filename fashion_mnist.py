from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(10):
  print(f"No: {i}: with the train shape of {X_train[i].shape} has the value of {y_train[i]}")
  print(f"Min of X_train is {X_train.min()} and Max of X_train is {X_train.max()}")