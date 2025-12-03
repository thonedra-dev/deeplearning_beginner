from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(10):
  print(f"No: {i}: with the train shape of {X_train[i].shape} has the value of {y_train[i]}")
  print(f"Min of X_train is {X_train.min()} and Max of X_train is {X_train.max()}")

print(f"Training data shape: {X_train.shape}")  
print(f"Test data shape: {X_test.shape}")      
print(f"Unique labels: {np.unique(y_train)}")   # 0-9 classes


X_train = X_train.astype('float32')/255.0
X_test  = X_test.astype('float32')/255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_train.reshape(-1, 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from tensorflow.keras import models, layers
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# 7️⃣ Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 8️⃣ Fit the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)