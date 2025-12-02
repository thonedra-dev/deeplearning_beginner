# 1️⃣ Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

# 2️⃣ Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 3️⃣ Explore the dataset
print("First 5 images info:")
for i in range(5):
    print(f"Image {i} shape: {X_train[i].shape}, Label: {y_train[i]}")
    print(f"Pixel range: [{X_train[i].min()}, {X_train[i].max()}]\n")

print(f"Training data shape: {X_train.shape}")  # (60000, 28, 28)
print(f"Test data shape: {X_test.shape}")       # (10000, 28, 28)
print(f"Unique labels: {np.unique(y_train)}")   # 0-9 classes

# 4️⃣ Preprocess the data
# Normalize pixel values to [0,1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Reshape for CNN (samples, rows, cols, channels)
X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Verify preprocessing
print("After preprocessing:")
print(f"X_train min/max: {X_train.min()}, {X_train.max()}")
print(f"X_train shape (for CNN): {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"First label one-hot: {y_train[0]}")

# 5️⃣ Build a simple CNN model
model = models.Sequential()

# Convolutional layer
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
# Pooling
model.add(layers.MaxPooling2D((2,2)))
# Flatten before Dense
model.add(layers.Flatten())
# Dense output layer
model.add(layers.Dense(10, activation='softmax'))

# 6️⃣ Show model summary
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

# ✅ At this point, the model is trained and ready for predictions
# You can later use model.predict(X_test) to get final predictions
# Evaluate the trained model on test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")


# Get predictions (probabilities)
pred_probs = model.predict(X_test)

# Convert probabilities → class labels
pred_labels = np.argmax(pred_probs, axis=1)

# Convert one-hot y_test → class labels
true_labels = np.argmax(y_test, axis=1)

# Print first 10 results
print("First 10 True Labels:     ", true_labels[:10])
print("First 10 Predicted Labels:", pred_labels[:10])

import matplotlib.pyplot as plt

# Map integer labels to fashion item names
fashion_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28,28), cmap='gray')
    ax.set_title(f"Pred: {fashion_labels[pred_labels[i]]}\nTrue: {fashion_labels[true_labels[i]]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
