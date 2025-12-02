# mnist_training.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

# ---------------------------
# 1. Load MNIST dataset
# ---------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Print top 5 samples info
print("Top 5 training samples:")
for i in range(5):
    print(f"Image {i}: shape {X_train[i].shape}, label: {y_train[i]}")
    print(f"Pixel range: [{X_train[i].min()}, {X_train[i].max()}]")
print()

# Dataset info
print(f"Training data: {X_train.shape}")   # (60000, 28, 28)
print(f"Training labels: {y_train.shape}") # (60000,)
print(f"Test data: {X_test.shape}")       # (10000, 28, 28)
print(f"Unique labels: {np.unique(y_train)}")  # [0-9]
print(f"First 10 labels: {y_train[:10]}")  # Show first 10 actual labels
print()

# ---------------------------
# 2. Preprocess data
# ---------------------------
# Normalize pixel values to [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels (10 classes)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Verify preprocessing
print("After preprocessing:")
print(f"X_train min/max: {X_train.min()}, {X_train.max()}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"First label one-hot: {y_train[0]}")
print()

# ---------------------------
# 3. Build the model
# ---------------------------
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # Flatten 28x28 → 784
    layers.Dense(10, activation='softmax')  # Output layer with 10 neurons
])

# Show model summary
model.summary()

# ---------------------------
# 4. Compile the model
# ---------------------------
model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# 5. Train the model
# ---------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ---------------------------
# 6. Evaluate on test data
# ---------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
print()

# ---------------------------
# 7. Predictions on test data
# ---------------------------
pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)    # Predicted classes
true_labels = np.argmax(y_test, axis=1)        # True classes

# Print first 10 actual vs predicted
print("First 10 True Labels:     ", true_labels[:10])
print("First 10 Predicted Labels:", pred_labels[:10])
print()

# ---------------------------
# 8. Visualize first 10 predictions
# ---------------------------
fashion_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap='gray')
    ax.set_title(f"Pred: {fashion_labels[pred_labels[i]]}\nTrue: {fashion_labels[true_labels[i]]}")
    ax.axis('off')
plt.tight_layout()
plt.show()
