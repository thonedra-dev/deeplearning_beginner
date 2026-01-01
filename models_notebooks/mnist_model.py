# -----------------------------------------------------------
# IMPORTANT:
# Copy and paste this entire code into Google Colab.
# Do NOT run in VS Code unless TensorFlow is installed.
# Each "==== CELL ====" marks a new Colab cell.
# -----------------------------------------------------------



# ==== CELL 1 =================================================
from tensorflow.keras.datasets import mnist
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Top 5 training samples:")
for i in range(5):
    print(f"Image {i}: shape {X_train[i].shape}, label: {y_train[i]}")
    print(f"Pixel range: [{X_train[i].min()}, {X_train[i].max()}]")

print(f"\nTraining data: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Unique labels: {np.unique(y_train)}")
print(f"First 10 labels: {y_train[:10]}")



# ==== CELL 2 =================================================
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)



# ==== CELL 3 =================================================
from tensorflow.keras import models, layers

model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)



# ==== CELL 4 =================================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")



# ==== CELL 5 =================================================
pred_probs = model.predict(X_test)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y_test, axis=1)

print("First 10 True Labels:     ", true_labels[:10])
print("First 10 Predicted Labels:", pred_labels[:10])



# ==== CELL 6 =================================================
import matplotlib.pyplot as plt
import numpy as np

preds = model.predict(X_test[:10])
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_test[:10], axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap='gray')
    ax.set_title(f"Pred: {pred_labels[i]}\nTrue: {true_labels[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
