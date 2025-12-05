import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, regularizers, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time


# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("⚡ Using TensorFlow", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# Normalization

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

# One-Hot Encoding

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Validation Splitting
print(f"Befire_Splitting: Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")
val_split = 0.1
val_samples = int(len(x_train) * val_split)
x_val, y_val = x_train[-val_samples:], y_train[-val_samples:]
x_train, y_train = x_train[:-val_samples], y_train[:-val_samples]

print(f"After Splitting: Train: {x_train.shape[0]}, Val: {x_val.shape[0]}, Test: {x_test.shape[0]}")


# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Plot first 20 images
plt.figure(figsize=(12, 6))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[np.argmax(y_train[i])])
    plt.axis('off')
plt.tight_layout()
plt.show()


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

def build_efficient_cifar10_model():
    """Compact but powerful model for CIFAR-10"""
    model = keras.Sequential([

        # Block 1

        # Layer 1
        # Output: (32, 32, 32) — 32 filters analyze input from different perspectives; padding='same' so, 32*32 will not be influenced by 3*3.
        # Number of Neurons: 32 * 32 * 32 = 32,768
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),

        # Layer 2
        # Input: (32,32,32), normalize values in each cell from 32 rows and 32 cols.
        # Number of Neurons: still 32,768
        layers.BatchNormalization(),

        # Layer 3
        layers.Activation('relu'),

        # Layer 4
        # Input shape same (32, 32, 32), second Conv2D inspects same input from 32 perspectives again.
        # Neurons remain 32,768
        layers.Conv2D(32, (3, 3), padding='same'),

        # Layer 5
        layers.BatchNormalization(),

        # Layer 6
        layers.Activation('relu'),

        # Layer 7
        # MaxPooling reduces spatial size from (32, 32) to (16, 16)
        # Neurons = 16 * 16 * 32 = 8,192
        layers.MaxPooling2D((2, 2)),

        # Layer 8
        # Dropout 25% of neurons randomly
        # Effective active neurons ≈ 8,192 * 0.75 ≈ 6,144, shape still (16, 16, 32)
        layers.Dropout(0.25),


        # Block 2

        # Layer 9
        # Input shape: (16, 16, 32), increase filters to 64 for more detailed perspectives
        # Output shape: (16, 16, 64), neurons = 16 * 16 * 64 = 16,384
        layers.Conv2D(64, (3, 3), padding='same'),

        # Layer 10
        layers.BatchNormalization(),

        # Layer 11
        layers.Activation('relu'),

        # Layer 12
        # Another Conv2D with 64 filters, shape unchanged, values refined
        layers.Conv2D(64, (3, 3), padding='same'),

        # Layer 13
        layers.BatchNormalization(),

        # Layer 14
        layers.Activation('relu'),

        # Layer 15
        # MaxPooling reduces spatial size from (16,16) to (8,8)
        # Neurons = 8 * 8 * 64 = 4,096
        layers.MaxPooling2D((2, 2)),

        # Layer 16
        # Dropout 25% of neurons
        # Effective active neurons ≈ 4,096 * 0.75 ≈ 3,072, shape still (8, 8, 64)
        layers.Dropout(0.25),


        # Block 3

        # Layer 17
        # Input shape: (8, 8, 64), increase filters to 128 for deeper feature extraction
        # Output shape: (8, 8, 128), neurons = 8 * 8 * 128 = 8,192
        layers.Conv2D(128, (3, 3), padding='same'),

        # Layer 18
        layers.BatchNormalization(),

        # Layer 19
        layers.Activation('relu'),

        # Layer 20
        # Another Conv2D with 128 filters, values refined, shape unchanged (8, 8, 128).
        layers.Conv2D(128, (3, 3), padding='same'),

        # Layer 21
        layers.BatchNormalization(),

        # Layer 22
        layers.Activation('relu'),

        # Layer 23
        # MaxPooling reduces spatial size from (8,8) to (4,4)
        # Neurons = 4 * 4 * 128 = 2,048
        layers.MaxPooling2D((2, 2)),

        # Layer 24
        # Dropout 25% of neurons
        # Effective active neurons ≈ 2,048 * 0.75 ≈ 1,536, shape still (4, 4, 128)
        layers.Dropout(0.25),


        # Dense layers

        # Layer 25 :  in here, we Flatten(4, 4, 128), so it iwll become a list [    ] of count 2048.
        layers.Flatten(),

        # Layer 26: we regulaize so, that values of 2048 from previous layers, their weights cannot go too beyond, to prevent overfitting.
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),

        # Layer 27
        layers.BatchNormalization(),
        # Layer 28
        layers.Activation('relu'),
        # Layer 29
        layers.Dropout(0.5),

        # Layer 30: we regulaize again when entering the 128 neurons layer.
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        # Layer 31, 32, 33
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        # Output Layer: Finally, as we have ten multi-classes, yep, there will be only 10 neurons.
        layers.Dense(10, activation='softmax')
    ])

    return model

model = build_efficient_cifar10_model()

# Quick compile
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n🔄 Creating simple augmentation...")
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Minimal augmentation for speed
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# This is the part of Data Augmentation, and the concept here is that,
# It will edit the images of each samples into slightly differnet version such as,
# horizontal_flipping, a bit degree changing, zooming in or out, like that,
# So, the image is that image, but become slightly differnet in terms of the structure and format.
# Data Augmentation is largely needed, so that, the model can really understand what is that image, no matter  how many degrees or angles it is changed.
# But take note that, yeah, it does not mean, it will crate new samples in x_train set, haha.
# Instead, everytime the model is trained on x_train, every epochs is runned, datagen will create a slightly different version of a sample and will replace.
# Lets say, if there are 10 epochs, it will be evolved 9 times. ( as the very first epoch is original epoch)

datagen.fit(x_train)


