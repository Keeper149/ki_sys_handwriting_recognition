# (c) Thomas Ortner

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, SeparableConv2D, BatchNormalization, MaxPooling2D, Dropout, LeakyReLU, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

# Laden der Daten
images = np.load("images_array.npy")
labels = np.load("labels_array.npy")

# Labels anpassen
labels = labels - 1

# Normierung der Bilder, falls nötig
if images.max() > 1:
    images = images / 255.0

# Falls die Bilder aktuell im Format (n, 28, 28) vorliegen, erweitere die Dimension zu (n, 28, 28, 1)
if images.ndim == 3:
    images = np.expand_dims(images, -1)

# Aufteilen in Trainings- und Testdaten
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42
)

# ---------------------------------------------------------------
# Data Augmentation mittels ImageDataGenerator
# ---------------------------------------------------------------
datagen = ImageDataGenerator(
    rotation_range=15,            # Zufälliges Rotieren um bis zu 15 Grad
    width_shift_range=0.1,        # Horizontale Verschiebung
    height_shift_range=0.1,       # Vertikale Verschiebung
    shear_range=0.1,              # Scherung
    zoom_range=0.1,               # Zoom
    horizontal_flip=False,        # Kein horizontales Spiegeln
    fill_mode="nearest"           # Methode zum Auffüllen neuer Pixel
)

datagen.fit(x_train)

# Bestimmen der Anzahl der Klassen
num_classes = int(np.max(labels)) + 1


model = Sequential([
    Input(shape=(28, 28, 1)),
    
    SeparableConv2D(64, (3, 3), padding='same'),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    SeparableConv2D(128, (3, 3), padding='same', 
                    depthwise_regularizer=tf.keras.regularizers.l2(0.001),
                    pointwise_regularizer=tf.keras.regularizers.l2(0.001)),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    SeparableConv2D(256, (3, 3), padding='same', 
                    depthwise_regularizer=tf.keras.regularizers.l2(0.001),
                    pointwise_regularizer=tf.keras.regularizers.l2(0.001)),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training des Modells mit Data Augmentation
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=50, validation_data=(x_test, y_test))

# Evaluierung des trainierten Modells
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc}")

# Speichern des trainierten Modells
model.save('trained_handwriting_model_CNN.keras')
print("Model saved as trained_handwriting_model_CNN.keras")
