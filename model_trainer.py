# (c) Thomas Ortner

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
from sklearn.model_selection import train_test_split

images = np.load("images_array.npy")
labels = np.load("labels_array.npy")

labels = labels - 1

if images.max() > 1:
    images = images / 255.0

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42
)

num_classes = int(np.max(labels)) + 1

model = Sequential([
    Flatten(input_shape=(28, 28)),  
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),    
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc}")

model.save('trained_handwriting_model.keras')
print("Model saved as trained_handwriting_model.keras")