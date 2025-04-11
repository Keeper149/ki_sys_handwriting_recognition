# (c) Thomas Ortner

import tensorflow as tf
import numpy as np
import string
import cv2
import os
from sklearn.model_selection import train_test_split

def load_data(input_folder, size=(28, 28)):
    images = []
    labels = []
    for label, letter in enumerate(sorted(os.listdir(input_folder))):
        letter_path = os.path.join(input_folder, letter)
        if os.path.isdir(letter_path):
            for filename in os.listdir(letter_path):
                if filename.lower().endswith(".png"):
                    img_path = os.path.join(letter_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                    img_array = 1 - img_resized / 255.0
                    images.append(img_array)
                    labels.append(label)
    x_data = np.array(images).reshape(-1, 28, 28, 1).astype('float32')
    y_data = np.array(labels).astype('int')
    return x_data, y_data

def index_to_letter(index):
    return string.ascii_uppercase[index]

def predict_letters(model, x_test):
    predictions = model.predict(x_test)
    predicted_indices = np.argmax(predictions, axis=1)
    predicted_letters = [index_to_letter(idx) for idx in predicted_indices]
    return predicted_letters

if __name__ == "__main__":
    model = tf.keras.models.load_model("trained_handwriting_model_CNN.keras")

    x, y = load_data("BigDataSet")
    
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    predicted_letters = predict_letters(model, x_test)

    print("\nManuelle Überprüfung der Vorhersagen:")
    for i in range(10):
        print(f"Bild {i+1}: Richtig = {index_to_letter(y_test[i])}, Vorhersage = {predicted_letters[i]}")