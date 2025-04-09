# (c) Thomas Ortner

import os
import cv2
import numpy as np

folder = "BigDataSet"

final_width = 28
final_height = 28

images = []
labels = []

label = 1

for letter in os.listdir(folder):
    letter_folder = os.path.join(folder, letter)
    if os.path.isdir(letter_folder):
        for file in os.listdir(letter_folder):
            if file.lower().endswith('.png'):
                file_path = os.path.join(letter_folder, file)
                image = cv2.imread(file_path, 0)
                resized_img = cv2.resize(image, (final_width, final_height))
                norm_img = 1 - resized_img.astype(np.float32) / 255.0
                images.append(norm_img)
                labels.append(label)
        label += 1

images_array = np.array(images)
labels_array = np.array(labels)

np.save("images_array.npy", images_array)
np.save("labels_array.npy", labels_array)