# (c) Thomas Ortner

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("trained_handwriting_model.keras")  

def predict_letter(probabilities):
    index = np.argmax(probabilities)
    return chr(index + ord('A'))  

class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("Buchstabenerkennung")

        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(master, text="Erkennen", command=self.predict_drawing)
        self.button_predict.pack()

        self.button_clear = tk.Button(master, text="Löschen", command=self.clear_canvas)
        self.button_clear.pack()

        self.label_result = tk.Label(master, text="Erkannt: ")
        self.label_result.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def predict_drawing(self):
        # Bild vorbereiten
        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        image_array = 1 - np.array(resized_image).astype(np.float32) / 255.0
        image_array = image_array.reshape(1, 28, 28)

        # Modellvorhersage
        prediction = model.predict(image_array)[0]

        # Normierung auf Summe = 1, falls keine Softmax im Modell ist
        if not np.isclose(np.sum(prediction), 1.0, atol=1e-3):
            prediction = prediction / np.sum(prediction)

        # Beste Vorhersage
        index = np.argmax(prediction)
        letter = chr(index + ord('A'))
        probability = prediction[index] * 100

        # Ausgabe im GUI
        self.label_result.config(text=f"Erkannt: {letter} ({probability:.2f}%)")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
