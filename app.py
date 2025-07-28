import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("tomato_model.h5")

class_names = ['blightTomatoes', 'healthyTomatoes', 'mosaicTomatoes', 'saptoriaTomatoes']

def predict(img):
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    predictions = model.predict(img_array)[0]
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="🍅 Tomato Disease Detector",
    description="Upload a tomato leaf image to detect diseases like mosaic, septoria, blight, or check if it's healthy."
)

demo.launch()
