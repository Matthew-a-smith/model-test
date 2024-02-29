import tensorflow as tf
import numpy as np

# Load the inference model
inference_model_path = "Models/Inferance/NMAP-inference_model.keras"
inference_model = tf.keras.models.load_model(inference_model_path)

# Convert the Keras model to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
tflite_model_path = "your_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

# You can now use the saved TensorFlow Lite model on your Raspberry Pi for inference.
