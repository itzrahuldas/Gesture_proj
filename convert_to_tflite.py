import tensorflow as tf
import os

# 1. Load the model
model_path = 'hand_gesture.keras'
print(f"Loading {model_path}...")

model = tf.keras.models.load_model(model_path)

# 2. Convert to TFLite
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization (Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3. Save the TFLite model
tflite_path = 'hand_gesture.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print("------------------------------------------------")
print(f"Success! Model saved as '{tflite_path}'")

keras_size = os.path.getsize(model_path) / (1024 * 1024)
tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)

print(f"Keras Size:  {keras_size:.2f} MB")
print(f"TFLite Size: {tflite_size:.2f} MB")
print("------------------------------------------------")
