import tensorflow as tf
import os
import logging

# Configure logging
logging.basicConfig(
    filename='conversion.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_PATH = 'gesture_model.h5'
TFLITE_PATH = 'gesture_model.tflite'

def convert_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Initializing TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations (Post-Training Quantization)
    print("Applying Post-Training Quantization (Default/Dynamic Range)...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # To strictly enforce float16 quantization (optional alternate if needed)
    # converter.target_spec.supported_types = [tf.float16]
    
    try:
        tflite_model = converter.convert()
        
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
            
        logging.info(f"Model converted and saved to {TFLITE_PATH}")
        print(f"Success! Model converted to {TFLITE_PATH}")
        
        # Check size
        size_bytes = os.path.getsize(TFLITE_PATH)
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"Final TFLite Model Size: {size_mb:.2f} MB")
        logging.info(f"Final TFLite Model Size: {size_mb:.2f} MB")
        
        if size_mb < 5:
            print("Pass: Model is under 5MB.")
        else:
            print("Warning: Model is larger than 5MB.")
            
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    convert_model()
