import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Constants
MODEL_PATH = 'gesture_model.tflite'
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

# Load labels (alphabetical order of folders from collect_data step)
# dataset/fist, dataset/palm, dataset/peace, dataset/thumbs_down, dataset/thumbs_up
LABELS = ['fist', 'palm', 'peace', 'thumbs_down', 'thumbs_up']

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Please run the conversion script first.")
        return

    # Load TFLite model and allocate tensors
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print(f"Model Input Shape: {input_shape}")

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting inference... Press 'q' to quit.")
    
    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocessing
        # 1. Resize
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        
        # 2. Convert to Grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # 3. Add batch dimension and channel dimension: (1, 224, 224, 1)
        input_data = np.expand_dims(gray_frame, axis=0) # (1, 224, 224)
        input_data = np.expand_dims(input_data, axis=-1) # (1, 224, 224, 1)
        
        # 4. Check input type and convert if necessary
        # Usually TFLite Float32 models expect float32 input.
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32)
        elif input_details[0]['dtype'] == np.uint8:
            # If model is fully quantized to uint8 input
            input_data = input_data.astype(np.uint8)
        elif input_details[0]['dtype'] == np.int8:
             # If model is fully quantized to int8 input (requires offset handling usually)
             # Assuming standard float model or dynamic range (which takes float input usually)
             # But if user requested int8, input might be int8.
             # Ideally we check quantization parameters.
             input_scale, input_zero_point = input_details[0]['quantization']
             if (input_scale, input_zero_point) != (0.0, 0):
                 input_data = input_data / input_scale + input_zero_point
                 input_data = input_data.astype(np.int8)
             else:
                 # Fallback for dynamic quantization (float inputs)
                 input_data = input_data.astype(np.float32)

        # Run Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get Output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_index = np.argmax(output_data[0])
        confidence = output_data[0][prediction_index]

        # Process Results
        label = LABELS[prediction_index]
        
        # FPS Calculation
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Draw UI
        # Only show prediction if confidence is high enough
        display_text = f"{label} ({confidence:.2f})"
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)
        
        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Show Output
        cv2.imshow('Gesture Recognition (TFLite)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
