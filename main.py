import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Constants
MODEL_PATH = 'hand_gesture.tflite'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.6

# Load labels (Alphabetical order from dataset)
LABELS = ['fist', 'none', 'palm', 'thumbs_up']

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Please run the conversion script first.")
        return

    # Load TFLite model
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting inference... Press 'q' to quit.")
    
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocessing
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Format input (1, 128, 128, 1)
        input_data = np.expand_dims(gray_frame, axis=0)
        input_data = np.expand_dims(input_data, axis=-1)
        input_data = input_data.astype(np.float32)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_index = np.argmax(output_data[0])
        confidence = output_data[0][prediction_index]
        label = LABELS[prediction_index]

        # Calculation FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Draw UI
        if confidence > CONFIDENCE_THRESHOLD:
            color = (0, 255, 0)
            text = f"{label} ({confidence:.2f})"
        else:
            color = (0, 0, 255)
            text = "Unsure"

        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
