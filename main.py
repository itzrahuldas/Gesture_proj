import cv2
import numpy as np
import time
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# Constants
MODEL_PATH = 'hand_gesture.tflite'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.7 # Higher threshold for robust model

# Labels
LABELS = ['fist', 'none', 'palm', 'thumbs_up']

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    print("Loading TFLite model...")
    # Add delegate if needed, but CPU is usually fine for MobileNetV2 on PC
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting inference... Press 'q' to quit.")
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    prediction_history = []
    HISTORY_SIZE = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing for MobileNetV2 (RGB, 128x128)
        # 1. Resize
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        
        # 2. Convert to RGB (OpenCV is BGR)
        # MobileNet expects RGB input
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # 3. Add batch dimension and type
        # Note: MobileNetV2 preprocessing ([-1, 1] scaling) is BUILT-IN to the model via layers.Lambda
        # So we just pass raw [0, 255] float32 values.
        input_data = rgb_frame.astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print(f"Raw: {output_data[0]}") # Debug
        
        prediction_index = int(np.argmax(output_data[0]))
        confidence = float(output_data[0][prediction_index])
        
        # Smooth predictions
        prediction_history.append((prediction_index, confidence))
        if len(prediction_history) > HISTORY_SIZE:
            prediction_history.pop(0)
        
        # Vote
        if prediction_history:
            indices = [p[0] for p in prediction_history]
            smoothed_index = max(set(indices), key=indices.count)
            smoothed_conf = np.mean([p[1] for p in prediction_history if p[0] == smoothed_index])
        else:
            smoothed_index = prediction_index
            smoothed_conf = confidence
        
        label = LABELS[smoothed_index] if smoothed_index < len(LABELS) else "Unknown"

        # FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # UI
        if smoothed_conf > CONFIDENCE_THRESHOLD:
            color = (0, 255, 0) # Green
            text = f"{label} ({int(smoothed_conf * 100)}%)"
        else:
            color = (0, 0, 255) # Red
            text = f"Waiting..."

        # Draw
        cv2.rectangle(frame, (5, 5), (320, 100), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Mini preview output
        # cv2.imshow('Input', rgb_frame) # Optional debug

        cv2.imshow('Gesture Recognition (MobileNetV2)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
