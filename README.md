# Real-Time Gesture Recognition for Mobile (Lightweight CNN)

## Overview
A lightweight gesture recognition system optimized for mobile deployment. It uses a custom CNN trained on 128x128 grayscale images and quantized for efficiency.

## Tech Stack
*   Python 3.8+
*   OpenCV, TensorFlow/Keras, NumPy

## Setup
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Execution Steps

### 1. Collect Data
Capture training images for your custom gestures.
```bash
python collect_data.py
```
-   **Keys:** `f` (Fist), `p` (Palm), `t` (Thumbs Up), `n` (None).
-   **Quit:** `q`.
-   **Goal:** Collect ~100-200 images per class for good results.

### 2. Train Model
Train the lightweight CNN on the collected data.
```bash
python train_model.py
```
-   **Output:** `hand_gesture.h5`

### 3. Convert & Optimize
Convert the model to TensorFlow Lite format (quantized).
```bash
python convert_to_tflite.py
```
-   **Output:** `hand_gesture.tflite`

### 4. Run Inference
Start the real-time recognition demo.
```bash
python main.py
```
