# Real-Time Gesture Recognition for Mobile (Lightweight CNN)

## Overview
This project implements a lightweight, real-time hand gesture recognition system optimized for mobile and edge devices. It utilizes a custom Convolutional Neural Network (CNN) architecture designed for efficiency, achieving a high frame rate (DFS) while maintaining a minimal memory footprint (< 5MB) suitable for deployment on iOS, Android, or IoT devices via TensorFlow Lite.

## Tech Stack
*   **Language:** Python 3.8+
*   **Computer Vision:** OpenCV (`cv2`)
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Edge Inference:** TensorFlow Lite Interpreter
*   **Data Processing:** NumPy

## Model Optimization & Architecture
The core of this system is a custom lightweight CNN specifically engineered for resource-constrained environments. Key optimization strategies include:

*   **Custom Architecture:** A 4-block convolutional network utilizing `GlobalAveragePooling2D` to drastically reduce parameter count compared to traditional Flatten/Dense layers.
*   **Parameter Count:** The model performs inference with significantly fewer than 1 million parameters.
*   **Post-Training Quantization:** The model is converted to TensorFlow Lite format using Post-Training Quantization (Dynamic Range), which quantizes weights to 8-bit precision.
*   **Target Size:** The final `.tflite` binary is optimized to be **under 5MB**, ensuring fast download and initialization times on mobile networks.

## Setup Instructions

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The workflow consists of four distinct stages: Data Collection, Training, Optimization, and Inference.

### 1. Data Collection
Capture training data using your webcam. Images are resized to 224x224 grayscale and saved into labeled directories.
```bash
python collect_data.py
```
*   **Controls:** Press `0`-`4` to save images for different classes (Fist, Palm, Thumbs Up, Thumbs Down, Peace). Press `q` to quit.

### 2. Model Training
Train the CNN on the collected dataset. The script automatically splits data into training and validation sets and saves the potentially large Keras model (`.h5`).
```bash
python train_model.py
```
*   **Artifact:** `gesture_model.h5`

### 3. Conversion & Optimization
Convert the Keras model to a quantized TensorFlow Lite model. This step applies optimization techniques to reduce the file size.
```bash
python convert_to_tflite.py
```
*   **Artifact:** `gesture_model.tflite` (Verified size < 5MB)

### 4. Real-Time Inference
Run the optimized model using the TFLite Interpreter for high-performance inference on the webcam feed.
```bash
python main.py
```
*   **Output:** Real-time video feed with bounding box (optional), predicted label, confidence score, and FPS.

## Directory Structure
```
Gesture_proj/
├── dataset/                  # Generated during data collection
├── scripts/                  # (Optional: Folder organization)
├── collect_data.py           # Data acquisition script
├── train_model.py            # Training pipeline
├── convert_to_tflite.py      # Conversion and Quantization
├── main.py                   # Inference engine
├── requirements.txt          # Python dependencies
├── gesture_model.h5          # Intermediate Keras model
└── gesture_model.tflite      # Deployment-ready model
```
