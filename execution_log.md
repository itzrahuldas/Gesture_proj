# Project Execution Report: Hand Gesture Recognition

## 1. Project Overview
**Objective:** Develop a real-time Hand Gesture Recognition system capable of detecting 4 classes: `Fist`, `Palm`, `Thumbs_up`, and `None` (background) using a webcam.
**Final Outcome:** A robust system using **MobileNetV2** (Transfer Learning) with **97% accuracy**, running efficiently on CPU via **TFLite**.

---

## 2. Technology Stack
- **Language:** Python 3.11
- **Deep Learning Framework:** TensorFlow / Keras 2.x
- **Computer Vision:** OpenCV (`cv2`)
- **Inference Engine:** TensorFlow Lite (TFLite) / LiteRT
- **Data Processing:** NumPy
- **Metrics:** Scikit-learn (Classification Report, Confusion Matrix, Class Weights)

---

## 3. Execution Journey (From Zero to Hero)

### Phase 1: Inception & Data Collection
- **Action:** Created `collect_data.py`.
- **Method:** Captured 200-300 images per class using a webcam.
- **Initial Config:** 128x128 resolution, **Grayscale** integration (aiming for lightweight processing).
- **Challenge:** Data imbalance occurred immediately (fewer 'thumbs_up' and 'none' images).

### Phase 2: The Custom CNN Approach (The Struggle)
- **Action:** Built a custom lightweight CNN (3 Conv layers, MaxPool, Dropout).
- **Result:** High training accuracy (~95%) but **terrible real-world inference**.
- **Issue - Bias:** The model would get stuck predicting "Palm" or "Fist" regardless of the input.
- **Diagnosis:** The model was overfitting to specific background/lighting conditions of the training set and failed to generalize.

### Phase 3: Iterative Optimization & Debugging
- **Attempt 1 (Augmentation):** Added aggressive random rotation/zoom. *Result: 'Thumbs_up' detection capability vanished.*
- **Attempt 2 (Normalization):** Added `Rescaling(1./255)` layer logic. *Result: Training collapse (20% accuracy).*
- **Attempt 3 (Class Weights & Focal Loss):** Implemented specific penalties for misclassifying widespread classes. *Result: Improved metrics, but real-world reliability remained shaky ("only fist works").*

### Phase 4: The Pivot Strategy (Transfer Learning)
- **Problem:** Custom model lacked "robustness" (ability to see shapes, not just pixels).
- **Solution:** Switched to **MobileNetV2** (pre-trained on ImageNet).
- **Implementation:**
    - Changed input from Grayscale to **RGB** (3 channels).
    - Froze base layers, trained a new classification head (Phase 1).
    - Fine-tuned top layers with a low learning rate (Phase 2).
- **Result:** Immediate jump to superior robustness.

### Phase 5: TFLite Conversion & Deployment
- **Challenge:** `ValueError: Deserializing Lambda layer...`
    - MobileNetV2 usually uses a `Lambda` layer for internal preprocessing (`[-1, 1]` scaling).
    - TFLite converter struggled to serialize this safely/correctly in the pipeline.
- **Fix:** replaced `Lambda` with a standard Keras `Rescaling(scale=1./127.5, offset=-1)` layer.
- **Optimization:** Converted to `.tflite` (float32). Model size reduced from **~27MB** to **~2.7MB**.

---

## 4. Key Challenges & Solutions

| Challenge | Impact | Solution |
| :--- | :--- | :--- |
| **Model Bias** | Inference stuck on one class (e.g., "Palm"). | Switched from Custom CNN to **Transfer Learning (MobileNetV2)**. |
| **FPS Instability** | Laggy video feed. | Implemented **threaded video capture** and **skip-frame Inference** logic. |
| **TFLite Errors** | `Lambda` layer serialization failure. | Replaced with native **`Rescaling` layer**. |
| **Data Imbalance** | Poor accuracy on 'None'/'Thumbs_up'. | Applied **Class Weights** (`sklearn.utils.class_weight`). |
| **Flickering** | Predictions jumping rapidly. | Added **Prediction Smoothing** (Deque history buffer). |

---

## 5. Final Performance Metrics
**Validation Accuracy:** 97%  
**Inference Speed:** ~30 FPS (Real-time on CPU)

**Class-wise Performance:**
```text
              precision    recall  f1-score
        fist       0.98      0.98      0.98
        none       1.00      0.93      0.96
        palm       0.95      1.00      0.98
   thumbs_up       0.98      0.98      0.98
```

## 6. How to Run
1. **Train:** `py train_model.py` (Generates `hand_gesture.keras`)
2. **Convert:** `py convert_to_tflite.py` (Generates `hand_gesture.tflite`)
3. **Run:** `py main.py` (Opens Webcam)
