# Execution Log - Gesture Recognition Project

## Final Solution: Transfer Learning with MobileNetV2

### Problem Diagnosis
**Initial Issues:**
- Model showed extreme bias (only "fist" working, then only "palm" working)
- Custom CNN (grayscale) failed to generalize to real-world webcam conditions
- FPS instability due to processing bottlenecks
- Lighting/background sensitivity

### Solution Implemented
**Architecture:** MobileNetV2 Transfer Learning
- **Why:** Pre-trained on ImageNet (millions of images), robust feature extraction
- **Input:** RGB (3 channels) instead of grayscale - better generalization
- **Normalization:** Rescaling layer (1./127.5, offset -1) for [-1, 1] range
- **Training:** Two-phase approach
  - Phase 1: Train head only (15 epochs, base frozen)
  - Phase 2: Fine-tune top layers (up to 40 epochs, low learning rate)

## Final Results

### Model Performance
```
Classification Report:
              precision    recall  f1-score   support
        fist       0.98      0.98      0.98        51
        none       1.00      0.93      0.96        42
        palm       0.95      1.00      0.98        61
   thumbs_up       0.98      0.98      0.98        46

    accuracy                           0.97       200

Confusion Matrix:
[[50  0  1  0]
 [ 0 39  2  1]
 [ 0  0 61  0]
 [ 1  0  0 45]]
```

### Model Specs
- **Keras Model Size:** 27.19 MB
- **TFLite Size:** 2.70 MB (90% compression)
- **Accuracy:** 97% validation accuracy
- **All Classes Balanced:** 93-100% recall across all gestures

## Files Updated

### Core Training
- **`train_model.py`**: MobileNetV2 transfer learning, Rescaling layer, class weights, comprehensive logging
- **`convert_to_tflite.py`**: Simplified conversion (no Lambda layer issues)
- **`main.py`**: RGB preprocessing, multi-threaded capture, prediction smoothing

### Configuration
- **`requirements.txt`**: Added `scikit-learn` for metrics

## Execution Pipeline

```bash
# 1. Train the model (Transfer Learning)
py train_model.py

# 2. Convert to TFLite
py convert_to_tflite.py

# 3. Test real-time inference
py main.py
```

## Key Technical Decisions

1. **RGB over Grayscale:** MobileNetV2 requires RGB, provides better feature extraction
2. **Rescaling Layer:** Replaces Lambda layer to avoid TFLite serialization issues
3. **Two-Phase Training:** Prevents catastrophic forgetting of pre-trained features
4. **Class Weights:** Handles imbalance (e.g., fewer 'none' samples)
5. **Prediction Smoothing:** Reduces flicker in real-time inference

## Testing Notes
- **Confidence Threshold:** Set to 0.7 (70%) for robust predictions
- **FPS Optimization:** Multi-threaded capture, prediction history smoothing
- **Input Shape:** [1, 128, 128, 3] (RGB)
