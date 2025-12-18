import tensorflow as tf
from tensorflow.keras import layers, models
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants
DATA_DIR = 'data'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = 'hand_gesture.h5'

def create_model(num_classes):
    """
    Simple Lightweight CNN.
    Input: (128, 128, 1)
    Layers: 3 Conv + MaxPool -> Flatten -> Dense -> Output
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # 1st Conv Block
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 2nd Conv Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # 3rd Conv Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten & Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    print("Loading dataset...")
    # Load data from 'data' folder
    # We use validation_split to create train/val sets
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='grayscale',
            label_mode='int'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='grayscale',
            label_mode='int'
        )
        
        class_names = train_ds.class_names
        num_classes = len(class_names)
        print(f"Classes found ({num_classes}): {class_names}")
        
    except ValueError as e:
        print(f"Error loading data: {e}")
        print("Make sure 'data' folder contains subfolders with images.")
        return

    # Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build Model
    model = create_model(num_classes)
    model.summary()

    # Check params
    total_params = model.count_params()
    print(f"Total Parameters: {total_params}")
    if total_params < 1000000:
        print("Check: Model is under 1 Million parameters.")
    else:
        print("Warning: Model exceeds 1 Million parameters.")

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print(f"Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Save
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
