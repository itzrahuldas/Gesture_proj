import tensorflow as tf
from tensorflow.keras import layers, models
import os
import logging

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'dataset'
MODEL_PATH = 'gesture_model.h5'

def create_model(num_classes):
    """
    Creates a lightweight CNN model with < 1M parameters.
    Architecture: Conv blocks -> GlobalAveragePooling -> Dense
    """
    model = models.Sequential([
        # Input layer - 224x224x1 (Grayscale)
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Fourth Conv Block (Optional, but helps learn features)
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling to drastically reduce parameters
        layers.GlobalAveragePooling2D(),
        
        # Dense Layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_data():
    """Loads images from the dataset directory."""
    try:
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found. Please run collect_data.py first.")

        # Load training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            color_mode='grayscale', # Important since we saved as grayscale
            label_mode='int'
        )

        # Load validation dataset
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
        
        # Print class names
        class_names = train_ds.class_names
        print(f"Classes found: {class_names}")
        logging.info(f"Classes found: {class_names}")

        # Optimization: cache and prefetch
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds, len(class_names)
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        return None, None, 0

def main():
    print("Loading data...")
    train_ds, val_ds, num_classes = load_data()
    
    if train_ds is None or num_classes == 0:
        print("Could not load data. Exiting.")
        return

    print("Creating model...")
    model = create_model(num_classes)
    
    # Print model summary and usage
    model.summary()
    
    # Check parameter count
    total_params = model.count_params()
    print(f"Total Parameters: {total_params}")
    logging.info(f"Model created with {total_params} parameters.")
    
    if total_params > 1000000:
        print("WARNING: Model exceeds 1M parameters!")
        logging.warning("Model exceeds 1M parameters!")
    else:
        print("Model is lightweight (< 1M parameters).")

    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Starting training for {EPOCHS} epochs...")
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS
        )
        logging.info("Training completed successfully.")
        
        print("Saving model...")
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        logging.info(f"Model saved to {MODEL_PATH}")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
