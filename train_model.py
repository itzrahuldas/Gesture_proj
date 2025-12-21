import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os
import logging
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = 'data'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 40  # Fewer epochs needed for Transfer Learning
MODEL_PATH = 'hand_gesture.keras'

def create_mobilenet_model(num_classes):
    """
    MobileNetV2 Transfer Learning Model
    """
    # Load MobileNetV2 (Pre-trained on ImageNet)
    # Include_top=False removes the final classification layers
    base_model = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), # Requires 3 channels (RGB)
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights (initially)
    base_model.trainable = False
    
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Rescaling for MobileNetV2 (expects [-1, 1] range)
        layers.Rescaling(scale=1./127.5, offset=-1),
        
        # Augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Base Model
        base_model,
        
        # Detection Head
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2), # Regularization
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

def main():
    logger.info("=" * 60)
    logger.info("STARTING TRANSFER LEARNING (MobileNetV2)")
    logger.info("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        logger.error(f"Directory '{DATA_DIR}' not found.")
        return

    # Load Data as RGB (even if files are grayscale)
    logger.info("Loading dataset (forcing RGB)...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='rgb', # FORCE RGB
        label_mode='int'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='rgb', # FORCE RGB
        label_mode='int'
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    logger.info(f"Classes: {class_names}")

    # Optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Class Weights
    train_labels = []
    for images, labels in train_ds.unbatch():
        train_labels.append(labels.numpy())
    
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    logger.info(f"Class Weights: {class_weights_dict}")

    # Build Model
    model, base_model = create_mobilenet_model(num_classes)
    model.summary(print_fn=logger.info)

    # Phase 1: Train Head (Base frozen)
    logger.info("Phase 1: Training Head Layers...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15, # Quick training for head
        class_weight=class_weights_dict
    )
    
    # Phase 2: Fine-Tuning (Unfreeze top layers)
    logger.info("Phase 2: Fine-Tuning Base Model...")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Low LR for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=history_phase1.epoch[-1],
        callbacks=[early_stopping],
        class_weight=class_weights_dict
    )

    # Save
    model.save(MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    # Evaluation
    val_images, val_labels_list = [], []
    for images, labels in val_ds.unbatch():
        val_images.append(images.numpy())
        val_labels_list.append(labels.numpy())
    val_images = np.array(val_images)
    val_labels_arr = np.array(val_labels_list)

    predictions = model.predict(val_images)
    y_pred = np.argmax(predictions, axis=1)

    print(classification_report(val_labels_arr, y_pred, target_names=class_names))
    print(confusion_matrix(val_labels_arr, y_pred))

if __name__ == "__main__":
    main()
