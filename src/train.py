"""
LeafGuard AI - Training Module
Defines transfer learning model structure using MobileNetV2 for crop disease classification.
Includes model compilation, data generator initialization, and training execution.
"""

try:
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
except ImportError:
    tf = None

from src.config import (
    IMAGE_SIZE,
    IMG_CHANNELS,
    LEARNING_RATE,
    NUM_CLASSES,
    MODEL_SAVE_PATH,
    EPOCHS,
    BATCH_SIZE,
    TRAIN_DATA_DIR,
    VAL_DATA_DIR,
)
from src.preprocessing import create_data_generators


def build_mobilenet_model(num_classes=NUM_CLASSES, input_shape=(*IMAGE_SIZE, IMG_CHANNELS)):
    """
    Builds an image classification model using MobileNetV2 pre-trained on ImageNet.
    Uses transfer learning by freezing base layers and adding a classification head.
    
    Args:
        num_classes: Number of disease categories to classify (default: 17 for LeafGuard v1).
        input_shape: Input image dimensions (H, W, C).
        
    Returns:
        Compiled Keras Model instance.
    """
    if tf is None:
        raise ImportError("TensorFlow must be installed to build the MobileNetV2 model.")

    # 1. Load pre-trained MobileNetV2 base model without top classification layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model layers for feature extraction
    base_model.trainable = False

    # 2. Attach custom classification head for 17 target disease classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # 3. Create full model architecture
    model = Model(inputs=base_model.input, outputs=outputs)

    # 4. Compile model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, train_generator, val_generator, epochs=EPOCHS, save_path=MODEL_SAVE_PATH):
    """
    Executes training loop with callbacks for checkpointing best model and early stopping.
    
    Args:
        model: Compiled Keras classification model.
        train_generator: ImageDataGenerator for training set.
        val_generator: ImageDataGenerator for validation set.
        epochs: Number of training epochs.
        save_path: Output file path for saved model checkpoint.
        
    Returns:
        Keras History object containing training metrics per epoch.
    """
    if tf is None:
        raise ImportError("TensorFlow must be installed to run training.")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    return history


if __name__ == "__main__":
    print("=== LeafGuard AI v1 Model Training Pipeline ===")
    print(f"Dataset Directories:\n - Train: {TRAIN_DATA_DIR}\n - Validation: {VAL_DATA_DIR}")
    
    # 1. Create data generators for training and validation datasets
    train_gen, val_gen = create_data_generators(
        train_dir=TRAIN_DATA_DIR,
        val_dir=VAL_DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # 2. Build MobileNetV2 transfer learning model (17 output classes)
    print(f"\nBuilding MobileNetV2 architecture for {NUM_CLASSES} classes...")
    model = build_mobilenet_model(num_classes=NUM_CLASSES)
    
    # 3. Start training execution
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = train_model(
        model=model,
        train_generator=train_gen,
        val_generator=val_gen,
        epochs=EPOCHS,
        save_path=MODEL_SAVE_PATH
    )
    
    print(f"\nTraining completed successfully! Best model saved to: {MODEL_SAVE_PATH}")
