"""
Face Mask Detection using MobileNetV2
=====================================
Real-time face mask detection using a fine-tuned MobileNetV2 model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(input_shape=(224, 224, 3), num_classes=2):
    """Build fine-tuned MobileNetV2 for mask detection."""
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_data_generators(train_dir, val_dir, batch_size=32):
    """Create augmented data generators."""
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=batch_size, class_mode="categorical"
    )
    val_data = val_gen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=batch_size, class_mode="categorical"
    )
    return train_data, val_data


def train(train_dir, val_dir, epochs=20, model_save_path="mask_detector.h5"):
    model = build_model()
    train_data, val_data = get_data_generators(train_dir, val_dir)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True),
    ]

    history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)
    return model, history


if __name__ == "__main__":
    # Example usage — update paths to your dataset
    model, history = train(
        train_dir="dataset/train",
        val_dir="dataset/val",
        epochs=20,
    )
    print("Training complete. Model saved to mask_detector.h5")
