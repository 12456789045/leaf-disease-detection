import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "dataset"  # folder with subfolders for each class
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_TOP = 10  # train top layers first
EPOCHS_FINE = 10  # fine-tuning deeper layers
NUM_CLASSES = len(os.listdir(DATASET_DIR))  # automatically get number of classes
MODEL_PATH = "leaf_model.keras"

# -----------------------------
# Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
    validation_split=0.2,  # 20% data for validation
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
)

# -----------------------------
# Model Architecture
# -----------------------------
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False  # freeze base initially

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
]

# -----------------------------
# Train top layers first
# -----------------------------
history_top = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_TOP,
    callbacks=callbacks,
)

# -----------------------------
# Fine-tuning
# -----------------------------
# Unfreeze last 50 layers for fine-tuning
for layer in base.layers[-50:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINE,
    callbacks=callbacks,
)

# -----------------------------
# Save final model
# -----------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
