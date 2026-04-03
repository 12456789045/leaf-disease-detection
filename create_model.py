import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -----------------------------
# AUTO CREATE DATASET FOLDERS
# -----------------------------
classes = ["Healthy", "Powdery_Mildew", "Rust", "Leaf_Spot"]

os.makedirs("dataset", exist_ok=True)

for c in classes:
    os.makedirs(f"dataset/{c}", exist_ok=True)

print("✅ Dataset folders ready")
print("➡️ Put images inside dataset folders before training")

# -----------------------------
# CHECK DATASET
# -----------------------------
total_images = sum([len(files) for r, d, files in os.walk("dataset")])

if total_images < 8:
    print("⚠️ Very few images found. Model will still be created.")
    print("Add more images for better accuracy")

IMG = 224
BATCH = 8  # smaller batch for small dataset

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

train = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG, IMG),
    batch_size=BATCH,
    subset="training",
    class_mode="categorical",
)

val = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG, IMG),
    batch_size=BATCH,
    subset="validation",
    class_mode="categorical",
)

print("📊 Class mapping:", train.class_indices)

# -----------------------------
# MODEL
# -----------------------------
base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
out = layers.Dense(len(classes), activation="softmax")(x)

model = Model(base.input, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------
# CALLBACKS
# -----------------------------
checkpoint = ModelCheckpoint(
    "model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
)

earlystop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

# -----------------------------
# TRAIN
# -----------------------------
model.fit(train, validation_data=val, epochs=8, callbacks=[checkpoint, earlystop])

# -----------------------------
# FORCE SAVE (IMPORTANT)
# -----------------------------
model.save("model.keras", save_format="keras")

print("✅ model.keras created successfully")
print("📁 Location:", os.path.abspath("model.keras"))
