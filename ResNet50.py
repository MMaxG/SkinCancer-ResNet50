# train_skin_cancer.py (fixed: consistent rescaling, verified image copying, balanced dataset)

import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import shutil

# === 1. DOWNLOAD HAM10000 DATASET FROM KAGGLE ===
print("\nChecking for dataset...")
if not os.path.exists("ham10000"):
    os.system("kaggle datasets download kmader/skin-cancer-mnist-ham10000")
    with zipfile.ZipFile("skin-cancer-mnist-ham10000.zip", 'r') as zip_ref:
        zip_ref.extractall("ham10000")
    print("Dataset downloaded and extracted.")
else:
    print("Dataset already exists. Skipping download.")

# === 2. MERGE IMAGE FOLDERS ===
print("Merging image folders...")
image_dir = "ham10000/images"
os.makedirs(image_dir, exist_ok=True)
for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
    part_path = os.path.join("ham10000", part)
    for fname in os.listdir(part_path):
        src = os.path.join(part_path, fname)
        dst = os.path.join(image_dir, fname) # Path to images folder
        if not os.path.exists(dst):
            shutil.copy(src, dst)
print("Images merged into:", image_dir)

# === 3. LOAD, FILTER & BALANCE DATA ===
data_dir = "ham10000/"
df = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata.csv"))
df = df[df['dx'].isin(['mel', 'nv'])]
df['label'] = df['dx'].replace({'mel': 'melanoma', 'nv': 'nevus'})
df['image_path'] = image_dir + "/" + df['image_id'] + ".jpg"

# Balance: sample 1000 of each class
# 1113 melanoma images and 6705 nevus
mel_df = df[df['label'] == 'melanoma'].sample(n=1000, random_state=42, replace=False) # replace=True for overfitting (if need more images than exist)
nv_df = df[df['label'] == 'nevus'].sample(n=1000, random_state=42)
df_balanced = pd.concat([mel_df, nv_df]).sample(frac=1, random_state=42)
print("\nBalanced dataset:")
print(df_balanced['label'].value_counts())

# === 4. SPLIT DATA ===
train_df, test_df = train_test_split(df_balanced, stratify=df_balanced['label'], test_size=0.3, random_state=42)
val_df, test_df = train_test_split(test_df, stratify=test_df['label'], test_size=0.5, random_state=42)
print("\nTrain/Val/Test split:")
print("Train:")
print(train_df['label'].value_counts())
print("Validation:")
print(val_df['label'].value_counts())
print("Test:")
print(test_df['label'].value_counts())

# === 5. COPY IMAGES INTO CLASS-SPECIFIC FOLDERS ===
def create_dataset_dir(name, subset_df):
    # name: A string like "train", "val", or "test"
    # subset_df: The DataFrame (like train_df, val_df, or test_df) containing info for this set

    # 1. Define base path for this set (e.g., "dataset/train")
    base_path = f"dataset/{name}"

    # 2. Clean up old directory (if it exists)
    if os.path.exists(base_path):
        shutil.rmtree(base_path) # Remove the entire folder and its contents

    # 3. Create class subdirectories (e.g., "dataset/train/melanoma", "dataset/train/nevus")
    for class_name in ['melanoma', 'nevus']:
        # os.path.join creates the correct path like "dataset/train/melanoma"
        # os.makedirs creates the directory; exist_ok=True prevents errors if it already exists
        os.makedirs(os.path.join(base_path, class_name), exist_ok=True)

    # 4. Initialize counter for missing source files
    missing = 0

    # 5. Loop through each row in the provided DataFrame (e.g., train_df)
    for _, row in subset_df.iterrows():
        # Get the source path (where the image *should* be in the merged folder)
        src = row['image_path'] # e.g., "ham10000/images/ISIC_XXXXXX.jpg"

        # Construct the destination path
        # os.path.basename(src) gets just the filename (e.g., "ISIC_XXXXXX.jpg")
        # This builds paths like "dataset/train/melanoma/ISIC_XXXXXX.jpg"
        dst = os.path.join(base_path, row['label'], os.path.basename(src))

        # 6. Check if the source image actually exists (in ham10000/images/)
        if os.path.exists(src):
            # 7a. If it exists, copy it to the destination
            shutil.copy(src, dst)
        else:
            # 7b. If it doesn't exist, increment the missing counter
            missing += 1

    # 8. Print a summary after processing all rows for this subset
    print(f"Copied {len(subset_df) - missing} images to {base_path} ({missing} missing)")

create_dataset_dir("train", train_df)
create_dataset_dir("val", val_df)
create_dataset_dir("test", test_df)

# === 6. LOAD DATASETS ===
IMG_HEIGHT = 224 #450
IMG_WIDTH = 224 #600
BATCH_SIZE = 32 # Keep batch size for now, but likely needs reduction

print("Preparing image datasets with augmentation...")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.1),
    #tf.keras.layers.RandomBrightness(0.2),
    #tf.keras.layers.RandomContrast(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train", image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, label_mode="categorical", subset="training", validation_split=0.2, seed=42
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val", image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, label_mode="categorical", subset="validation", validation_split=0.2, seed=42
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test", image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, label_mode="categorical"
)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # Shuffle train, cache & prefetch all
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE) # Cache & prefetch val/test

# === 7. BUILD MODEL ===
print("Building ResNet50 model...")

# Load the base ResNet50 model without the top layer
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze base model initially

# Define the input and preprocessing pipeline
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs, training=True)
x = tf.keras.applications.resnet.preprocess_input(x)

# Pass through base model and add custom classifier
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

# Create the full model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# === WARM-UP PHASE ===
print("Starting warm-up training...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_warmup = model.fit(train_ds, validation_data=val_ds, epochs=5)

# === FINE-TUNING PHASE ===
print("Unfreezing all layers for fine-tuning...")

# Unfreeze some or all of the base model
# Option 1: Unfreeze all layers
#base_model.trainable = True

# Option 2: Only fine-tune deeper layers (unfreeze last 25 layers)
for layer in base_model.layers[:-25]:
    layer.trainable = False
for layer in base_model.layers[-25:]:
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=10)

model.summary()

# === 8. TRAIN MODEL ===
print("Training model...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# === 9. EVALUATE ===

print("Evaluating model on test set...")
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_true = np.argmax(y_true, axis=1)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["melanoma", "nevus"]))
print("AUC:", roc_auc_score(y_true, y_pred_probs[:, 1]))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["melanoma", "nevus"], yticklabels=["melanoma", "nevus"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.show()

# === 10. SAVE MODEL ===

