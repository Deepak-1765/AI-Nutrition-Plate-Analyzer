"""
train_model.py
----------------
Trains a food image classifier using transfer learning on top of
MobileNetV2 (pretrained on ImageNet).

This version of the project is trained on a custom Indian food dataset
(12 classes — see class_names.json / nutrition.csv), collected and
labeled independently rather than pulled from Food-101. The training
pipeline itself is dataset-agnostic: point it at any ImageFolder-style
directory and it will happily train on whatever classes you put there,
as long as nutrition.csv has a matching row for each class.

DATASET LAYOUT EXPECTED
------------------------
This script expects a directory of images organized like this
(the standard "ImageFolder" layout, one sub-folder per class):

    dataset/
        train/
            aloo_gobi/
                img_001.jpg
                img_002.jpg
                ...
            biryani/
                ...
            ...
        val/
            aloo_gobi/
                ...
            biryani/
                ...

If you started from the official Food-101 dataset
(https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz) instead, that
download contains a single `images/<class_name>/*.jpg` folder plus text
files (`meta/train.txt`, `meta/test.txt`) that define the train/test
split. The optional helper function `build_dataset_from_food101()`
below can copy just the classes you support into the
`dataset/train` / `dataset/val` layout shown above — it's unused for
this custom dataset but kept here in case you extend the project with
Food-101 classes later.

USAGE
-----
    # (optional) only relevant if pulling classes from a raw Food-101 download
    python train_model.py --prepare --food101_dir /path/to/food-101

    # train the model on your own dataset/train + dataset/val folders
    python train_model.py --data_dir dataset --epochs 15 --fine_tune_epochs 5

OUTPUT
------
    food_model.keras   -> trained Keras model (native Keras v3 format)
    class_names.json   -> list of class names, in the exact order used
                           by the model's output layer (index -> label)
    training_history.png -> accuracy / loss curves for the run
"""

import argparse
import json
import os
import shutil

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)   # required input size for MobileNetV2
BATCH_SIZE = 32
SEED = 42


def build_dataset_from_food101(food101_dir: str, supported_classes: list, out_dir: str = "dataset"):
    """
    Filters the raw Food-101 download down to only the classes this project
    supports, and arranges them into `dataset/train/<class>` and
    `dataset/val/<class>` using the official train/test split files.

    food101_dir : path to the extracted food-101 folder
                  (must contain 'images/' and 'meta/train.txt', 'meta/test.txt')
    """
    images_dir = os.path.join(food101_dir, "images")
    meta_dir = os.path.join(food101_dir, "meta")

    def copy_split(split_file, split_name):
        with open(os.path.join(meta_dir, split_file)) as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            class_name, filename = line.split("/")
            if class_name not in supported_classes:
                continue  # skip classes we don't support

            src = os.path.join(images_dir, class_name, f"{filename}.jpg")
            dst_dir = os.path.join(out_dir, split_name, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{filename}.jpg")

            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    print("Copying training images...")
    copy_split("train.txt", "train")
    print("Copying validation images...")
    copy_split("test.txt", "val")
    print(f"Dataset prepared at: {out_dir}/")


def load_datasets(data_dir: str):
    """Loads train/val datasets from an ImageFolder-style directory."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="categorical",
    )

    class_names = train_ds.class_names  # alphabetically sorted by Keras

    # Preprocess for MobileNetV2 (scales pixels to [-1, 1]) + light augmentation
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    def prep_train(x, y):
        x = augmentation(x)
        x = preprocess_input(x)
        return x, y

    def prep_val(x, y):
        x = preprocess_input(x)
        return x, y

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int):
    """Builds a MobileNetV2-based transfer learning classifier."""
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for the initial training phase

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history_1, history_2=None, out_path="training_history.png"):
    """Saves accuracy/loss curves. history_2 is the optional fine-tuning phase."""
    acc = history_1.history["accuracy"]
    val_acc = history_1.history["val_accuracy"]
    loss = history_1.history["loss"]
    val_loss = history_1.history["val_loss"]

    if history_2 is not None:
        acc += history_2.history["accuracy"]
        val_acc += history_2.history["val_accuracy"]
        loss += history_2.history["loss"]
        val_loss += history_2.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")

    plt.savefig(out_path)
    print(f"Saved training curves to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Food Nutrition Analyzer CNN.")
    parser.add_argument("--data_dir", type=str, default="dataset",
                         help="Path to dataset/{train,val} folders")
    parser.add_argument("--epochs", type=int, default=15,
                         help="Epochs for the initial (frozen base) training phase")
    parser.add_argument("--fine_tune_epochs", type=int, default=5,
                         help="Epochs for the fine-tuning phase (unfrozen top layers)")
    parser.add_argument("--fine_tune_at", type=int, default=100,
                         help="Layer index in MobileNetV2 from which to unfreeze for fine-tuning")
    parser.add_argument("--prepare", action="store_true",
                         help="Prepare dataset/ folder from a raw Food-101 download first")
    parser.add_argument("--food101_dir", type=str, default=None,
                         help="Path to the extracted food-101 folder (required with --prepare)")
    parser.add_argument("--model_out", type=str, default="food_model.keras")
    args = parser.parse_args()

    # Load supported class list from nutrition.csv (single source of truth)
    import pandas as pd
    nutrition_df = pd.read_csv("nutrition.csv")
    supported_classes = sorted(nutrition_df["Food"].tolist())

    if args.prepare:
        if not args.food101_dir:
            raise ValueError("--food101_dir is required when using --prepare")
        build_dataset_from_food101(args.food101_dir, supported_classes, args.data_dir)

    print("Loading datasets...")
    train_ds, val_ds, class_names = load_datasets(args.data_dir)
    print(f"Found {len(class_names)} classes.")

    # Sanity check: model classes should match nutrition.csv entries
    missing = set(class_names) - set(supported_classes)
    if missing:
        print(f"WARNING: these classes are not present in nutrition.csv: {missing}")

    print("Building model (MobileNetV2 backbone, frozen)...")
    model, base_model = build_model(num_classes=len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    print("\n=== Phase 1: training classification head (base frozen) ===")
    history_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
    base_model.trainable = True
    # Freeze all layers before `fine_tune_at`, only fine-tune the later ones
    for layer in base_model.layers[:args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # low LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks,
    )

    # Save artifacts
    model.save(args.model_out)
    print(f"Saved trained model to {args.model_out}")

    with open("class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print("Saved class_names.json")

    plot_history(history_1, history_2)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal validation accuracy: {val_acc:.4f} | validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
