"""Train a reproducible MRI image classifier.

This program supports two *explicitly labelled* data layouts:

* ``image``: one directory per class, for example ``data/no_tumor/*.png`` and
  ``data/tumor/*.png``.
* ``nifti``: NIfTI files plus a CSV manifest with ``path,label`` columns.

BraTS 2020 contains tumour cases only and its subject IDs do not encode HGG/LGG.
It must therefore be used with a label manifest (or for a segmentation task),
not by inferring labels from file/folder names.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
NIFTI_EXTENSIONS = (".nii", ".nii.gz")


def is_nifti_file(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_EXTENSIONS)


def collect_image_samples(dataset_path: Path) -> tuple[list[Path], list[str]]:
    """Return image files and labels from immediate class subdirectories."""
    classes = sorted(item.name for item in dataset_path.iterdir() if item.is_dir())
    if len(classes) < 2:
        raise ValueError("Image mode needs at least two class subdirectories.")
    paths: list[Path] = []
    labels: list[str] = []
    for class_name in classes:
        files = sorted(
            path for path in (dataset_path / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(files)
        labels.extend([class_name] * len(files))
    return paths, labels


def collect_nifti_samples(dataset_path: Path, manifest_path: Path) -> tuple[list[Path], list[str]]:
    """Read labels from a CSV manifest; paths may be absolute or data-relative."""
    paths: list[Path] = []
    labels: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
            raise ValueError("Label manifest must have 'path' and 'label' columns.")
        for row_number, row in enumerate(reader, start=2):
            raw_path = Path(row["path"].strip())
            path = raw_path if raw_path.is_absolute() else dataset_path / raw_path
            label = row["label"].strip()
            if not label:
                raise ValueError(f"Empty label in manifest row {row_number}.")
            if not path.is_file() or not is_nifti_file(path):
                raise FileNotFoundError(f"Invalid NIfTI path in manifest row {row_number}: {path}")
            paths.append(path)
            labels.append(label)
    return paths, labels


def validate_samples(paths: Iterable[Path], labels: list[str], validation_size: float) -> list[str]:
    paths = list(paths)
    if not paths:
        raise ValueError("No usable samples were found.")
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("Classification needs at least two non-empty classes.")
    too_small = [label for label, count in counts.items() if count < 2]
    if too_small:
        raise ValueError(f"Each class needs at least two samples; too small: {too_small}")
    test_count = math.ceil(len(labels) * validation_size)
    train_count = len(labels) - test_count
    if test_count < len(counts) or train_count < len(counts):
        raise ValueError(
            "The validation split is too small for a stratified split. Add data or increase "
            "--validation-size so both train and validation contain every class."
        )
    return sorted(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet50 classifier for labelled MRI data.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--data-mode", choices=("image", "nifti"), required=True)
    parser.add_argument("--labels-csv", type=Path, help="Required in nifti mode; CSV columns: path,label")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    return parser.parse_args()


def import_ml_dependencies():
    try:
        import cv2
        import matplotlib.pyplot as plt
        import nibabel as nib
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
    except ImportError as error:
        raise SystemExit("Missing dependency. Install with: pip install -r requirements.txt") from error
    return cv2, plt, nib, np, tf, classification_report, confusion_matrix, train_test_split


def load_nifti_slice(path: Path, image_size: int, nib, cv2, np):
    volume = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32)).squeeze()
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {volume.shape}: {path}")
    if not np.isfinite(volume).all():
        raise ValueError(f"NIfTI volume contains NaN or infinity: {path}")
    # Select the axial slice with the largest non-background area, rather than
    # averaging the whole volume and discarding its spatial structure.
    foreground = volume > np.percentile(volume, 20)
    slice_index = int(np.argmax(foreground.sum(axis=(0, 1))))
    image = volume[:, :, slice_index]
    low, high = np.percentile(image, (1, 99))
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.repeat(image[..., np.newaxis], 3, axis=-1).astype(np.float32)


def load_image(path: Path, image_size: int, tf):
    image = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
    return tf.keras.utils.img_to_array(image).astype("float32") / 255.0


def build_model(image_size: int, class_count: int, weights: str, tf):
    base = tf.keras.applications.ResNet50(
        include_top=False, weights=None if weights == "none" else weights,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    args = parse_args()
    if not args.dataset_path.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {args.dataset_path}")
    if not 0 < args.validation_size < 1:
        raise SystemExit("--validation-size must be between 0 and 1.")
    if args.image_size < 32 or args.batch_size < 1 or args.epochs < 1:
        raise SystemExit("--image-size must be at least 32; --batch-size and --epochs must be positive.")
    if args.data_mode == "nifti" and not args.labels_csv:
        raise SystemExit("--labels-csv is required when --data-mode nifti.")

    if args.data_mode == "image":
        paths, labels = collect_image_samples(args.dataset_path)
    else:
        paths, labels = collect_nifti_samples(args.dataset_path, args.labels_csv)
    try:
        class_names = validate_samples(paths, labels, args.validation_size)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    class_to_index = {name: index for index, name in enumerate(class_names)}
    encoded_labels = [class_to_index[label] for label in labels]

    cv2, plt, nib, np, tf, classification_report, confusion_matrix, train_test_split = import_ml_dependencies()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(exist_ok=True)

    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, encoded_labels, test_size=args.validation_size, random_state=args.seed, stratify=encoded_labels
    )
    loader = (lambda p: load_image(p, args.image_size, tf)) if args.data_mode == "image" else \
             (lambda p: load_nifti_slice(p, args.image_size, nib, cv2, np))
    x_train = np.stack([loader(path) for path in train_paths])
    x_test = np.stack([loader(path) for path in test_paths])
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    train_counts = Counter(y_train.tolist())
    class_weights = {index: len(y_train) / (len(class_names) * train_counts[index]) for index in range(len(class_names))}
    (args.output_dir / "label_map.json").write_text(
        json.dumps({"class_to_index": class_to_index}, indent=2), encoding="utf-8"
    )

    model = build_model(args.image_size, len(class_names), args.weights, tf)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.output_dir / "best_model.keras", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, class_weight=class_weights)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    report = classification_report(y_test, predictions, target_names=class_names, zero_division=0, output_dict=True)
    matrix = confusion_matrix(y_test, predictions).tolist()
    evaluation = {"classes": class_names, "class_counts": dict(Counter(labels)), "train_class_weights": class_weights,
                  "confusion_matrix": matrix,
                  "classification_report": report, "test_accuracy": float((predictions == y_test).mean())}
    (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    model.save(args.output_dir / "final_model.keras")

    plt.figure(figsize=(10, 4))
    for index, metric in enumerate(("accuracy", "loss"), start=1):
        plt.subplot(1, 2, index)
        plt.plot(history.history[metric], label="train")
        plt.plot(history.history[f"val_{metric}"], label="validation")
        plt.title(metric.capitalize()); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(args.output_dir / "plots" / "training_history.png", dpi=150); plt.close()
    print(f"Completed. Test accuracy: {evaluation['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
