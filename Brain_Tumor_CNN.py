"""Train and use a reproducible, explicitly-labelled MRI classifier.

This is an educational baseline, not a clinical decision-support system.  It
accepts either ordinary 2-D images arranged in one directory per class or 3-D
NIfTI files referenced by a reviewed ``path,label`` CSV manifest.
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
    """Return whether *path* has a supported NIfTI extension."""
    return path.name.lower().endswith(NIFTI_EXTENSIONS)


def collect_image_samples(dataset_path: Path) -> tuple[list[Path], list[str]]:
    """Collect supported images recursively from immediate class directories."""
    classes = sorted(item.name for item in dataset_path.iterdir() if item.is_dir())
    if len(classes) < 2:
        raise ValueError("Image mode needs at least two class subdirectories.")
    paths: list[Path] = []
    labels: list[str] = []
    empty_classes: list[str] = []
    for class_name in classes:
        files = sorted(
            item for item in (dataset_path / class_name).rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            empty_classes.append(class_name)
        paths.extend(files)
        labels.extend([class_name] * len(files))
    if empty_classes:
        raise ValueError(f"No supported images in class folders: {', '.join(empty_classes)}")
    return paths, labels


def collect_nifti_samples(dataset_path: Path, manifest_path: Path) -> tuple[list[Path], list[str]]:
    """Read reviewed NIfTI paths and labels from a ``path,label`` CSV."""
    paths: list[Path] = []
    labels: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"path", "label"}.issubset(reader.fieldnames):
            raise ValueError("Label manifest must have 'path' and 'label' columns.")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            label = (row.get("label") or "").strip()
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else dataset_path / candidate
            if not raw_path or not label:
                raise ValueError(f"Empty path or label in manifest row {row_number}.")
            if not resolved.is_file() or not is_nifti_file(resolved):
                raise FileNotFoundError(f"Invalid NIfTI path in manifest row {row_number}: {resolved}")
            paths.append(resolved)
            labels.append(label)
    return paths, labels


def validate_samples(paths: Iterable[Path], labels: list[str], validation_size: float) -> list[str]:
    """Validate a labelled dataset can support a stratified holdout split."""
    paths = list(paths)
    if len(paths) != len(labels) or not paths:
        raise ValueError("Each usable sample must have exactly one label.")
    if not 0 < validation_size < 1:
        raise ValueError("Validation size must be between 0 and 1.")
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("Classification needs at least two non-empty classes.")
    too_small = [label for label, count in counts.items() if count < 2]
    if too_small:
        raise ValueError(f"Each class needs at least two samples; too small: {too_small}")
    validation_count = math.ceil(len(labels) * validation_size)
    if validation_count < len(counts) or len(labels) - validation_count < len(counts):
        raise ValueError("The validation split is too small to contain every class in both splits.")
    return sorted(counts)


def read_patient_ids(manifest_path: Path, paths: list[Path], dataset_path: Path) -> list[str]:
    """Map every sample to a patient ID from a ``path,patient_id`` CSV."""
    mapping: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"path", "patient_id"}.issubset(reader.fieldnames):
            raise ValueError("Patient manifest must have 'path' and 'patient_id' columns.")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            patient_id = (row.get("patient_id") or "").strip()
            if not raw_path or not patient_id:
                raise ValueError(f"Empty path or patient_id in patient manifest row {row_number}.")
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else dataset_path / candidate
            key = str(resolved.resolve())
            if key in mapping:
                raise ValueError(f"Duplicate sample path in patient manifest row {row_number}: {resolved}")
            mapping[key] = patient_id
    missing = [str(item) for item in paths if str(item.resolve()) not in mapping]
    if missing:
        raise ValueError(f"Patient manifest is missing {len(missing)} sample paths; first: {missing[0]}")
    return [mapping[str(item.resolve())] for item in paths]


def load_nifti_slice(path: Path, image_size: int, nib, cv2, np):
    """Select the most foreground axial slice and robustly normalize it."""
    volume = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {volume.shape}: {path}")
    if not np.isfinite(volume).all():
        raise ValueError(f"NIfTI volume contains NaN or infinity: {path}")
    foreground = np.abs(volume) > np.percentile(np.abs(volume), 20)
    image = volume[:, :, int(np.argmax(foreground.sum(axis=(0, 1))))]
    low, high = np.percentile(image, (1, 99))
    if not np.isfinite(low) or high - low < 1e-6:
        raise ValueError(f"NIfTI volume is empty or nearly constant: {path}")
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.repeat(image[..., np.newaxis], 3, axis=-1).astype(np.float32)


def load_image(path: Path, image_size: int, tf):
    """Load a 2-D image as a normalized RGB tensor."""
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    image = tf.keras.utils.load_img(path, target_size=(image_size, image_size), color_mode="rgb")
    return tf.keras.utils.img_to_array(image).astype("float32") / 255.0


def build_model(image_size: int, class_count: int, weights: str, tf, learning_rate: float):
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_top_layers(model, count: int) -> None:
    """Unfreeze the requested final ResNet layers while keeping batch norm frozen."""
    if count <= 0:
        return
    base = next((layer for layer in model.layers if layer.name.startswith("resnet50")), None)
    if base is None or count > len(base.layers):
        raise ValueError("--fine-tune-layers exceeds the ResNet feature extractor size.")
    base.trainable = True
    for layer in base.layers[:-count]:
        layer.trainable = False
    for layer in base.layers[-count:]:
        layer.trainable = layer.__class__.__name__ != "BatchNormalization"


def import_ml_dependencies():
    try:
        import cv2
        import matplotlib.pyplot as plt
        import nibabel as nib
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    except ImportError as error:
        raise SystemExit("Missing dependency. Create .venv and run: python -m pip install -r requirements.txt") from error
    return cv2, plt, nib, np, tf, classification_report, confusion_matrix, StratifiedGroupKFold, train_test_split


def split_samples(paths, encoded_labels, patient_ids, validation_size, seed, class_count, group_splitter, random_splitter):
    """Create an image-level or leakage-safe patient-level validation split."""
    if patient_ids is None:
        return random_splitter(paths, encoded_labels, test_size=validation_size, random_state=seed, stratify=encoded_labels)
    group_count = len(set(patient_ids))
    if group_count < 2:
        raise ValueError("Patient-level splitting needs at least two distinct patient IDs.")
    splitter = group_splitter(n_splits=min(5, group_count), shuffle=True, random_state=seed)
    try:
        candidates = [
            (train, validation) for train, validation in splitter.split(paths, encoded_labels, groups=patient_ids)
            if len({encoded_labels[i] for i in train}) == class_count
            and len({encoded_labels[i] for i in validation}) == class_count
        ]
    except ValueError as error:
        raise ValueError(
            "Could not make a patient-level split; each class needs enough distinct patients. "
            "Add patients or review labels."
        ) from error
    if not candidates:
        raise ValueError("Could not make a patient-level split containing every class; add patients or review labels.")
    train_indices, validation_indices = min(candidates, key=lambda split: abs(len(split[1]) / len(paths) - validation_size))
    return ([paths[i] for i in train_indices], [paths[i] for i in validation_indices],
            [encoded_labels[i] for i in train_indices], [encoded_labels[i] for i in validation_indices])


def train_classifier(args: argparse.Namespace) -> dict:
    """Train, evaluate, and save the model/artifacts described in ``args``."""
    if not args.dataset_path.is_dir():
        raise ValueError(f"Dataset directory does not exist: {args.dataset_path}")
    if args.image_size < 32 or args.batch_size < 1 or args.epochs < 1:
        raise ValueError("--image-size must be at least 32; --batch-size and --epochs must be positive.")
    if args.learning_rate <= 0 or args.fine_tune_learning_rate <= 0 or args.fine_tune_layers < 0:
        raise ValueError("Learning rates must be positive and --fine-tune-layers cannot be negative.")
    if args.data_mode == "nifti":
        if not args.labels_csv or not args.labels_csv.is_file():
            raise ValueError("NIfTI mode requires an existing --labels-csv file.")
        paths, labels = collect_nifti_samples(args.dataset_path, args.labels_csv)
    else:
        paths, labels = collect_image_samples(args.dataset_path)
    class_names = validate_samples(paths, labels, args.validation_size)
    if args.patient_csv and not args.patient_csv.is_file():
        raise ValueError(f"Patient manifest does not exist: {args.patient_csv}")
    patient_ids = read_patient_ids(args.patient_csv, paths, args.dataset_path) if args.patient_csv else None
    class_to_index = {name: index for index, name in enumerate(class_names)}
    encoded_labels = [class_to_index[label] for label in labels]
    cv2, plt, nib, np, tf, classification_report, confusion_matrix, group_splitter, random_splitter = import_ml_dependencies()
    random.seed(args.seed); np.random.seed(args.seed); tf.keras.utils.set_random_seed(args.seed)
    train_paths, validation_paths, y_train, y_validation = split_samples(
        paths, encoded_labels, patient_ids, args.validation_size, args.seed, len(class_names), group_splitter, random_splitter)
    loader = (lambda item: load_image(item, args.image_size, tf) if args.data_mode == "image"
              else load_nifti_slice(item, args.image_size, nib, cv2, np))
    try:
        x_train = np.stack([loader(item) for item in train_paths])
        x_validation = np.stack([loader(item) for item in validation_paths])
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load or prepare an MRI sample: {error}") from error
    y_train, y_validation = np.asarray(y_train), np.asarray(y_validation)
    train_counts = Counter(y_train.tolist())
    class_weights = {index: len(y_train) / (len(class_names) * train_counts[index]) for index in range(len(class_names))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plots").mkdir(exist_ok=True)
    label_map = {"class_to_index": class_to_index, "data_mode": args.data_mode, "image_size": args.image_size}
    (args.output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    tf.keras.backend.clear_session()
    model = build_model(args.image_size, len(class_names), args.weights, tf, args.learning_rate)
    best_model = args.output_dir / "best_model.keras"
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(best_model, monitor="val_loss", save_best_only=True)]
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,
                        batch_size=args.batch_size, callbacks=callbacks, class_weight=class_weights)
    if args.fine_tune_layers:
        unfreeze_top_layers(model, args.fine_tune_layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        fine_history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation),
                                 epochs=max(1, args.epochs // 2), batch_size=args.batch_size, callbacks=callbacks, class_weight=class_weights)
        for metric, values in fine_history.history.items():
            history.history[metric] = history.history.get(metric, []) + values
    model = tf.keras.models.load_model(best_model)
    predictions = np.argmax(model.predict(x_validation, verbose=0), axis=1)
    report = classification_report(y_validation, predictions, target_names=class_names, zero_division=0, output_dict=True)
    evaluation = {"classes": class_names, "class_counts": dict(Counter(labels)), "train_class_weights": class_weights,
                  "confusion_matrix": confusion_matrix(y_validation, predictions).tolist(), "classification_report": report,
                  "validation_accuracy": float((predictions == y_validation).mean()),
                  "validation_recall_by_class": {name: report[name]["recall"] for name in class_names},
                  "split_unit": "patient" if patient_ids else "image"}
    (args.output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    model.save(args.output_dir / "final_model.keras")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("accuracy", "loss")):
        axis.plot(history.history[metric], label="train"); axis.plot(history.history[f"val_{metric}"], label="validation")
        axis.set_title(metric.capitalize()); axis.grid(True); axis.legend()
    figure.tight_layout(); figure.savefig(args.output_dir / "plots" / "training_history.png", dpi=150); plt.close(figure)
    return evaluation


def predict_image(image_path: Path, model_path: Path, label_map_path: Path, image_size: int | None = None) -> dict:
    """Return scores for one 2-D image using matching model artifacts."""
    import numpy as np
    import tensorflow as tf
    image_path, model_path, label_map_path = Path(image_path), Path(model_path), Path(label_map_path)
    if not label_map_path.is_file():
        raise FileNotFoundError(f"Label map does not exist: {label_map_path}")
    metadata = json.loads(label_map_path.read_text(encoding="utf-8"))
    if metadata.get("data_mode", "image") != "image":
        raise ValueError("This model was trained in NIfTI mode; use a matching volume-inference pipeline.")
    resolved_size = image_size or int(metadata.get("image_size", 224))
    index_to_class = {int(index): name for name, index in metadata.get("class_to_index", {}).items()}
    if sorted(index_to_class) != list(range(len(index_to_class))):
        raise ValueError("The label map must contain consecutive class indices starting at zero.")
    model = tf.keras.models.load_model(model_path)
    scores = model.predict(np.expand_dims(load_image(image_path, resolved_size, tf), axis=0), verbose=0)[0]
    if len(scores) != len(index_to_class):
        raise ValueError("Model output count does not match the label map.")
    return {"prediction": index_to_class[int(np.argmax(scores))], "confidence": float(np.max(scores)),
            "scores": {index_to_class[index]: float(score) for index, score in enumerate(scores)}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a labelled ResNet50 MRI classifier or score one 2-D image.")
    parser.add_argument("--dataset-path", type=Path, help="Dataset root; required for training.")
    parser.add_argument("--data-mode", choices=("image", "nifti"), default="image")
    parser.add_argument("--labels-csv", type=Path, help="NIfTI labels CSV with path,label columns.")
    parser.add_argument("--patient-csv", type=Path, help="Optional CSV with path,patient_id columns.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int,
                        help="Training image size. Prediction uses the size stored in label_map.json unless set.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--fine-tune-layers", type=int, default=0)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--predict-image", type=Path, help="Score one 2-D image instead of training.")
    parser.add_argument("--model-path", type=Path); parser.add_argument("--label-map", type=Path)
    args = parser.parse_args()
    if args.predict_image:
        args.model_path = args.model_path or args.output_dir / "best_model.keras"
        args.label_map = args.label_map or args.output_dir / "label_map.json"
    elif not args.dataset_path:
        parser.error("--dataset-path is required unless --predict-image is used.")
    else:
        args.image_size = args.image_size or 224
    return args


def main() -> None:
    args = parse_args()
    try:
        result = (predict_image(args.predict_image, args.model_path, args.label_map, args.image_size) if args.predict_image
                  else train_classifier(args))
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
