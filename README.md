# MRI Classification with ResNet50

An executable baseline for **labelled** brain-MRI classification. It trains a
frozen ResNet50 classifier, creates a stratified validation split, saves the
best/final model, metrics, and training plot in an output directory.


> This is a research/education project, not a clinical diagnostic tool. Never
> use its predictions for patient care without appropriate validation,
> governance, and regulatory review.

> **Status:** 🚧 In Active Development

## Important data note

BraTS 2020 is a tumour dataset: subject directory names such as
`BraTS20_Training_001` do not identify HGG/LGG, and the dataset does not supply
healthy controls. A binary "tumour/no tumour" model cannot be trained from
those folder names. For classification, supply an auditable label manifest; for
BraTS tumour masks, build a segmentation workflow instead.

## Local setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No package is installed outside `.venv`. Python 3.12 is the tested local
interpreter for this project; Google Colab supplies its own compatible runtime.

## Train from ordinary images

Use one immediate folder per class. Class names are taken from folder names.

```text
data/
  no_tumor/scan_001.png
  tumor/scan_002.png
```

```bash
python Brain_Tumor_CNN.py --data-mode image --dataset-path data --epochs 20
```

## Train from NIfTI

Use a CSV with `path,label` columns. Relative paths are resolved from
`--dataset-path`; `.nii` and `.nii.gz` are supported. Every class requires at
least two samples, and each NIfTI must be a 3D volume.

```csv
path,label
BraTS20_Training_001/BraTS20_Training_001_flair.nii.gz,grade_a
BraTS20_Training_002/BraTS20_Training_002_flair.nii.gz,grade_b
```

```bash
python Brain_Tumor_CNN.py --data-mode nifti --dataset-path data --labels-csv data/labels.csv
```

The NIfTI loader selects a high-foreground axial slice and uses robust 1st–99th
percentile normalization. This is deliberately simple; a clinically meaningful
3D model should preserve the volume and use patient-level external validation.

Each class needs at least two files. The validation set must also be large
enough to contain every class; the script stops with a clear message if a
stratified split would be invalid. During training, automatic class weights
reduce the effect of class imbalance.

## Google Colab: complete training guide

1. Open [the Colab notebook](https://colab.research.google.com/github/Arshia-Moghbel/Processing-MRI-pictures-with-M.L./blob/main/brain_tumor_classifier_colab.ipynb).
   In Colab, select **Runtime → Change runtime type → T4 GPU** (if available),
   then choose **Runtime → Run all**. The first cell mounts Google Drive and
   downloads this repository to the temporary Colab machine.
2. In Google Drive create `MyDrive/mri-data`. For 2-D images, use this exact
   structure (class names are your labels):

   ```text
   MyDrive/mri-data/
     no_tumor/scan_001.png
     no_tumor/scan_002.png
     tumor/scan_001.png
     tumor/scan_002.png
   ```

3. In the notebook configuration cell, choose `DATA_MODE = "image"`, then run
   its training cell. Start with `BATCH_SIZE = 4`, `EPOCHS = 12`, and
   `FINE_TUNE_LAYERS = 0`. The notebook prints the GPU it sees before training.
4. For NIfTI data, set `DATA_MODE = "nifti"` and create
   `MyDrive/mri-data/labels.csv` using paths relative to `mri-data`:

   ```csv
   path,label
   subject_001/subject_001_flair.nii.gz,grade_a
   subject_002/subject_002_flair.nii.gz,grade_b
   ```

   The sample shown is only a format example: each class needs at least two
   labelled volumes and the labels must be independently reviewed.
5. If there is more than one image/volume per patient, create
   `MyDrive/mri-data/patient_ids.csv`, set `PATIENT_CSV` in the notebook, and
   use this format:

   ```csv
   path,patient_id
   no_tumor/scan_001.png,patient_001
   no_tumor/scan_002.png,patient_001
   ```

   Do not omit this file when samples from a patient repeat: otherwise the
   reported validation score is not patient-independent.
6. Results persist in `MyDrive/mri-artifacts/`: `best_model.keras`,
   `final_model.keras`, `label_map.json`, `evaluation.json`, and
   `plots/training_history.png`. To improve a baseline only after it completes
   without memory errors, set `FINE_TUNE_LAYERS = 20` and rerun the training
   cell. This starts a new run in the configured output folder; use a new
   `OUTPUT_DIR` if you want to preserve the earlier run.

The notebook invokes `Brain_Tumor_CNN.py`; model logic exists in one place.
It never derives labels from BraTS folder names.

For multiple images or slices from the same patient, pass `--patient-csv` with
`path,patient_id` columns. This keeps each patient entirely in either training
or validation and avoids an overly optimistic score caused by slice leakage.
Without patient IDs, the default split is at image/volume level and must not be
interpreted as patient-independent evaluation.

## Outputs

By default, files are written to `artifacts/`:

* `best_model.keras` and `final_model.keras`
* `label_map.json` — the exact class-to-output-index mapping
* `evaluation.json` — class counts, class weights, confusion matrix, per-class metrics, accuracy
* `plots/training_history.png`

Use `--output-dir results/run-01` to keep runs separate. Set `--weights none`
when working offline (ImageNet weights otherwise download on first use).

To make the patient-level split from the command line, add
`--patient-csv data/patient_ids.csv`. The paths inside that CSV are relative to
`--dataset-path`, just like the NIfTI label manifest.

Predict one ordinary image using artifacts from the same run:

```bash
python Brain_Tumor_CNN.py --predict-image data/example.png \
  --model-path artifacts/best_model.keras --label-map artifacts/label_map.json
```

This returns the model's class scores, not a tumor location or a clinical
diagnosis. A classifier trained on 2D images cannot locate a tumor; use a
segmentation model and reviewed masks for localization.

## Checks

```bash
python -m unittest discover -s tests -v
python -m py_compile Brain_Tumor_CNN.py
```

## Roadmap

Completed work is checked only when it is present in the repository and covered
by the automated checks where practical.

- [x] Restore the documented command-line training program.
- [x] Require explicit labels; do not derive a diagnosis from BraTS folder IDs.
- [x] Support labelled 2-D images and labelled 3-D NIfTI volumes.
- [x] Validate empty classes, malformed manifests, invalid volumes, and
  impossible stratified splits before training.
- [x] Add optional patient-level splits to prevent slice leakage.
- [x] Save model, label map, evaluation metrics, and training-history plot.
- [x] Add optional top-layer fine tuning after the frozen baseline.
- [x] Add reproducible single-image inference with class scores.
- [x] Keep a Colab notebook aligned with the supported data layouts.
- [x] Add automated regression tests for validation, split safety, and inference helpers.

Out of scope: clinical validation, tumour localisation/segmentation, and a
claim of diagnostic suitability. Those require reviewed datasets, an appropriate
3-D workflow, external validation, and clinical/regulatory governance.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![ResNet50](https://img.shields.io/badge/Model-ResNet50-red)
![License](https://img.shields.io/badge/License-GPL--3.0-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-green)
