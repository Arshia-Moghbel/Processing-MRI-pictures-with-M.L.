# MRI Classification with ResNet50

An executable baseline for **labelled** brain-MRI classification. It trains a
frozen ResNet50 classifier, creates a stratified validation split, saves the
best/final model, metrics, and training plot in an output directory.

> This is a research/education project, not a clinical diagnostic tool. Never
> use its predictions for patient care without appropriate validation,
> governance, and regulatory review.

## Important data note

BraTS 2020 is a tumour dataset: subject directory names such as
`BraTS20_Training_001` do not identify HGG/LGG, and the dataset does not supply
healthy controls. A binary "tumour/no tumour" model cannot be trained from
those folder names. For classification, supply an auditable label manifest; for
BraTS tumour masks, build a segmentation workflow instead.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
python Brain_Tumor_CNN.py --data-mode nifti --dataset-path data --labels-csv labels.csv
```

The NIfTI loader selects a high-foreground axial slice and uses robust 1st–99th
percentile normalization. This is deliberately simple; a clinically meaningful
3D model should preserve the volume and use patient-level external validation.

Each class needs at least two files. The validation set must also be large
enough to contain every class; the script stops with a clear message if a
stratified split would be invalid. During training, automatic class weights
reduce the effect of class imbalance.

## Google Colab (recommended for training)

Open `latest_version(for_google_colab).ipynb` in Colab, choose **Runtime →
Change runtime type → T4 GPU** if available, mount Drive, and set `PROJECT` in
the notebook to the uploaded repository folder. Run the setup and checks cell.
Keep the dataset and output directory on Drive so they survive Colab session
restarts. Use one training command from the notebook and edit its paths to your
Drive folders. Start with `--fine-tune-layers 0` and batch size 4 or 8; use the
optional `--fine-tune-layers 20` only after the first run fits in memory.

The current Colab notebook is only a launcher and usage guide; model logic is
maintained solely in `Brain_Tumor_CNN.py`. Don't run a separate old notebook
cell that derives labels from BraTS folder names.

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
