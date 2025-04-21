# Processing-MRI-pictures-with-M.L.

GitHub README برای پروژه تشخیص تومور مغزی با ResNet50

# Brain Tumor Classification using ResNet50

This project implements a **Convolutional Neural Network (CNN)** for classifying brain tumor images using **ResNet50** architecture. The model leverages **Transfer Learning** and is trained on both MRI images in `.nii` format and standard image datasets. The goal is to distinguish between healthy and tumorous brain tissues.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Data Preprocessing](#data-preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Model Saving and Deployment](#model-saving-and-deployment)
- [📤 Data Access via Google Drive](#data-access-via-google-drive)
- [License](#license)

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
pip install -r requirements.txt

Dependencies:
	•	TensorFlow
	•	Keras
	•	NumPy
	•	Matplotlib
	•	scikit-learn
	•	nibabel
	•	OpenCV
	•	Seaborn

Usage

To train the model, you can use the hybrid_brain_tumor_classifier.py script. Simply choose the input data format by setting the use_nifti variable:
	•	Set use_nifti = True for NIfTI files (.nii)
	•	Set use_nifti = False to use images from directory structure (train, val, test folders)

Example:

python hybrid_brain_tumor_classifier.py

This will start training using the dataset in the specified directory. The model will be saved as brain_tumor_classifier in the saved_model directory and also as a TFLite model (brain_tumor_classifier.tflite) for edge devices.

Model Overview

The model uses the ResNet50 architecture pre-trained on ImageNet, with a custom top layer for binary classification (tumor or no tumor). The base layers of ResNet50 are frozen, and the model is fine-tuned to the brain tumor classification task.

Architecture:
	•	Base Model: ResNet50 (pre-trained)
	•	Top Layer:
	•	Global Average Pooling
	•	Dense Layer (1024 units, ReLU activation)
	•	Output Layer (2 units for binary classification)

Data Preprocessing

The dataset can either be loaded from:
	1.	NIfTI (.nii) files: MRI scans are loaded and preprocessed using the nibabel library.
	2.	Image Files: Data augmentation is applied using ImageDataGenerator for better generalization.

The images are resized to 224x224 pixels, and pixel values are normalized for model input.

Training and Evaluation
	•	The model is trained using categorical cross-entropy loss and accuracy as the metric.
	•	EarlyStopping and ModelCheckpoint callbacks are used for efficient training and to prevent overfitting.
	•	Before execution, ensure the dataset path exists:
	```python
	if not os.path.exists(DATASET_PATH):
	    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
	```
	•	After training, the model is evaluated on a test set, and classification results are saved in a text file.

Results

The model provides:
	•	Classification Report: Precision, Recall, F1-Score, etc.
	•	Confusion Matrix: Visual representation of model performance.
	•	Accuracy Plot: Shows the training and validation accuracy over epochs.
	•	Loss Plot: Shows the training and validation loss over epochs.

Example output:

Test accuracy: 92.5%
Confusion Matrix:
[[200   5]
 [  7 188]]
Classification Report:
              precision    recall  f1-score   support
        0       0.97      0.98      0.97       205
        1       0.97      0.96      0.97       195

## Visualization
  
The following plots are generated during training:
  
![Accuracy Plot](images/accuracy_plot.png)
![Loss Plot](images/loss_plot.png)
  
You can generate them by running the training script, which automatically displays and optionally saves them.

Model Saving and Deployment
	•	The trained model is saved in both TensorFlow SavedModel format (saved_model/brain_tumor_classifier) and TFLite format (brain_tumor_classifier.tflite) for deployment to mobile or edge devices.
	•	You can load the model as follows:

model = tf.keras.models.load_model('saved_model/brain_tumor_classifier')

## 📤 Data Access via Google Drive

If you want to train this model using Google Colab and access the BraTS2020 dataset from your own Google Drive:
  
```python
from google.colab import drive
drive.mount('/content/drive')
DATASET_PATH = "/content/drive/MyDrive/path_to_your_dataset"
```
  
Replace `path_to_your_dataset` with the actual location of your dataset in Google Drive. Ensure proper permissions and structure (either .nii files or image folders) before training.

License

This project is licensed under the MIT License - see the LICENSE file for details.

⸻

📝 Additional Notes:
	•	You can easily deploy this model for mobile or embedded devices using the TFLite model.
	•	The dataset used in this project is the BraTS 2020 dataset, which can be found here.
	•	If using .nii files, ensure you have the necessary libraries installed for loading NIfTI formats. If using directory structure, ensure it follows the train, val, test format with respective images in each folder.
	•	The evaluation results can be found in the `evaluation_results.txt` file after training.
	•	To optimize reading .nii files, you can check for missing files:
	```python
	if not os.path.exists(t1_path) or not os.path.exists(flair_path):
	    print(f"Missing files for {folder}, skipping.")
	    continue
	```
	•	At the end of the script, manage memory with:
	```python
	import gc
	tf.keras.backend.clear_session()
	gc.collect()
	```
	•	Log the completion status:
	```python
	print(f"✅ All tasks completed successfully. Final accuracy: {accuracy*100:.2f}%")
	```

---

### 📄 دستورالعمل‌های تکمیلی:
- **شات‌های پروژه**: می‌توانید تصاویری از نمودارها و گراف‌های دقت و خسارت را در این README قرار دهید تا آن را بیشتر جذاب کنید.
- **لینک‌های مستقیم**: اگر پروژه به اینترنت نیاز دارد (برای دانلود داده‌ها یا استفاده از مدل‌های ذخیره‌شده)، می‌توانید لینک‌ها را در بخش‌های مربوطه اضافه کنید.

این README می‌تواند به عنوان یک **مستندات کامل** برای پروژه شما استفاده شود. اگر نیاز به تغییرات یا افزودن بخش‌های خاصی دارید، بگویید تا ویرایش کنم!