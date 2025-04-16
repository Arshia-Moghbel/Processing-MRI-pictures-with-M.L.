# نصب و تنظیم Kaggle API
!pip install kaggle
from google.colab import files
files.upload()  # فایل kaggle.json را اینجا آپلود کن

# تنظیم دسترسی به Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# دانلود دیتاست BraTS از Kaggle
!kaggle datasets download -d awsaf49/brats20-dataset-training-validation
!unzip brats20-dataset-training-validation.zip -d BraTS/

# نصب کتابخانه‌های موردنیاز
!pip install nibabel tensorflow opencv-python matplotlib

import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import zipfile

# مسیر دیتاست
DATASET_PATH = "/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# تابع خواندن و پردازش تصاویر MRI
def load_mri_image(file_path):
    img = nib.load(file_path).get_fdata()
    img = np.mean(img, axis=-1)  # تبدیل به تصویر سیاه و سفید
    img = cv2.resize(img, (224, 224))

    if np.max(img) != 0:
        img = img / np.max(img)

    img = np.repeat(img[..., np.newaxis], 3, axis=-1)
    return img

# بارگذاری داده‌ها
X, y = [], []
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    t1_path = os.path.join(folder_path, f"{folder}_t1.nii")  # تغییر نام فایل‌ها
    flair_path = os.path.join(folder_path, f"{folder}_flair.nii")

    print(f"Checking: {t1_path}")  
    print(f"Checking: {flair_path}")

    if os.path.exists(t1_path) and os.path.exists(flair_path):
        img_t1 = load_mri_image(t1_path)
        img_flair = load_mri_image(flair_path)
        img_combined = (img_t1 + img_flair) / 2  # ترکیب دو تصویر MRI

        X.append(img_combined)
        y.append(1 if "HGG" in folder else 0)

X = np.array(X).reshape(-1, 224, 224, 3)
y = np.array(y)

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ X_test and y_test saved successfully!")

# طراحی مدل CNN با ResNet-50
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# فریز کردن لایه‌های اولیه
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# نمایش معماری مدل
model.summary()

# آموزش مدل
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=4)

# ذخیره مدل
model.save("BraTS_CNN_Model_Optimized.keras")

# نمایش نمودار دقت مدل
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()
