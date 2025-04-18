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

# 📁 مسیر دیتاست (حتماً آن را به مسیر واقعی خود تغییر دهید)
DATASET_PATH = "/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# 📥 تابع خواندن و پردازش MRI
def load_mri_image(file_path, target_size=(128, 128)):
    img = nib.load(file_path).get_fdata()
    img = np.mean(img, axis=-1)  # تبدیل به تصویر تک‌کاناله (Grayscale)
    img = cv2.resize(img, target_size)

    if np.max(img) != 0:
        img = img / np.max(img)

    img = np.repeat(img[..., np.newaxis], 3, axis=-1)  # ۳ کاناله کردن تصویر
    return img.astype(np.float32)

# 🧠 بارگذاری و پردازش داده‌ها
X, y = [], []

for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    t1_path = os.path.join(folder_path, f"{folder}_t1.nii")
    flair_path = os.path.join(folder_path, f"{folder}_flair.nii")

    if os.path.exists(t1_path) and os.path.exists(flair_path):
        img_t1 = load_mri_image(t1_path)
        img_flair = load_mri_image(flair_path)
        img_combined = (img_t1 + img_flair) / 2

        X.append(img_combined)
        y.append(1 if "HGG" in folder else 0)

X = np.array(X)
y = np.array(y)

# ✂️ تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 💾 ذخیره داده‌های تست (در صورت نیاز برای تحلیل بیشتر)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)
print("✅ X_test and y_test saved successfully!")

# 🧩 ساخت مدل با ResNet50 (ورودی 128×128)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# ❄️ فریز کردن لایه‌های اولیه
for layer in base_model.layers:
    layer.trainable = False

# 🔧 اضافه کردن لایه‌های Dense بالای مدل پایه
x = Flatten()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# 🏋️ آموزش مدل
history = model.fit(
    X_train, y_train,
    epochs=10,  # می‌توان تغییر داد
    batch_size=4,
    validation_data=(X_test, y_test)
)

# 💾 ذخیره مدل
model.save("BraTS_CNN_Model_Optimized.keras")
print("✅ Model saved as 'BraTS_CNN_Model_Optimized.keras'")

# 📊 نمایش دقت
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
