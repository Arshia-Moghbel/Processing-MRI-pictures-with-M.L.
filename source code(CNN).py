import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import datetime

# 📁 مسیر دیتاست (حتماً آن را به مسیر واقعی خود تغییر دهید)
DATASET_PATH = "/content/brats2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
use_nifti = True  # تغییر این متغیر برای انتخاب نوع داده

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

if use_nifti:
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

else:
    # بارگذاری داده‌ها از دایرکتوری‌های تصویری
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(DATASET_PATH, target_size=(128, 128),
                                            class_mode='binary', batch_size=4)

    for _ in range(len(generator)):
        img, label = generator.next()
        X.extend(img)
        y.extend(label)

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

# 📈 تنظیمات برای EarlyStopping و ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

# 🏋️ آموزش مدل با TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    X_train, y_train,
    epochs=10,  # می‌توان تغییر داد
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint, tensorboard_callback]
)

# 💾 ذخیره مدل به صورت SavedModel و TFLite
model.save("BraTS_CNN_Model_Optimized", save_format='tf')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("BraTS_CNN_Model_Optimized.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Model saved as 'BraTS_CNN_Model_Optimized' and 'BraTS_CNN_Model_Optimized.tflite'")

# 📊 نمایش دقت و Loss
plt.figure(figsize=(12, 4))

# دقت
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.show()

# 📊 نمایش confusion matrix و classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# 💾 ذخیره نتایج ارزیابی در فایل متنی
with open("evaluation_results.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf_matrix))
    f.write("\nClassification Report:\n")
    f.write(class_report)

# 📉 ارزیابی نهایی مدل روی داده تست
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Final Test Accuracy: {accuracy*100:.2f}%")

# 📝 ذخیره مجدد گزارش ارزیابی با دقت نهایی
with open("evaluation_results.txt", "a") as f:
    f.write(f"\nFinal Test Accuracy: {accuracy*100:.2f}%\n")

# 🕒 ذخیره مدل نسخه‌گذاری شده با زمان
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f"saved_model/brain_tumor_classifier_{timestamp}")
print(f"✅ Time-stamped model saved to saved_model/brain_tumor_classifier_{timestamp}")
