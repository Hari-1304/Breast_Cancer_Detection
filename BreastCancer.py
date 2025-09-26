import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, Dropout,
                                     Reshape, LSTM, Bidirectional)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Dataset directory
data_dir = '/Users/mac/Documents/breast-cancer-organized'

# Load dataset
def load_dataset(data_dir, img_size=(50, 50)):
    X, y = [], []
    for label_name in ['class_0', 'class_1']:
        label = int(label_name.split('_')[1])
        class_dir = os.path.join(data_dir, label_name)
        for file in os.listdir(class_dir):
            if file.startswith('.'):
                continue
            try:
                img_path = os.path.join(class_dir, file)
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"Skipped {img_path}: {e}")
    print(f"✅ Loaded {len(X)} images.")
    return np.array(X), np.array(y)

X, y = load_dataset(data_dir)
if len(X) == 0:
    raise ValueError("🚨 No images loaded. Check if your dataset has 'class_0' and 'class_1' folders with valid images.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# Define model
inputs = Input(shape=(50, 50, 3))
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Reshape((x.shape[1]*x.shape[2], x.shape[3]))(x)
x = Bidirectional(LSTM(64))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy:.4f}")

# Predict
def classify_image(image_array, model):
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)[0]
    class_idx = np.argmax(prediction)
    confidence = round(prediction[class_idx] * 100, 2)
    label = "Malignant" if class_idx == 1 else "Benign"
    print(f"Prediction: {label} ({confidence}%)")
    return label, confidence

sample_img = X_test[0]
classify_image(sample_img, model)

plt.imshow(sample_img)
plt.title("Sample Test Image")
plt.axis('off')
plt.show()

# Training history
plt.plot(history.history['accuracy'], label='Training Acc')
plt.plot(history.history['val_accuracy'], label='Validation Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()
