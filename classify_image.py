import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model("breast_cancer_model.h5")
print("✅ Model loaded.")

# Function to preprocess and classify a new image
def classify_new_image(img_path, img_size=(50, 50)):
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    confidence = round(prediction[class_idx] * 100, 2)
    label = "Malignant" if class_idx == 1 else "Benign"

    print(f"🔍 Prediction: {label} ({confidence}%)")
    return label, confidence

# Example usage
img_path = "/Users/mac/Mini_Project/Model_script.py/sample_img2.png"  # Replace with actual path
classify_new_image(img_path)
