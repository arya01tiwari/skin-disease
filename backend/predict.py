from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the fine-tuned MobileNetV2 model
MODEL_PATH = "model/mobilenetv2_finetuned.h5"
model = load_model(MODEL_PATH)

# Assuming you have your dataset folders structured like: dataset/train/acne, dataset/train/eczema, etc.
# Get class names from the training folder used during training
CLASS_NAMES = sorted(os.listdir("dataset/train"))  # Modify this if your folder is named differently

def prepare_image(img):
    """Preprocess image to match training input requirements."""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    processed_img = prepare_image(img)
    predictions = model.predict(processed_img)

    predicted_class_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    predicted_class = CLASS_NAMES[predicted_class_index]

    result = {
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
