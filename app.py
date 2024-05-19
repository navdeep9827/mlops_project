from flask import Flask, request, jsonify
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
from model2 import ResNet9, to_device, get_default_device

app = Flask(__name__)

# Set up device
device = get_default_device()

# Define the number of classes
num_classes = 35  # 26 letters + 9 digits

# Mapping indices to letters and digits
index_to_char = {i: chr(65 + i) for i in range(26)}  # A-Z
index_to_char.update({26 + i: str(i + 1) for i in range(9)})  # 1-9

# Load the model and map it to the appropriate device
model = ResNet9(3, num_classes)  # Initialize the model structure
model.load_state_dict(torch.load('ISN-2-custom-resnet.pth', map_location=device))
model = to_device(model, device)

def preprocess_image(image_data):
    # Decoding the base64 image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    # Converting to RGB format
    image = image.convert('RGB')
    # Resizing and preprocess as required by your model
    image = image.resize((224, 224))  # Example size
    image = np.array(image) / 255.0  # Normalize if required
    image = np.transpose(image, (2, 0, 1))  # Convert to channels first
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
    image = to_device(image, device)  # Move to device
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['data']
    image = preprocess_image(image_data)
    prediction = model(image)
    _, preds = torch.max(prediction, dim=1)
    pred_char = index_to_char[preds.item()]  # Map the prediction to the corresponding character
    return jsonify({'prediction': pred_char})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
