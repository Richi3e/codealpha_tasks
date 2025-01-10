from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = load_model('face_recognition_model.h5')  # Ensure this file exists in the same directory

# Initialize Flask app
app = Flask(__name__)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image is included in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']

        # Read the image from the uploaded file
        img = load_img(file.stream, target_size=(224, 224))  # Use file.stream for in-memory file reading
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        return jsonify({
            'predicted_class': int(predicted_class),
            'confidence_scores': predictions.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a default route
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Face Recognition API! Use the /predict endpoint to classify images.", 200

if __name__ == '__main__':
    app.run(debug=True)
