from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model(r"C:\Users\Microsoft\Desktop\DL project\Mountain\Semester_end.keras")

# Define class names
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Preprocessing function
def preprocess_image(image_path, image_size=(150, 150)):
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
   
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
   
    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)
   
    # Preprocess the image
    processed_image = preprocess_image(file_path)
   
    # Make a prediction
    predictions = model.predict(processed_image)
    max_confidence = np.max(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
   
    # Set a confidence threshold (e.g., 50%)
    confidence_threshold = 0.5
   
    if max_confidence < confidence_threshold:
        predicted_class = "not a buildings', 'forest', 'glacier', 'mountain', 'sea' or 'street"
   
    # Return the rendered template with the uploaded image and prediction
    return render_template('result.html',
                           image_url=f'/uploads/{file.filename}',
                           predicted_class=predicted_class)
if __name__ == '__main__':
    # Ensure the 'uploads' folder exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=False)
