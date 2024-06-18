import os
from flask import Flask, request, render_template, redirect
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the Keras model outside of the main app block to load it once
model = None
model_path = 'almond_ripeness_model.keras'

def load_my_model():
    global model
    if not model:
        model = load_model(model_path)

def predict_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image to match model's expected sizing
    image = image / 255.0  # Normalize pixel values to between 0 and 1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return 'Ripe' if prediction > 0.5 else 'Unripe'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Ensure the model is loaded
        load_my_model()

        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save uploaded file to uploads directory
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Make prediction using saved file and model
            result = predict_image(file_path, model)
            
            # Render result template with prediction
            return render_template('result.html', result=result)
    
    # Render upload page template for GET requests or if no file uploaded
    return render_template('upload.html')

if __name__ == "__main__":
    # Ensure uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Load the model when the application starts
    load_my_model()

    # Run Flask app
    app.run(debug=True)
