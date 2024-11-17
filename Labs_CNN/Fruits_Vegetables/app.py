import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and set up categories
model = load_model('Image_classify.keras')
data_cat = ['lemon', 'lettuce', 'mango', 'orange']  # Update based on your model classes
img_height = 180
img_width = 180

# Home route to display the image upload form
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Ensure that a file was uploaded
        if 'file' not in request.files:
            return "No file part in the request"
        
        file = request.files["file"]
        if file.filename == "":
            return "No image selected for uploading"
        
        if file:
            # Read and preprocess the image
            image = Image.open(file.stream).convert('RGB')
            image = image.resize((img_height, img_width))
            img_arr = tf.keras.utils.img_to_array(image)
            img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

            # Perform prediction
            prediction = model.predict(img_bat)
            score = tf.nn.softmax(prediction[0])

            # Get the predicted label and accuracy
            predicted_label = data_cat[np.argmax(score)]
            accuracy = np.max(score) * 100

            # Render result page with prediction
            return render_template("result.html", label=predicted_label, accuracy=accuracy)
    
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
