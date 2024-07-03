from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
import numpy as np
import keras
app = Flask(__name__)

model = keras.models.load_model('skin.keras')

# Define class labels
class_labels = ['Shingles', 'Chickenpox', 'Cutaneous-larva-migrans','Ringworm']

def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess input (based on VGG16 requirements)
    return img_array

@app.route('/skin', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
       return jsonify({'error': 'No selected file'})

    if file:
        try:
            img = Image.open(file)  # Open image
            img_array = preprocess_image(img)
            
            predictions = model.predict(img_array)

            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]

            class_probabilities = predictions[0].tolist()

            return jsonify({
                'predicted_class': predicted_class_label,
                'class_probabilities': dict(zip(class_labels, class_probabilities))
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 5050, debug=True)
