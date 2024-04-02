from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('C:/Users/WINDOWS/Documents/TA/Referensi TA/Code/model5.h5')

# Definisi kelas yang sesuai dengan indeks prediksi
classes = ['Chickenpox', 'Cowpox', 'Healthy', 'Monkeypox']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/index1', methods=['GET'])
@app.route('/predict', methods=['POST'])
def index1():
    if request.method == 'POST':
        file = request.files['image']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (227, 227))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = classes[predicted_class_idx]
        return jsonify({'prediction': str(predicted_class)})

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(debug=True)