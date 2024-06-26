from flask import Flask, render_template, request, jsonify, url_for, redirect, session
from tensorflow.keras.models import load_model
import tensorflow as tf
import time
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'sbgbfmjwbo'

# Definisikan LRN custom layer
class LRN(tf.keras.layers.Layer):
    def __init__(self, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, **kwargs):
        super(LRN, self).__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        return tf.nn.local_response_normalization(inputs, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)

local_model_path = 'model7.h5'

model = load_model(local_model_path, custom_objects={'LRN': LRN})

# Definisi kelas yang sesuai dengan indeks prediksi
classes = ['Chickenpox', 'Cowpox', 'Healthy', 'Monkeypox']

users = {
    'user1': 'password1',
    'user2': 'password2'
}

@app.before_request
def check_login():
    allowed_routes = ['index', 'login', 'static']
    if request.endpoint not in allowed_routes and 'username' not in session:
        return redirect(url_for('login'))

@app.route('/', methods=['GET'])
def index():
    username = session.get('username', None)  
    return render_template('login.html', username=username) 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username not in users and password not in users.values():
            error = "INVALID USERNAME AND PASSWORD"
        elif username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        elif username not in users:
            error = 'INVALID USERNAME'
        else:
            error = 'INVALID PASSWORD'
            
        return render_template('login.html', error=error)
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session.get('username')
    return render_template('dashboard.html', username=username)

@app.route('/index1', methods=['GET'])
@app.route('/predict', methods=['POST'])
def index1():
    if request.method == 'POST':
        start_time = time.time()

        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (227, 227))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = classes[predicted_class_idx]

        end_time = time.time()
        duration = end_time - start_time
        format = 60-duration
        min = (format)

        return jsonify({'prediction': str(predicted_class), 'duration': duration})

    return render_template('index1.html')

@app.route('/index2', methods=['GET'])
@app.route('/predict1', methods=['POST'])
def index2():
    if request.method == 'POST':
        img = request.files['image'].read()
        npimg = np.fromstring(img, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (227, 227))
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = classes[predicted_class_idx]
        return jsonify({'prediction': str(predicted_class)})

    return render_template('index2.html')


if __name__ == '__main__':
    app.run(debug=True)
