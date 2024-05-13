from flask import Flask, render_template, request, jsonify, url_for, redirect, session
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'secret_key'
model = load_model('C:/Users/WINDOWS/Documents/TA/Referensi TA/Code/Project/TA/model7.h5')

# Definisi kelas yang sesuai dengan indeks prediksi
classes = ['Chickenpox', 'Cowpox', 'Healthy', 'Monkeypox']

users = {
    'user1': 'password1',
    'user2': 'password2'
}

# Fungsi dekorator untuk memeriksa apakah pengguna telah login sebelum mengakses halaman tertentu
@app.before_request
def check_login():
    allowed_routes = ['index', 'login', 'static']
    if request.endpoint not in allowed_routes and 'username' not in session:
        return redirect(url_for('login'))

@app.route('/', methods=['GET'])
def index():
    username = session.get('username', None)  # Ambil nama pengguna dari session
    return render_template('login.html', username=username)  # Teruskan nama pengguna ke template HTML

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        elif username in users:
            session['username'] = username
            error = 'INVALID USERNAME'
            return render_template('login.html', error=error)
        elif password in users:
            session['password'] = password
            error = 'INVALID PASSWORD'
            return render_template('login.html', error=error)
        else:
            error = 'INVALID USERNAME AND PASSWORD'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

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
