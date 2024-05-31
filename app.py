from flask import Flask, render_template, request, redirect, url_for, session
from datetime import timedelta, datetime
import pytz
from torchvision import transforms, models
from PIL import Image
import torch
import os
import torch.nn as nn
import logging
logging.basicConfig(level=logging.DEBUG)

# Device setting to CPU
device = torch.device("cpu")

# Define the model structure
rn50_model = models.resnet50()
rn50_model.fc = nn.Sequential(
    nn.Linear(in_features=rn50_model.fc.in_features, out_features=38)  # Sesuaikan out_features dengan jumlah kelas di model yang disimpan
)

# Load the model state
rn50_model.load_state_dict(torch.load("rn501_model.pth", map_location=device))
rn50_model = rn50_model.to(device)
rn50_model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((232, 232)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = [
    'Bacterialblight', 'Leafsmut', 'Brownspot', 
]


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.permanent_session_lifetime = timedelta(minutes=5)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Dummy user for demonstration
USERS = {'userpadi': 'userpadi'}

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            session.permanent = True
            session['user'] = username
            session['login_time'] = datetime.now(pytz.utc).isoformat()
            return redirect(url_for('home'))
        else:
            return "Invalid credentials, please try again."
    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'user' in session:
        login_time = session.get('login_time')
        if login_time:
            login_time = datetime.fromisoformat(login_time)
            if datetime.now(pytz.utc) - login_time > timedelta(minutes=5):
                session.pop('user', None)
                session.pop('login_time', None)
                return redirect(url_for('login'))
        return render_template('home.html')
    return redirect(url_for('/login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        image = Image.open(file_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = rn50_model(image)
            _, predicted = torch.max(output, 1)
            if predicted.item() >= len(classes):
                return 'Prediction out of range'
            prediction = classes[predicted.item()]

        file_url = url_for('static', filename='uploads/' + file.filename)
        return render_template('home.html', prediction=prediction, file_url=file_url)



@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    session.pop('login_time', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
