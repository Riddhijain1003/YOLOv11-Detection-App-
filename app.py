from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
import shutil

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Load YOLOv11 model
model = YOLO('yolo11n.pt')  # make sure this file is in your project directory or specify full path

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Run YOLO detection
    results = model.predict(source=filepath, save=True, project=app.config['RESULT_FOLDER'], name='detect')

    # YOLO saves result in a new subfolder, get that path
    output_dir = os.path.join(app.config['RESULT_FOLDER'], 'detect')
    detected_files = os.listdir(output_dir)
    detected_path = os.path.join(output_dir, detected_files[0])

    # Move output to static/results for serving
    final_path = os.path.join(app.config['RESULT_FOLDER'], os.path.basename(detected_path))
    shutil.move(detected_path, final_path)

    # Return display page with result
    return redirect(url_for('display_result', filename=os.path.basename(final_path)))

@app.route('/display/<filename>')
def display_result(filename):
    file_ext = filename.split('.')[-1].lower()
    return render_template('display.html', filename=filename, file_ext=file_ext)

if __name__ == '__main__':
    app.run(debug=True)
