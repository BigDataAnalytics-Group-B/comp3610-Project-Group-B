from flask import render_template,request, jsonify
from werkzeug.utils import secure_filename
import os
from App.database import get_migrate
from App.main import create_app
from App.Controllers.controllers import get_employee_tenure_predictions
import csv

app = create_app()

from flask_dropzone import Dropzone
dropzone = Dropzone(app)
migrate = get_migrate(app)


@app.errorhandler(404)
def page_not_found(error):
    return "Page not found"

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html") 

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     # if not file.filename.endswith('.csv'):
#     #     return 'Invalid file type'
#     filename = secure_filename(file.filename)
#     file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
#     return 'File uploaded successfully'
from flask import Flask, request, session
@app.route('/upload', methods=['POST'])
def upload_file():
    delete_file()
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))

    # Remember the filename for next time
    session['filename'] = filename
    return 'File uploaded successfully'

@app.route('/delete-file', methods=['POST','GET'])
def delete_file():
    filename = session.get('filename')
    if filename:
        try:
            os.remove(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
            session.pop('filename', None)  # Remove the filename from the session
        except FileNotFoundError:
            pass  # File does not exist
    return 'File deleted'


import time
@app.route('/run-model', methods=['POST'])
def run_model():
    try:
        model = request.form['model']
        print(session)
        if model == 'all':
            pass
        elif model == 'tenure':
            results = get_employee_tenure_predictions()
            with open("App\\results.csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results)
            return render_template('index.html', results=results, download=True)
        elif model == 'clustering':
            pass
        elif model == 'anomaly':
            pass
        return "model here"
    except Exception as e:
        print(e)
        return render_template('index.html', error_message=str(e))

from flask import send_file

@app.route('/download')
def download_file():
    return send_file('results.csv', as_attachment=True)
