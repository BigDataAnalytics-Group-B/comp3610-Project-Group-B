from flask import render_template,request, jsonify
from werkzeug.utils import secure_filename
import os
from App.database import get_migrate
from App.main import create_app
from App.Controllers.controllers import get_employee_tenure_predictions

app = create_app()
migrate = get_migrate(app)


@app.errorhandler(404)
def page_not_found(error):
    return "Page not found"

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html") 

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    # if not file.filename.endswith('.csv'):
    #     return 'Invalid file type'
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename))
    return 'File uploaded successfully'

import time
@app.route('/run-model', methods=['POST'])
def run_model():
    model = request.form['model']
    if model == 'all':
        pass
    elif model == 'tenure':
        results = get_employee_tenure_predictions()
        return render_template('index.html', results=results)
        pass
    elif model == 'clustering':
        pass
    elif model == 'anomaly':
        pass
    return "model here"

