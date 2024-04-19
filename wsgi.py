from flask import render_template,request, jsonify
from werkzeug.utils import secure_filename
import os
from App.database import get_migrate
from App.main import create_app
from App.Controllers.controllers import get_employee_tenure_predictions, get_employee_clusters, get_employee_anomalies
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
        print("model is " + model)
        if model == 'all':
            pass
        elif model == 'tenure':
            try:
                results = get_employee_tenure_predictions()
                with open("App\\results.csv", 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(results)
            except Exception as E:
                print(E)
            return render_template('index.html', results=results, download=True)
        elif model == 'clustering':
            try:
                insights = get_employee_clusters()

                print("toothache")
                print(insights)

                # with open("App\\resultsClustering.csv", 'w', newline='') as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerows(insights)

                with open("App\\resultsClustering.csv", 'w', newline='') as csvfile:
                    fieldnames = ['cluster', 'satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'count', 'turnover_rate']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Convert insights dictionary to a list of dictionaries
                    insights_list = [{**{'cluster': k}, **v} for k, v in insights.items()]

                    writer.writerows(insights_list)


            except Exception as E:
                print(E)

            return render_template('index.html', insights=insights, download=True)

        elif model == 'anomaly':
            try:
                resultsA = get_employee_anomalies()
     
                with open("App\\resultsAnomaly.csv", 'w', newline='') as csvfile:
                    fieldnames = ['Emp_Id', 'average_monthly_hours']  # Define field names
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()  # Write the header row based on fieldnames
                    for result in resultsA:
                        writer.writerow(result)  # Write each dictionary row
                   
            except Exception as E:
                print(E)
            return render_template('index.html', resultsA=resultsA, download=True)
            pass
        return "model here"
    except Exception as e:
        print(e)
        return render_template('index.html', error_message=str(e))

from flask import send_file
import pandas as pd
from tempfile import NamedTemporaryFile

@app.route('/download')
def download_file():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('App\\results.csv')

    # Use a temporary file to avoid file management issues
    with NamedTemporaryFile(delete=False, suffix='.xlsx', mode='w+b') as tmp:
        # Convert the DataFrame to an XLSX file and save it
        df.to_excel(tmp.name, index=False)

        # Prepare the response, sending the temporary file as an attachment
        response = send_file(tmp.name, as_attachment=True)
        # Specify the desired download name in the Content-Disposition header
        response.headers["Content-Disposition"] = "attachment; filename=results.xlsx"
        
        return response
    
@app.route('/downloadAnomaly')
def download_file_A():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('App\\resultsAnomaly.csv')

    # Use a temporary file to avoid file management issues
    with NamedTemporaryFile(delete=False, suffix='.xlsx', mode='w+b') as tmp:
        # Convert the DataFrame to an XLSX file and save it
        df.to_excel(tmp.name, index=False)

        # Prepare the response, sending the temporary file as an attachment
        response = send_file(tmp.name, as_attachment=True)
        # Specify the desired download name in the Content-Disposition header
        response.headers["Content-Disposition"] = "attachment; filename=resultsAnomaly.xlsx"
        
        return response

@app.route('/downloadClustering')
def download_file_Clustering():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('App\\resultsClustering.csv')

    # Use a temporary file to avoid file management issues
    with NamedTemporaryFile(delete=False, suffix='.xlsx', mode='w+b') as tmp:
        # Convert the DataFrame to an XLSX file and save it
        df.to_excel(tmp.name, index=False)

        # Prepare the response, sending the temporary file as an attachment
        response = send_file(tmp.name, as_attachment=True)
        # Specify the desired download name in the Content-Disposition header
        response.headers["Content-Disposition"] = "attachment; filename=resultsClustering.xlsx"
        
        return response