from joblib import load
import pandas as pd
from flask import session
from ProjectFiles.DataPreprocessing.data_cleaning import *
import os

def convert_to_years_months(total_years):
    # Separate the year into its integer and fractional components
    years_int = int(total_years)
    months_fraction = total_years - years_int
    # Convert the fractional year to months, rounding to the nearest whole number
    months = round(months_fraction * 12)
    # Adjust for rounding up to 12 months, which should roll over to an additional year
    if months == 12:
        years_int += 1
        months = 0
    return years_int, months  # Or return f"{years_int} years, {months} months" for a string representation


def load_data(filename):
    file_extension = os.path.splitext(filename)[1]
    read_functions = {
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.csv': lambda f: pd.read_csv(f, converters={'satisfaction_level':percentConverterCSV})
    }
    read_function = read_functions.get(file_extension)
    if read_function is None:
        raise ValueError('Unsupported file type')
    return read_function(filename)

def get_employee_tenure_predictions():
    loaded_model = load('App/Controllers/best_model.joblib')
    filename = 'App/uploads/' + session.get('filename')
    df = load_data(filename)
    clean_df = df[['satisfaction_level', 'number_project']]
    new_predictions = loaded_model.predict(clean_df)
    predictions_in_years_months = [convert_to_years_months(pred) for pred in new_predictions]
    return predictions_in_years_months

