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
    loaded_model = load('ProjectFiles/Models/best_model.joblib')
    filename = 'App/uploads/' + session.get('filename')
    df = load_data(filename)
    
    # Save the 'Emp_Id' column
    emp_ids = df['Emp_Id']
    
    clean_df = df[['satisfaction_level', 'number_project']]
    new_predictions = loaded_model.predict(clean_df)
    
    # Convert predictions to years and months
    predictions_in_years_months = [convert_to_years_months(pred) for pred in new_predictions]
    
    # Combine 'Emp_Id' and predictions into a list of lists
    results = [[emp_id] + list(prediction) for emp_id, prediction in zip(emp_ids, predictions_in_years_months)]
    
    return results

def get_employee_anomalies():
    
    loaded_model = load('ProjectFiles/Models/anomaly_model.joblib')
    filename = 'App/uploads/' + session.get('filename')
    
    df = load_data(filename)

    if 'average_montly_hours' in df.columns:
        df['average_monthly_hours'] = df['average_montly_hours'].copy()
        df.drop(columns=['average_montly_hours'], inplace=True)
    
    df = pd.get_dummies(df, columns=['Department', 'salary'])

    df.dropna(inplace=True)
    emp_ids = df['Emp_Id']

    df = df.drop(['Emp_Id'], axis=1)
    
    work_hours = df['average_monthly_hours'].values.reshape(-1, 1)

    lm_outliers = loaded_model.predict(work_hours)

    lm_anomaly_indices = df.index[lm_outliers == -1]

    lm_anomaly_df = df.iloc[lm_anomaly_indices]

    lm_emp_ids = emp_ids.iloc[lm_anomaly_df.index.values]

    lm_average_monthly_hours = lm_anomaly_df['average_monthly_hours']

    results_df = pd.DataFrame({
        'Emp_Id': lm_emp_ids,
        'average_monthly_hours': lm_average_monthly_hours
    })
    
    results_list = results_df.to_dict(orient='records')
    
    # print(results_list)
    
    return results_list

    
    

    
    



