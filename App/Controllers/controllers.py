from joblib import load
import pandas as pd

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


def get_employee_tenure_predictions():

    loaded_model = load('App/Controllers/best_model.joblib')
    
    df = pd.read_excel('App\\uploads\\HR_Employee_Data.xlsx')
    
    clean_df = df[['satisfaction_level', 'number_project']]
    
    new_predictions = loaded_model.predict(clean_df)
    
    predictions_in_years_months = [convert_to_years_months(pred) for pred in new_predictions]
    
    
    return predictions_in_years_months
