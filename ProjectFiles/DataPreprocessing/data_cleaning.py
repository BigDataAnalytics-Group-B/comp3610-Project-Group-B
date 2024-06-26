import pandas as pd

def one_hot_encode(df):
    # Handle categorical variables: 'Department' and 'salary'
    df = pd.get_dummies(df, columns=['Department', 'salary'], prefix=['Department_', 'salary_'])

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df

def percentConverterCSV(x):
        return float(x.strip('%'))/100
