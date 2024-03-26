import pandas as pd

def one_hot_encode(df):
    # Handle categorical variables: 'Department' and 'salary'
    df = pd.get_dummies(df, columns=['Department', 'salary'])

    # Drop rows with missing values
    df.dropna(inplace=True)

    return df


