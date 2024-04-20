from joblib import load
import pandas as pd
from flask import session
from ProjectFiles.DataPreprocessing.data_cleaning import *
import os
from sklearn.preprocessing import StandardScaler

def convert_to_years_months(total_years):
    # Separate the year into its integer and fractional components
    years_int = int(total_years)
    months_fraction = total_years - years_int
    
    # Convert the fractional year to months
    months = round(months_fraction * 12)
    
    # Adjust for rounding up to 12 months, which should roll over to an additional year
    if months == 12:
        years_int += 1
        months = 0
    return years_int, months  


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
    
    clean_df = one_hot_encode(df)
    clean_df = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours']]
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
       
    return results_list
  

def get_employee_clusters():
    loaded_model = load('ProjectFiles/Models/kmeans_model.joblib')

    filename = 'App/uploads/' + session.get('filename')
    df = load_data(filename)
    df.dropna(inplace=True)

    if filename.endswith('.csv'):
        df['last_evaluation'] = df['last_evaluation'].str.rstrip('%').astype(float) / 100.0

    selected_features = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours']


    df_subset = df[selected_features]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_subset)

    km_clusters = loaded_model.predict(scaled_features)

    df_subset['cluster'] = km_clusters
    # print(df_subset.head())


    test_df = df_subset.copy()
    # test_df.reset_index(inplace=True)



    keep_features = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'left', 'Emp_Id']

    df_selected_left = df[keep_features]

    # Merge cluster_df with original_dataset using the index as the joining key
    merged_df = df_selected_left.merge(df_subset, left_index=True, right_index=True)

    # Select only the specified columns
    merged_df = merged_df[['satisfaction_level_x', 'last_evaluation_x', 'number_project_x',
                        'average_montly_hours_x', 'left', 'cluster', 'Emp_Id']]

    merged_df.rename(columns={
        'satisfaction_level_x': 'satisfaction_level',
        'last_evaluation_x': 'last_evaluation',
        'number_project_x': 'number_project',
        'average_montly_hours_x': 'average_monthly_hours'
    }, inplace=True)
    
    
    project_scales = define_scale(merged_df, 'number_project', None, None)
    satisfaction_scales = define_scale(merged_df, 'satisfaction_level', 0, 1)
    hours_scales = define_scale(merged_df, 'average_monthly_hours', None, None)
    evaluation_scales = define_scale(merged_df, 'last_evaluation', 0, 1)

    scales = {
        'number_project': project_scales,
        'satisfaction_level': satisfaction_scales,
        'average_monthly_hours': hours_scales,
        'last_evaluation': evaluation_scales
    }

    cluster_means = cluster_analysis(merged_df, 'cluster', ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'left'])

    insights = generate_insights(cluster_means, scales)

    cluster_counts = merged_df.groupby('cluster').size().to_dict()

    for cluster, count in cluster_counts.items():
        # Calculate turnover rate for the cluster
        cluster_turnover_rate = merged_df[(merged_df['cluster'] == cluster) & (merged_df['left'] == 1)].shape[0] / count
        cluster_turnover_rate = round(cluster_turnover_rate * 100, 2)     

        insights[cluster]['count'] = count # Add count of employees in cluster
        insights[cluster]['turnover_rate'] = cluster_turnover_rate

    # print(insights)

    employee_clusters = merged_df.copy()
    employee_clusters = employee_clusters[['Emp_Id', 'cluster']]

    return insights, employee_clusters


def define_scale(data, feature, lower_bound, upper_bound):
    if lower_bound is None:
        lower_bound = data[feature].min()
    if upper_bound is None:
        upper_bound = data[feature].max()

    # Define scale labels
    scales = {}

    # Define value ranges for each scale label
    scale_range = (upper_bound - lower_bound) / 3
    scales['Low'] = (round(lower_bound, 2), round(lower_bound + scale_range, 2))
    scales['Moderate'] = (round(lower_bound + scale_range, 2), round(lower_bound + 2 * scale_range, 2))
    scales['High'] = (round(lower_bound + 2 * scale_range, 2), round(upper_bound, 2))

    return scales

def cluster_analysis(data, cluster_column, feature_columns):
    cluster_means = data.groupby(cluster_column)[feature_columns].mean()

    return cluster_means

def generate_insights(cluster_means, scales):
    insights = {}

    cluster_counts = cluster_means.index.value_counts().to_dict()

    for cluster, means in cluster_means.iterrows():
        cluster_insights = {}
        
        # Compare mean feature values to defined scales for each feature
        for feature, value in means.items():
            # Skip if the feature is not in the scales dictionary
            if feature not in scales:
                continue
            
            # Determine the scale label for the feature value
            for scale_label, scale_range in scales[feature].items():
                if scale_range[0] <= value <= scale_range[1]:
                    cluster_insights[feature] = scale_label
                    break

        cluster_insights['count'] = cluster_counts.get(cluster, 0)
        
        insights[cluster] = cluster_insights


    return insights
