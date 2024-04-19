# comp3610-Project-Group-B

## Dependencies
- Python3/pip3
- Packages listed in requirements.txt

## Installing Dependencies
```python
$ pip install -r requirements.txt
```

# Employee Turnover Analysis

## Problem Statement
Frequent turnover in multinational corporations obstructs growth and stability. This is driven by insufficient career pathways, lack of acknowledgement, and ineffective diversity strategies. The significant issue of employee turnover impacts operational efficiency, increases recruitment costs, and affects overall organisational morale. Current initiatives fall short due to insufficient knowledge about the factors leading to employee attrition and the typical tenure within the workforce. This gap in understanding hinders the development and implementation of effective strategies for employee retention and workforce optimization.

## Data Sources
- [Kaggle Database](https://www.kaggle.com/datasets/kmldas/hr-employee-data-descriptive-analytics/data)
- [Alt. Link and Documentation](https://github.com/ryankarlos/Human-Resource-Analytics/tree/master/Original_Kaggle_Dataset) (Kaggle Usability Score: 8.53)

The dataset, from an undisclosed MNC, offers comprehensive HR information facilitating analysis of employees and their turnover. It comprises 15,000 samples with 10 variables, exhibiting a class imbalance with 11,428 samples representing employees who stayed and 3,571 samples representing those who left.

## Assumptions
Insights and conclusions drawn from analysis of this dataset may be generalized to other corporations.

## Analysis Methods and Algorithms
- Survival Analysis for Employee Tenure: Estimate and assess the time until an employee leaves the company using survival analysis techniques.
- Employee Clustering: Use unsupervised learning to segment employees into distinct groups based on attributes such as satisfaction.
- Anomaly Detection for Work Hours: Identify unusual patterns or anomalies in average monthly hours as an indication of issues such as burnout or resignation.

## Expected Outcomes
- Determine Factors leading to High Turnovers: Identify the factors that significantly determine the time an employee stays at the company along with discovering the correlation between these factors and the turnover at the MNC.
- Predict Employee’s Duration at the company: Create a model to predict and determine how long (in years) an employee will stay at an MNC company based on several features.
- Categorise Employees based on specific Factors: Categorise employees based on factors such as employee satisfaction, employee competency, and employees’ work ethic.

## Deliverables
- [Github Repository](https://github.com/BigDataAnalytics-Group-B/comp3610-Project-Group-B): A repository containing the Jupyter Notebook with source code.
- Final Report: A 10-page final report outlining the details, findings, and other key information about the project.
- [Web Application](https://comp3610-project-group-b.onrender.com/): A simple web-based flask application that enables companies to upload employee data in csv or xslx format. The application will analyse the data to identify key retention drivers and forecast employee tenure.