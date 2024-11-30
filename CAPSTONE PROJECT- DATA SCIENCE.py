#!/usr/bin/env python
# coding: utf-8

# In[50]:


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# # DATA LOADING

# In[51]:


# Load dataset
data = pd.read_csv('covid_19_clean_complete.csv')  # covid_19_clean_complete.csv is file name
data


# In[52]:


data.drop(columns=['Province/State'], inplace=True)
data.head(5)


# In[53]:


print(data.columns)


# # DATA PREPOSSESSING (CLEANING, TRANSFORMATION AND FEATURE ENGINEERING). 

# In[54]:


# --- Data Preprocessing ---

## Cleaning
# Address missing values
data.fillna(method='ffill', inplace=True)  # Forward fill for time-series data

# Remove duplicates
data.drop_duplicates(inplace=True)

# Standardize date format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Standardize location names
data['WHO Region'] = data['WHO Region'].str.strip().str.title()

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Convert 'Date' to the number of days since the earliest date in the dataset
data['Date_numeric'] = (data['Date'] - data['Date'].min()).dt.days

# Rename the 'Confirmed' column to 'Confirmed_cases'
data.rename(columns={'Confirmed': 'Confirmed_cases'}, inplace=True)


# In[55]:


## Transformation
# Normalize numerical columns
from sklearn.preprocessing import MinMaxScaler

numerical_columns = ['Lat', 'Long', 'Date_numeric', 'Confirmed_cases', 'Deaths', 'Recovered', 'Active']  # numerical columns
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])


# Feature Engineering
# Daily growth rates
data['daily_growth_rate'] = data.groupby('WHO Region')['Confirmed_cases'].pct_change().fillna(0)

# Mortality ratio
data['mortality_ratio'] = data['Deaths'] / data['Date_numeric'].replace(0, np.nan)

# Cases per population
data['confirmed_per_population'] = data['Confirmed_cases'] / data['Date_numeric'].replace(0, np.nan)

# Replace infinities and NaNs resulting from division
for col in ['mortality_ratio', 'confirmed_per_population']:
    data[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    data[col].fillna(0, inplace=True)


# In[56]:


# Define X and y again
X = data[['Confirmed_cases', 'Recovered', 'Active', 'daily_growth_rate', 'Date_numeric']]
y = (data['Deaths'] > 0.05).astype(int)

# Check for NaN values
print("NaN values in X:")
print(X.isnull().sum())

# Check for infinity values
print("Infinity values in X:")
print(np.isinf(X).sum())

# Check if any values are extremely large
print("Max values in X:")
print(X.max())
print("Min values in X:")
print(X.min())

# Check for invalid values in y
print("NaN or invalid values in y:")
print(y.isnull().sum())
print(np.isinf(y).sum())

# Replace NaN and infinity in X
X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinity with NaN
X = X.fillna(0)  # Replace NaN with 0, or use imputation if preferred

# Replace NaN or infinity in y (if applicable)
y = y.replace([np.inf, -np.inf], np.nan)
y = y.fillna(0)


# # EXPLORATORY DATA ANALYSIS (EDA)

# In[57]:



## General dataset overview
data.describe()
print(data.info())
print(data.describe())
data.isnull().sum() # missing values check


## Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[58]:



# Trend of Confirmed Cases Over Time
plt.figure(figsize=(10, 6))
plt.plot(data['Date_numeric'], data['Confirmed_cases'], label='Confirmed Cases', color='blue')
plt.xlabel('Days Since First Case')
plt.ylabel('Number of Cases')
plt.title('Trend of Confirmed COVID-19 Cases Over Time')
plt.legend()
plt.show()

# Mortality Rate Over Time
data['mortality_rate'] = data['Deaths'] / data['Confirmed_cases']
plt.figure(figsize=(10, 6))
plt.plot(data['Date_numeric'], data['mortality_rate'], label='Mortality Rate', color='red')
plt.xlabel('Days Since First Case')
plt.ylabel('Mortality Rate')
plt.title('COVID-19 Mortality Rate Over Time')
plt.legend()
plt.show()

# Daily Growth Rate Over Time
# Calculate the daily growth rate: (New Cases Today / Cases Yesterday) - 1
data['daily_growth_rate'] = data['Confirmed_cases'].pct_change()

plt.figure(figsize=(10, 6))
plt.plot(data['Date_numeric'], data['daily_growth_rate'], label='Daily Growth Rate', color='green')
plt.xlabel('Days Since First Case')
plt.ylabel('Daily Growth Rate')
plt.title('COVID-19 Daily Growth Rate Over Time')
plt.legend()
plt.show()

# Scatter Plot of Cases vs. Date_numeric
plt.figure(figsize=(10, 6))
plt.scatter(data['Confirmed_cases'], data['Confirmed_cases'] / data['Date_numeric'], alpha=0.5)
plt.xlabel('Confirmed_Cases')
plt.ylabel('Cases per Date_numeric')
plt.title('Scatter Plot of Confirmed Cases vs. Cases per Date_numeric')
plt.show()

# Box Plot for Outliers in Confirmed Cases by WHO Region
plt.figure(figsize=(12, 6))
sns.boxplot(x='WHO Region', y='Confirmed_cases', data=data)
plt.title('Boxplot of Confirmed Cases by WHO Region')
plt.show()


# In[59]:


time_series_predictions = {}
location_data = data[['Date_numeric', 'Confirmed_cases']].set_index('Date_numeric')
location_data = location_data.sort_index()

# Fit ARIMA model
model = ARIMA(location_data['Confirmed_cases'], order=(5, 1, 0))
model_fit = model.fit()

# Predict next 30 days
last_index = int(location_data.index[-1])  # Ensure integer index
forecast = model_fit.forecast(steps=30)
prediction_index = range(last_index + 1, last_index + 31)

# Store predictions
time_series_predictions['Overall'] = pd.Series(forecast, index=prediction_index)

# Visualize
for location, prediction in time_series_predictions.items():
    plt.figure(figsize=(10, 6))
    plt.plot(location_data.index, location_data['Confirmed_cases'], label='Actual Cases', color='blue')
    plt.plot(prediction.index, prediction, label='Predicted Cases', color='red')
    plt.title(f'Time-Series Forecast for {location}')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    plt.show()


# # MACHINE LEARNING MODELS

# In[60]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale features to have mean=0 and std=1
# Check for remaining NaN or infinity
print("After cleaning:")
print("NaN in X:", X.isnull().sum().sum())
print("Infinity in X:", np.isinf(X).sum())

# If scaled:
print("Max value in scaled X:", X_scaled.max())
print("Min value in scaled X:", X_scaled.min())


# In[61]:


# 1. Classification Models: Predicting Mortality Risk

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-Score: {f1_score(y_test, y_pred)}")


# In[62]:


# Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf)}")
print(f"Recall: {recall_score(y_test, y_pred_rf)}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf)}")


# In[63]:


# 2. Time-Series Models: ARIMA for Predicting Future Cases

train_data = data[['Date_numeric', 'Confirmed_cases']].set_index('Date_numeric')
train_data = train_data.sort_index()

# Fit ARIMA model
model = ARIMA(train_data['Confirmed_cases'], order=(5,1,0))  # ARIMA(p,d,q) hyperparameters
model_fit = model.fit()

# Predict next 30 days
forecast = model_fit.forecast(steps=30)

# Convert index to integer if needed
last_index = int(train_data.index[-1])  # Ensure the index is an integer
forecast_index = range(last_index + 1, last_index + 31)  # Create a range for the next 30 days

# Plot the historical data and forecast
plt.plot(train_data.index, train_data['Confirmed_cases'], label='Historical Data')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast of Confirmed COVID-19 Cases')
plt.xlabel('Days Since First Case')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()


# # MODEL EVALUATION

# In[64]:


# HYPERPARAMETER TUNING
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Define StratifiedKFold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Perform cross-validation using StratifiedKFold
cv_scores = cross_val_score(log_reg, X, y, cv=skf, scoring='f1')

# Print results
print("Stratified Cross-Validation F1-Scores:", cv_scores)
print("Mean F1-Score:", cv_scores.mean())


# In[65]:


# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))  # For regression or time-series models
print(f"RMSE: {rmse}")

