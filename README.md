# Covid-19
Using historical COVID-19 data to conduct data cleaning, perform exploratory data analysis (EDA), and develop predictive models to forecast COVID-19 trends.
During the course of the project, the following were carried out:
  -Data Collecting 
  -Data Prepossessing (Cleaning, Transformation and Feature Engineering
  -Exploration Data Analysis (EDA)
  -Machine Learning Modelling
  -Data Evaluations/Predictions
The dataset was accessed from the COVID-19 Open Research Dataset (CORD-19) on Kaggle, which includes COVID-19 case counts, demographic data, and various health metrics.
Data Cleaning involved replacing missing values with `0` for numerical features and ensured categorical features were imputed appropriately. Also, removed outliers by appling z-score normalization to identify and address extreme outliers. NaN and infinity were replaced with 0, or imputation if preferred. 
Feature Engineering involved calculation of Daily Growth Rate, Mortality Ratio, Cases per Population by analysis the final features to include Numerical (Confirmed Cases, Recovered, Active, Daily Growth Rate, Mortality Ratio, Cases per Population) and Target Variables (which includes Regression to predict future confirmed cases) and Classification to identify regions with high mortality rates of threshold > 5%). 
Transformation involves Normalization i.e Scaled numerical features (e.g., Confirmed Cases, Deaths, Recovered) using MinMaxScaler to range between 0 and 1.

![image](https://github.com/user-attachments/assets/1f77c8fe-47ee-4d66-a4c3-9ba1618a063a)
