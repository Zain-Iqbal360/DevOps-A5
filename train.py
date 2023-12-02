import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
# Assuming your dataset is in a CSV file named 'temperatures.csv' with columns 'Date', 'Min_Temperature', and 'Max_Temperature'
data = pd.read_csv('./year_lahore_weather_data.csv')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extracting the number of days since the start date as a feature
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Splitting the dataset into features (X) and target variables (y_min, y_max)
X = data[['Days']]
y_min = data['Min_Temperature']
y_max = data['Max_Temperature']

# Splitting data into training and testing sets
X_train, X_test, y_min_train, y_min_test, y_max_train, y_max_test = train_test_split(X, y_min, y_max, test_size=0.2, random_state=42)

# Initialize Linear Regression models for Min_Temperature and Max_Temperature
model_min_temp = LinearRegression()
model_max_temp = LinearRegression()

# Fit the models on the training data
model_min_temp.fit(X_train, y_min_train)
model_max_temp.fit(X_train, y_max_train)

# Save the trained models to files as .joblib
joblib.dump(model_min_temp, 'trained_model_min_temp.joblib')
joblib.dump(model_max_temp, 'trained_model_max_temp.joblib')

print("Models for Min_Temperature and Max_Temperature trained and saved as trained_model_min_temp.joblib and trained_model_max_temp.joblib")

