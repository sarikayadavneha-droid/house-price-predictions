# house_price.py

# Step 1: Import Libraries
import streamlit as st
import pandas as pd
import numpy as np


# Step 2: Load Dataset
df = pd.read_csv("housing.csv")

# Step 3: Explore Data
print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
df.info()

print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Data Cleaning
df = df.dropna()

# Step 5: Feature Selection
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Step 6: Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Step 7: Feature Scaling (BEFORE model training)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train Model
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Step 8: Prediction
y_pred_lr = lr_model.predict(X_test)

# Step 9: Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Step 9: Evaluation
from sklearn.metrics import mean_squared_error, r2_score

print("\n--- Linear Regression ---")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

print("\n--- Random Forest ---")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))



# Step 10: Visualization
plt.scatter(y_test, y_pred_lr)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Step 11: Predict New Data
new_house = pd.DataFrame({
    'area': [1500],
    'bedrooms': [3],
    'bathrooms': [2]
})
new_house_scaled = scaler.transform(new_house)

predicted_price = lr_model.predict(new_house)
print("\nPredicted Price for new house:", predicted_price[0])
