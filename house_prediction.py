import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Title
st.title("🏠 House Price Prediction")

# Load dataset
df = pd.read_csv("housing.csv")

# Show data
st.subheader("Dataset Preview")
st.write(df.head())

# Features
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# User Input
st.subheader("Enter House Details")

area = st.number_input("Area", value=1000)
bedrooms = st.number_input("Bedrooms", value=2)
bathrooms = st.number_input("Bathrooms", value=2)

# Prediction
if st.button("Predict Price"):
    input_data = scaler.transform([[area, bedrooms, bathrooms]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹ {int(prediction[0])}")
