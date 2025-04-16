import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset from the given URL
filename = "https://raw.githubusercontent.com/klamsal/Fall2024Exam/main/car_dataset.csv"
df = pd.read_csv(filename)

# Streamlit Title
st.title("Part 2: Car Dataset - Data Cleaning and Analysis")

# Display raw data
st.subheader("Raw Data")
st.write(df.head())

# Step 1: Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

# Step 2: Handling Missing Values
# Convert columns with numeric data to float first
df["horsepower"] = df["horsepower"].astype(float)
df["engine-size"] = df["engine-size"].astype(float)

# Fill missing horsepower with mean
avg_horsepower = df["horsepower"].mean()
df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

# Fill missing engine-size with mean
avg_engine_size = df["engine-size"].mean()
df["engine-size"].replace(np.nan, avg_engine_size, inplace=True)

# Drop rows with missing 'price' values
df["price"] = pd.to_numeric(df["price"], errors='coerce')
df.dropna(subset=["price"], inplace=True)

# Show cleaned data
st.subheader("Cleaned Data")
st.write(df.head())

# Step 3: Unit Transformation
# Convert city-mpg to city-L/100km
df["city-mpg"] = df["city-mpg"].astype(float)
df["city-L/100km"] = 235 / df["city-mpg"]

# Convert highway-mpg to highway-L/100km
df["highway-mpg"] = df["highway-mpg"].astype(float)
df["highway-L/100km"] = 235 / df["highway-mpg"]

# Drop old mpg columns
df.drop(["city-mpg", "highway-mpg"], axis=1, inplace=True)

# Step 4: Normalization
df["length"] = df["length"].astype(float)
df["width"] = df["width"].astype(float)
df["normalized-length"] = df["length"] / df["length"].max()
df["normalized-width"] = df["width"] / df["width"].max()

# Step 5: Indicator Variable Creation
df = pd.get_dummies(df, columns=["fuel-type"], prefix="fuel", drop_first=True)
df = pd.get_dummies(df, columns=["aspiration"], prefix="asp", drop_first=True)

# Step 6: Final DataFrame Preview
st.subheader("Transformed Data")
st.dataframe(df.head())

# Step 7: Missing Value Visualization
st.subheader("Missing Values Summary")
st.bar_chart(df.isnull().sum())

# Step 8: Visualization - city L/100km vs price
st.subheader("City L/100km vs Price")
fig1, ax1 = plt.subplots()
ax1.scatter(df["city-L/100km"], df["price"])
ax1.set_xlabel("City L/100km")
ax1.set_ylabel("Price")
ax1.set_title("Fuel Consumption vs Price")
st.pyplot(fig1)

# Step 9: Visualization - Normalized Length vs Width
st.subheader("Normalized Length vs Width")
fig2, ax2 = plt.subplots()
ax2.scatter(df["normalized-length"], df["normalized-width"])
ax2.set_xlabel("Normalized Length")
ax2.set_ylabel("Normalized Width")
ax2.set_title("Normalized Car Dimensions")
st.pyplot(fig2)

