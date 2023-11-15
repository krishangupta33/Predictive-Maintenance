import pandas as pd
import numpy as np
import streamlit as st
import pickle
#krishan_frontend.py
# Importing model
model = pickle.load(open('krishan_demo_model.pkl', 'rb'))

# Read krishan_train_demo.csv
df = pd.read_csv('krishan_train_demo.csv')

# Predicting on the first 10 rows of the dataset
predictions = model.predict(df.drop(['logerror'], axis=1).head(10))

# Making front end using streamlit
st.title("Zillow's Home Value Prediction")
st.write("This app predicts the logerror of a house.")
st.write("The logerror is the difference between the actual price and the predicted price.")

# Showing demo data for 5 rows
st.subheader("Demo data")
st.write(df.head())

# Getting user input for all the columns of the dataset except logerror
new_df = pd.DataFrame(index=[0])  # Create a new DataFrame with one row for input

for i in df.columns[:-1]:
    st.write("Enter the value of", i)
    default_value = int(df[i][0])  # Convert to integer, assuming df[i][0] is not NaN and is integer-compatible
    new_df.at[0, i] = st.number_input("", value=default_value, step=1, format="%d", key=f"input_{i}")




# Check if input DataFrame has been filled to trigger the prediction
if new_df[df.columns[:-1]].isnull().values.any() == False:
    # Predicting the logerror
    pred = model.predict(new_df)

    # Displaying the logerror
    st.write("The predicted logerror is:", pred[0])
else:
    st.write("Please enter the values for all fields.")
