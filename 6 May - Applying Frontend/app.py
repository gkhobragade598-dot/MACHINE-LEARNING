import streamlit as st
import pickle
import numpy as np

# load the saved model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

# set the title of the streamlit app
st.title("Salary Prediction App")

# Add a brief description of the app
st.write("This app predicts the salary based on years of experience using a linear regression model.")

# Add input widget for user to enter years of experience
years_of_experience = st.number_input("Enter years of experience:", min_value=0.0, max_value=50.0, step=0.1)

# When the button is clicked, make predictions
if st.button("Predict Salary"):
    # Make a pradicton using the train model
    experience_input = np.array([[years_of_experience]]) # convert the input to a 2D array for prediction
    prediction = model.predict(experience_input)

    # Display the result
    st.success(f"The predicted salary for {years_of_experience} years of experience is: ${prediction[0]:,.2f}")

# Display information about the model
st.write("The model was trained using a salaries and years of experience dataset. It uses linear regression to predict the salary based on the input years of experience.")
