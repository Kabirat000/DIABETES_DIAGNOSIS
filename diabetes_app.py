import numpy as np
import streamlit as st
import pickle

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Create a function for prediction

def diabetes_prediction(input_data):
    # Convert the input data into a numpy array
    input_data_as_numpy = np.asarray(input_data)

    # Reshape the array since we are predicting for one instance
    input_data_reshape = input_data_as_numpy.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshape)

    if prediction[0] == 0:
        return 'This patient is not Diabetic'
    else:
        return 'This patient is Diabetic'
    
def main():
    # Give a title for the interface
    st.title('Diabetes Prediction Web App')

    # Getting the input from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age of the patient')

    # Code for prediction
    diagnosis = ''

    # Create a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                         DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()
