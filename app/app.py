import streamlit as st
import pandas as pd
import joblib

# Set up the title
st.title("House Price Prediction App")

# Sidebar for user input
st.sidebar.header("Please Enter Housing Details")

type_encoder = joblib.load('C:\\House_Price_Prediction\\src\\encoders\\type_encoder.joblib')
region_encoder = joblib.load('C:\\House_Price_Prediction\\src\\encoders\\region_encoder.joblib')
subregion_encoder = joblib.load('C:\\House_Price_Prediction\\src\\encoders\\subregion_encoder.joblib')

# User inputs
Number_of_Beds = st.sidebar.number_input("Number of Beds", min_value=1.0, max_value=46.0, value=23.0)
Number_of_Baths = st.sidebar.number_input("Number of Baths", min_value=1.0, max_value=10.0, value=5.0)
Area = st.sidebar.number_input("Area", min_value=320.0, max_value=43344.0, value=21672.0)

subregion_name = st.sidebar.selectbox("Sub-region", subregion_encoder.classes_)
# Transform to encoded value
subregion_encoded = subregion_encoder.transform([subregion_name])[0]

region_name = st.sidebar.selectbox("Region", region_encoder.classes_)

# Transform to encoded value
region_encoded = region_encoder.transform([region_name])[0]

type_name = st.sidebar.selectbox("Type_n", type_encoder.classes_)

# Transform to encoded value
type_encoded = type_encoder.transform([type_name])[0]


# Create DataFrame for prediction
input_data = {
    "No. Beds" : Number_of_Beds,
    "No. Baths" : Number_of_Baths,
    "Area" : Area,
    "Type_n" : type_encoded,
    "Region_n" : region_encoded,
    "Sub-region_n" : subregion_encoded,
}

input_data_df = pd.DataFrame([input_data])

# Load trained model
model_path = "C:\\House_Price_Prediction\\src\models\\Random_Forest.joblib"


try:
    model = joblib.load(model_path)
    
    # Make a prediction
    result = model.predict(input_data_df)

    # Display input data
    st.table(input_data_df)

    
    st.metric('Predicted House Price: ', f'{result[0]:,.2f}', 'Tk')

except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Train and save the model first.")