import pandas as pd
import pickle as pk
import streamlit as st

# Load the trained model
model = pk.load(open('model.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Custom CSS styling for a clean professional look
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 2.2rem;
            color: #2c3e50;
            font-weight: 700;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .sub-title {
            text-align: center;
            color: #6c757d;
            font-size: 1.05rem;
            margin-bottom: 30px;
        }
        .stButton button {
            background-color: #2c3e50;
            color: white;
            border-radius: 8px;
            height: 3rem;
            font-size: 1.05rem;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1abc9c;
            color: white;
            transform: scale(1.02);
        }
        .result-box {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.25rem;
            color: #2c3e50;
            font-weight: 600;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            margin-top: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# Titles
st.markdown('<h1 class="main-title">Car Price Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Provide the car details below to estimate its market selling price.</p>', unsafe_allow_html=True)

# Load dataset
cars_data = pd.read_csv('Cardetails.csv')

# Extract brand names
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

# Two-column layout for form inputs
col1, col2 = st.columns(2)

with col1:
    name = st.selectbox('Car Brand', cars_data['name'].unique())
    year = st.slider('Manufacturing Year', 1994, 2024, 2020)
    km_driven = st.slider('Kilometers Driven', 10, 200000, 50000)
    fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique())
    seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique())
    transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique())

with col2:
    owner = st.selectbox('Owner Category', cars_data['owner'].unique())
    mileage = st.slider('Mileage (km/l)', 10, 40, 15)
    engine = st.slider('Engine Capacity (CC)', 700, 5000, 1500)
    max_power = st.slider('Maximum Power (bhp)', 0, 200, 80)
    seats = st.slider('Number of Seats', 4, 10, 5)

# Predict button
if st.button("Predict Car Price"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encoding categorical variables
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'], [1,2,3,4,5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1,2,3,4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1,2,3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1,2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
       [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], inplace=True)

    # Make prediction
    predicted_price = model.predict(input_data_model)[0]

    # Display result
    st.markdown(f'<div class="result-box">Estimated Selling Price: ₹ {predicted_price:,.2f}</div>', unsafe_allow_html=True)
