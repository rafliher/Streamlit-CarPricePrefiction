import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model
model = load_model('model.h5')

def preprocess_input(features):
    # Define categorical and numerical columns
    categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                           'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
    numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                         'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                         'peakrpm', 'citympg', 'highwaympg']

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        features[column] = label_encoder.fit_transform(features[column])

    # Feature engineering
    features['power_to_weight_ratio'] = features['horsepower'] / features['curbweight']
    for column in numerical_columns:
        features[f'{column}_squared'] = features[column] ** 2
    features['log_enginesize'] = np.log(features['enginesize'] + 1)

    # Feature scaling
    scaler = StandardScaler()
    features[numerical_columns] = scaler.fit_transform(features[numerical_columns])

    return features

def predict_price(features):
    input_data = preprocess_input(features)
    predicted_price = model.predict(input_data)

    return predicted_price[0][0]

def main():
    st.title("Prediksi Harga Mobil")

    user_input = {}
    user_input['car_ID'] = 1
    user_input['symboling'] = 0.834146
    user_input['brand'] = st.text_input("Masukkan Brand Mobil:")
    user_input['model'] = st.text_input("Masukkan Model Mobil:")
    col1, col2, col3 = st.columns(3)
    user_input['fueltype'] = col1.selectbox("Pilih Jenis Bahan Bakar", ['gas', 'diesel'])
    user_input['aspiration'] = col2.selectbox("Pilih Aspirasi Mesin", ['std', 'turbo'])
    user_input['doornumber'] = col3.selectbox("Pilih Jumlah Pintu", ['two', 'four'])
    user_input['carbody'] = col1.selectbox("Pilih Bentuk Mobil", ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'])
    user_input['drivewheel'] = col2.selectbox("Pilih Jenis Roda Penggerak", ['fwd', 'rwd', '4wd'])
    user_input['enginelocation'] = col3.selectbox("Pilih Lokasi Mesin", ['front', 'rear'])
    user_input['wheelbase'] = col1.number_input("Masukkan Wheelbase", value=100.0)
    user_input['carlength'] = col2.number_input("Masukkan Car Length", value=180.0)
    user_input['carwidth'] = col3.number_input("Masukkan Car Width", value=70.0)
    user_input['carheight'] = col1.number_input("Masukkan Car Height", value=50.0)
    user_input['curbweight'] = col2.number_input("Masukkan Curb Weight", value=2000.0)
    user_input['enginetype'] = col3.selectbox("Pilih Tipe Mesin", ['dohc', 'ohcv', 'ohc', 'l', 'rotor'])
    user_input['cylindernumber'] = col1.selectbox("Pilih Jumlah Silinder", ['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'])
    user_input['enginesize'] = col2.number_input("Masukkan Engine Size", value=150.0)
    user_input['fuelsystem'] = col3.selectbox("Pilih Sistem Bahan Bakar", ['mpfi', '2bbl', 'idi', '1bbl', 'spdi', '4bbl', 'spfi'])
    user_input['boreratio'] = col1.number_input("Masukkan Bore Ratio", value=3.5)
    user_input['stroke'] = col2.number_input("Masukkan Stroke", value=2.8)
    user_input['compressionratio'] = col3.number_input("Masukkan Compression Ratio", value=9.0)
    user_input['horsepower'] = col1.number_input("Masukkan Horsepower", value=100.0)
    user_input['peakrpm'] = col2.number_input("Masukkan Peak RPM", value=5000.0)
    user_input['citympg'] = col3.number_input("Masukkan City MPG", value=25.0)
    user_input['highwaympg'] = col2.number_input("Masukkan Highway MPG", value=30.0)

    if st.button("Prediksi Harga"):
        input_df = pd.DataFrame([user_input])
        predicted_price = predict_price(input_df)
        st.write("Prediksi Harga Mobil:", predicted_price)

if __name__ == "__main__":
    main()
