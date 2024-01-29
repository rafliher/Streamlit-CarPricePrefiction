import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')
category_mapping_dict = {
    "fueltype": {"diesel": 0, "gas": 1},
    "aspiration": {"std": 0, "turbo": 1},
    "doornumber": {"four": 0, "two": 1},
    "carbody": {"convertible": 0, "hardtop": 1, "hatchback": 2, "sedan": 3, "wagon": 4},
    "drivewheel": {"4wd": 0, "fwd": 1, "rwd": 2},
    "enginelocation": {"front": 0, "rear": 1},
    "enginetype": {"dohc": 0, "dohcv": 1, "l": 2, "ohc": 3, "ohcf": 4, "ohcv": 5, "rotor": 6},
    "cylindernumber": {"eight": 0, "five": 1, "four": 2, "six": 3, "three": 4, "twelve": 5, "two": 6},
    "fuelsystem": {"1bbl": 0, "2bbl": 1, "4bbl": 2, "idi": 3, "mfi": 4, "mpfi": 5, "spdi": 6, "spfi": 7},
    "brand": {
        "Nissan": 0, "alfa-romero": 1, "audi": 2, "bmw": 3, "buick": 4, "chevrolet": 5, "dodge": 6, "honda": 7,
        "isuzu": 8, "jaguar": 9, "maxda": 10, "mazda": 11, "mercury": 12, "mitsubishi": 13, "nissan": 14,
        "peugeot": 15, "plymouth": 16, "porcshce": 17, "porsche": 18, "renault": 19, "saab": 20, "subaru": 21,
        "toyota": 22, "toyouta": 23, "vokswagen": 24, "volkswagen": 25, "volvo": 26, "vw": 27,
    },
    "model": {
        "": 0, "100 ls": 1, "100ls": 2, "1131 deluxe sedan": 3, "12tl": 4, "144ea": 5, "145e (sw)": 6,
        "244dl": 7, "245": 8, "246": 9, "264gl": 10, "304": 11, "320i": 12, "4000": 13, "411 (sw)": 14,
        "5 gtl": 15, "5000": 16, "5000s (diesel)": 17, "504": 18, "504 (sw)": 19, "505s turbo diesel": 20,
        "604sl": 21, "626": 22, "99e": 23, "99gle": 24, "99le": 25, "D-Max ": 26, "D-Max V-Cross": 27,
        "MU-X": 28, "Quadrifoglio": 29, "accord": 30, "accord cvcc": 31, "accord lx": 32, "baja": 33, "boxter": 34,
        "brz": 35, "carina": 36, "cayenne": 37, "celica gt": 38, "celica gt liftback": 39, "century": 40,
        "century luxus (sw)": 41, "century special": 42, "challenger se": 43, "civic": 44, "civic (auto)": 45,
        "civic 1300": 46, "civic 1500 gl": 47, "civic cvcc": 48, "clipper": 49, "colt (sw)": 50,
        "colt hardtop": 51, "corolla": 52, "corolla 1200": 53, "corolla 1600 (sw)": 54, "corolla liftback": 55,
        "corolla tercel": 56, "corona": 57, "corona hardtop": 58, "corona liftback": 59, "corona mark ii": 60,
        "coronet custom": 61, "coronet custom (sw)": 62, "cougar": 63, "cressida": 64, "cricket": 65, "d200": 66,
        "dart custom": 67, "dasher": 68, "dayz": 69, "diesel": 70, "dl": 71, "duster": 72, "electra 225 custom": 73,
        "fox": 74, "fuga": 75, "fury gran sedan": 76, "fury iii": 77, "g4": 78, "giulia": 79, "glc": 80, "glc 4": 81,
        "glc custom": 82, "glc custom l": 83, "glc deluxe": 84, "gt-r": 85, "impala": 86, "juke": 87, "kicks": 88,
        "lancer": 89, "latio": 90, "leaf": 91, "macan": 92, "mark ii": 93, "mirage": 94, "mirage g4": 95,
        "model 111": 96, "monaco (sw)": 97, "monte carlo": 98, "montero": 99, "note": 100, "nv200": 101,
        "opel isuzu deluxe": 102, "otti": 103, "outlander": 104, "pajero": 105, "panamera": 106, "prelude": 107,
        "r1": 108, "r2": 109, "rabbit": 110, "rabbit custom": 111, "rampage": 112, "regal sport coupe (turbo)": 113,
        "rogue": 114, "rx-4": 115, "rx-7 gs": 116, "rx2 coupe": 117, "rx3": 118, "satellite custom (sw)": 119,
        "skyhawk": 120, "skylark": 121, "starlet": 122, "stelvio": 123, "super beetle": 124, "teana": 125,
        "tercel": 126, "titan": 127, "trezia": 128, "tribeca": 129, "type 3": 130, "valiant": 131, "vega 2300": 132,
        "versa": 133, "x1": 134, "x3": 135, "x4": 136, "x5": 137, "xf": 138, "xj": 139, "xk": 140, "z4": 141,
    },
}


def preprocess_input(features):
    # Define categorical and numerical columns
    categorical_columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                           'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'brand', 'model']
    numerical_columns = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight',
                         'enginesize', 'boreratio', 'stroke', 'compressionratio', 'horsepower',
                         'peakrpm', 'citympg', 'highwaympg']

    for column, mapping in category_mapping_dict.items():
        features[column] = features[column].map(mapping).fillna(0).astype(int)

    features['power_to_weight_ratio'] = features['horsepower'] / features['curbweight']
    for column in numerical_columns:
        features[f'{column}_squared'] = features[column] ** 2
    features['log_enginesize'] = np.log(features['enginesize'] + 1)

    return features

def predict_price(features):
    input_data = features.copy()
    input_data = preprocess_input(input_data)
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
