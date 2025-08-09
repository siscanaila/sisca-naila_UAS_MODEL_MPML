# men_shoes_streamlit.py
import streamlit as st
import pandas as pd
from joblib import load
import seaborn as sns

# Set custom title and icon
st.set_page_config(page_title="Prediksi Penjualan Sepatu", page_icon="👟", layout="wide")

# Load dataset dan model
# Correct file paths (using escape character or forward slashes)
df = pd.read_csv('D:/KULIAH/SEMESTER 4/Machine Learning/UAS/men_shoes/MEN_SHOES.csv')
model = load('D:/KULIAH/SEMESTER 4/Machine Learning/UAS/men_shoes/random_forest_regressor.joblib')


# UI Streamlit
st.title("Prediksi Penjualan Sepatu")
st.write("Aplikasi ini memprediksi jumlah sepatu yang terjual berdasarkan harga saat ini.")

# Tampilkan dataset
if st.checkbox('Tampilkan Dataset'):
    st.write(df.head())

# Input untuk prediksi
harga_input = st.number_input('Masukkan Harga Sepatu', min_value=0, max_value=10000)

# Prediksi saat tombol ditekan
if st.button('Prediksi Penjualan'):
    prediksi = model.predict([[harga_input]])
    st.write(f"Prediksi Penjualan: {prediksi[0]}")
