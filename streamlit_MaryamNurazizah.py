import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model, scaler, dan label encoder
model = tf.keras.models.load_model("voice_gender_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Judul aplikasi
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🎤 Voice Gender Predictor 🔮</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666;'>Masukkan data ciri suara & kita tebak kamu cowok atau cewek 😎</h4>", unsafe_allow_html=True)
st.markdown("---")

# Daftar fitur sesuai training
feature_names = [
    "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt",
    "sp.ent", "sfm", "mode", "centroid", "meanfun", "minfun", "maxfun",
    "meandom", "mindom", "maxdom", "dfrange", "modindx"
]

# Input fitur dengan 2 kolom
st.markdown("### 🎛️ Input Ciri-ciri Suara")
col1, col2 = st.columns(2)
input_data = []

for i, feature in enumerate(feature_names):
    with (col1 if i % 2 == 0 else col2):
        value = st.number_input(f"{feature}", value=0.0, format="%.6f")
        input_data.append(value)

# Tombol prediksi
if st.button("🔍 Prediksi Sekarang"):
    # Buat DataFrame dengan nama kolom yang benar
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Scaling fitur
    input_scaled = scaler.transform(input_df)

    # Prediksi dengan model
    prediction = model.predict(input_scaled)

    # Ambil kelas prediksi dari softmax output
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Mapping label
    mapped_label = label_encoder.inverse_transform([predicted_class])[0]

    st.write("🎯 Kelas Prediksi (angka):", predicted_class)

    # Emoji sesuai gender
    emoji = "👧🏻" if mapped_label == "female" else "👦🏻"

    # Tampilkan hasil prediksi
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>Hasil Prediksi:</h2>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #00B386;'>{emoji} {mapped_label.capitalize()}!</h1>", unsafe_allow_html=True)
    st.balloons()
    
    