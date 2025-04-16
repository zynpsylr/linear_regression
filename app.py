import streamlit as st
import joblib
import numpy as np

# Başlık    
st.title("💰Maaş Tahmini Uygulaması")
st.write("Lütfen tecrübenizi yıl cinsinden giriniz:")

# Kullanıcıdan bilgi girişi al
experience = st.number_input("Tecrübe (yıl):",min_value=0, max_value=60, value=1, step=1)

# Butona basıldığında tahmin edilecek
if st.button("Hesapla"):
    # Modeli yükle
    model= joblib.load('linear_model.pkl')
  
    # Tahmin yap
    prediction = model.predict(np.array([[experience]]))

    # Sonucu göster
    st.success(f"Tahmini maaş: {prediction[0]:,.2f} ₺ ")