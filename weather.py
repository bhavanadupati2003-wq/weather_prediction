import streamlit as st
import pandas as pd
import joblib

# Load model & encoder
model = joblib.load("weather_classification_model.pkl")
label_encoder = joblib.load("weather_label_encoder.pkl")

st.set_page_config(page_title="Weather Prediction App", layout="centered")

st.title("🌦️ Weather Classification App")
st.write("Predict weather type based on input conditions")

# Input fields
precipitation = st.slider("🌧️ Precipitation", 0.0, 50.0, 0.0)
temp_max = st.slider("🌡️ Max Temperature (°C)", -10.0, 50.0, 25.0)
temp_min = st.slider("❄️ Min Temperature (°C)", -20.0, 40.0, 10.0)
wind = st.slider("💨 Wind Speed", 0.0, 20.0, 3.0)

# Predict button
if st.button("🔮 Predict Weather"):
    input_data = pd.DataFrame({
        "precipitation": [precipitation],
        "temp_max": [temp_max],
        "temp_min": [temp_min],
        "wind": [wind]
    })

    prediction = model.predict(input_data)
    weather = label_encoder.inverse_transform(prediction)[0]

    # Weather icons & colors
    weather_display = {
        "sun": ("☀️", "orange"),
        "drizzle": ("🌦️", "blue"),
        "Rain": ("🌧️", "green"),
        "Snow": ("❄️", "skyblue")
    }

    icon, color = weather_display.get(weather, ("🌤️", "black"))

    st.markdown(
        f"""
        <h2 style='color:{color}; text-align:center;'>
            {icon} {weather}
        </h2>
        """,
        unsafe_allow_html=True
    )
