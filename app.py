import streamlit as st
import numpy as np
import pickle
import base64

# --- Page config (MUST BE FIRST Streamlit call) ---
st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model()

# --- Custom CSS for style ---
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #f0f4ff, #e6f7ff);
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2739/2739675.png", width=120)
st.sidebar.title("About")
st.sidebar.markdown("""
This app predicts **California housing prices** using a trained XGBoost model.  
Fill in the values below to estimate the house price 💰
""")

# --- App Header ---
st.markdown("<h2 style='text-align: center;'>🏡 California House Price Prediction</h2>", unsafe_allow_html=True)

st.markdown("### 🔢 Enter House Details")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    MedInc = st.number_input("Median Income", min_value=0.0, format="%.2f", help="In tens of thousands USD")
    HouseAge = st.number_input("House Age", min_value=0.0, max_value=100.0, format="%.1f")
    AveRooms = st.number_input("Average Rooms", min_value=0.0, format="%.2f")
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, format="%.2f")

with col2:
    Population = st.number_input("Population", min_value=0.0, format="%.0f")
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, format="%.2f")
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, format="%.4f")
    Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, format="%.4f")

# --- Prediction ---
if st.button("🔮 Predict House Price"):
    input_data = np.array([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude])
    input_scaled = scaler.transform(input_data.reshape(1, -1))
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.success(f"🏷 **Estimated House Price:** ${prediction * 100000:,.2f}")
    st.markdown("*(Price is in USD)*")
   
