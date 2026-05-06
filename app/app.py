import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import joblib

# Load the trained model and feature list
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("final_anaemia_model.cbm")
    features = joblib.load("model_features.pkl")
    return model, features

model, required_features = load_model()

# --- UI Configuration ---
st.set_page_config(page_title="Odisha Anaemia Risk Predictor", page_icon="🩸", layout="centered")

st.title("🩸 Odisha Anaemia Risk Predictor")
st.markdown("""
**Development and Comparative Evaluation of a Cost-Aware Ensemble Machine Learning Framework**
This application predicts the likelihood of Anaemia in women using non-invasive, cost-aware socio-demographic and health indicators, without requiring a blood test.
""")

st.divider()

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Biological Indicators")
    age = st.number_input("Age (Years)", min_value=15, max_value=49, value=30)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    
    st.subheader("Maternal History")
    pregnant = st.selectbox("Currently Pregnant?", ["No", "Yes"])
    pregnant_val = 1 if pregnant == "Yes" else 0
    
    breastfeeding = st.selectbox("Currently Breastfeeding?", ["No", "Yes"])
    breastfeeding_val = 1 if breastfeeding == "Yes" else 0
    
    births_5y = st.number_input("Births in last 5 years", min_value=0, max_value=10, value=0)

with col2:
    st.subheader("Socio-Economic Factors")
    wealth = st.selectbox("Wealth Index", ["Poorest (1)", "Poorer (2)", "Middle (3)", "Richer (4)", "Richest (5)"])
    wealth_val = int(wealth.split("(")[1].split(")")[0])
    
    education = st.selectbox("Education Level", ["No Education (0)", "Primary (1)", "Secondary (2)", "Higher (3)"])
    edu_val = int(education.split("(")[1].split(")")[0])
    
    residence = st.selectbox("Residence Type", ["Urban (1)", "Rural (2)"])
    residence_val = int(residence.split("(")[1].split(")")[0])
    
    insurance = st.selectbox("Covered by Health Insurance", ["No (0)", "Yes (1)"])
    insurance_val = int(insurance.split("(")[1].split(")")[0])
    
    region_val = 21 # Hardcoded for Odisha
    religion_val = 1 # Default Hindu (common in Odisha dataset)

st.subheader("Diet & Lifestyle")
col3, col4 = st.columns(2)
with col3:
    eat_eggs = st.selectbox("Eats Eggs", ["Never (0)", "Occasionally (1)", "Daily (2)"])
    eggs_val = int(eat_eggs.split("(")[1].split(")")[0])
    eat_meat = st.selectbox("Eats Meat", ["Never (0)", "Occasionally (1)", "Daily (2)"])
    meat_val = int(eat_meat.split("(")[1].split(")")[0])

with col4:
    read_news = st.selectbox("Reads Newspaper", ["Not at all (0)", "Less than once a week (1)", "At least once a week (2)"])
    news_val = int(read_news.split("(")[1].split(")")[0])
    listen_radio = st.selectbox("Listens to Radio", ["Not at all (0)", "Less than once a week (1)", "At least once a week (2)"])
    radio_val = int(listen_radio.split("(")[1].split(")")[0])
    watch_tv = st.selectbox("Watches TV", ["Not at all (0)", "Less than once a week (1)", "At least once a week (2)"])
    tv_val = int(watch_tv.split("(")[1].split(")")[0])

st.divider()

# --- Prediction Logic ---
if st.button("Predict Anaemia Risk", type="primary", use_container_width=True):
    # Construct the input dictionary exactly matching the training data columns
    input_data = {
        'v012': age,
        'v445': bmi,
        'v106': edu_val,
        'v190': wealth_val,
        'v213': pregnant_val,
        'v404': breastfeeding_val,
        'v208': births_5y,
        'v024': region_val,
        'v025': residence_val,
        'v130': religion_val,
        'v481': insurance_val,
        'v157': news_val,
        'v158': radio_val,
        'v159': tv_val,
        'v414e': eggs_val,
        'v414f': meat_val
    }
    
    # Ensure columns match required order
    input_df = pd.DataFrame([input_data])[required_features]
    
    # The optimal threshold found by Youden's J Index during our research
    optimal_threshold = 0.65
    
    with st.spinner("Analyzing patient indicators..."):
        probability = model.predict_proba(input_df)[0][1]
        
        # Note: we use our optimal threshold, not standard 0.5
        is_anaemic = probability >= optimal_threshold
        
    if is_anaemic:
        st.error(f"### High Risk of Anaemia Detected")
        st.write(f"Confidence Score: **{(probability * 100):.1f}%** (Exceeds {optimal_threshold*100}% threshold)")
        st.warning("Recommendation: Patient should be referred for a clinical Hemoglobin test.")
    else:
        st.success(f"### Low Risk of Anaemia")
        st.write(f"Risk Score: **{(probability * 100):.1f}%**")
        st.info("Patient statistics align with healthy demographics. Standard nutritional monitoring is advised.")

# --- Footer ---
st.caption("Disclaimer: This tool is a machine learning prototype developed for academic research purposes and is not a substitute for clinical diagnosis.")
