import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="V2 Access-Aware Anaemia Triage", page_icon="🚨", layout="wide")

# --- Load Model Bundle ---
@st.cache_resource
def load_v2_model():
    import os
    model_path = os.path.join(os.path.dirname(__file__), "production_model_v2.pkl")
    bundle = joblib.load(model_path)
    return bundle

bundle = load_v2_model()
base_rfs = bundle['base_rfs']
base_anns = bundle['base_anns']
base_lgbms = bundle['base_lgbms']
meta_learner = bundle['meta_learner']
threshold = bundle['optimal_threshold']
features = bundle['feature_names']

# Pre-calculate SHAP explainer for the first RF model (as a proxy for feature importance)
@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(base_rfs[0])
explainer = get_explainer()

# --- Personas (Quick Load) ---
PERSONAS = {
    "Healthy Urban Professional": {
        'v012': 28, 'v438': 160.0, 'v445': 22.5, 'v106': 3, 'v190': 5,
        'v213': 0, 'v404': 0, 'v208': 0, 'v201': 0, 'v501': 1, 'v312': 3,
        'v467b': 2, 'v467d': 2, 'v463z': 1, 'v113': 11, 'v116': 11, 'v161': 2,
        'v136': 3, 'v137': 0, 'v414e': 2, 'v414f': 2, 'v157': 2, 'v158': 2, 'v159': 2,
        'v481': 1, 'v130': 1, 'v025': 1, 'v024': 21
    },
    "High-Risk Rural Mother": {
        'v012': 24, 'v438': 150.0, 'v445': 17.5, 'v106': 0, 'v190': 1,
        'v213': 1, 'v404': 1, 'v208': 2, 'v201': 3, 'v501': 1, 'v312': 0,
        'v467b': 1, 'v467d': 1, 'v463z': 0, 'v113': 41, 'v116': 31, 'v161': 6,
        'v136': 7, 'v137': 2, 'v414e': 0, 'v414f': 0, 'v157': 0, 'v158': 0, 'v159': 0,
        'v481': 0, 'v130': 1, 'v025': 2, 'v024': 21
    },
    "Moderate-Risk Villager": {
        'v012': 35, 'v438': 155.0, 'v445': 19.0, 'v106': 1, 'v190': 2,
        'v213': 0, 'v404': 0, 'v208': 0, 'v201': 4, 'v501': 1, 'v312': 1,
        'v467b': 1, 'v467d': 2, 'v463z': 1, 'v113': 21, 'v116': 21, 'v161': 6,
        'v136': 5, 'v137': 1, 'v414e': 1, 'v414f': 0, 'v157': 0, 'v158': 1, 'v159': 1,
        'v481': 1, 'v130': 1, 'v025': 2, 'v024': 21
    }
}

# Initialize session state with default
if 'input_data' not in st.session_state:
    st.session_state.input_data = PERSONAS["Healthy Urban Professional"].copy()

def load_persona(name):
    st.session_state.input_data = PERSONAS[name].copy()

# --- UI Header ---
st.title("🚨 Access-Aware Severe Anaemia Triage (V2)")
st.markdown("""
**Version 2.0 Upgrade:** This tool is designed for ASHA workers to pre-screen women in Odisha for **Moderate-to-Severe Anaemia** using 32 non-invasive demographic and socioeconomic indicators. 
By focusing on *Access Gaps* rather than just *Cost*, this system identifies vulnerable women who face geographic or financial barriers to clinical testing.
""")

st.divider()

# --- Persona Buttons ---
st.subheader("👥 Quick Load Patient Personas")
col1, col2, col3 = st.columns(3)
with col1:
    st.button("🏥 Healthy Urban Professional", on_click=load_persona, args=("Healthy Urban Professional",), use_container_width=True)
with col2:
    st.button("⚠️ Moderate-Risk Villager", on_click=load_persona, args=("Moderate-Risk Villager",), use_container_width=True)
with col3:
    st.button("🚨 High-Risk Rural Mother", on_click=load_persona, args=("High-Risk Rural Mother",), use_container_width=True)

st.divider()

# --- Form Inputs ---
# We link the inputs to session_state so buttons can update them
st.subheader("📋 Patient Information")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.session_state.input_data['v012'] = st.number_input("Age", min_value=15, max_value=49, value=st.session_state.input_data['v012'])
    st.session_state.input_data['v445'] = st.number_input("BMI", min_value=10.0, max_value=50.0, value=st.session_state.input_data['v445'])
    st.session_state.input_data['v438'] = st.number_input("Height (cm)", min_value=120.0, max_value=190.0, value=st.session_state.input_data['v438'])
    
with c2:
    st.session_state.input_data['v190'] = st.selectbox("Wealth Index", [1, 2, 3, 4, 5], index=st.session_state.input_data['v190']-1)
    st.session_state.input_data['v106'] = st.selectbox("Education Level", [0, 1, 2, 3], index=st.session_state.input_data['v106'])
    st.session_state.input_data['v025'] = st.selectbox("Residence (1=Urban, 2=Rural)", [1, 2], index=st.session_state.input_data['v025']-1)

with c3:
    st.session_state.input_data['v213'] = st.selectbox("Pregnant?", [0, 1], index=st.session_state.input_data['v213'])
    st.session_state.input_data['v404'] = st.selectbox("Breastfeeding?", [0, 1], index=st.session_state.input_data['v404'])
    st.session_state.input_data['v201'] = st.number_input("Total Children", value=st.session_state.input_data['v201'])

with c4:
    st.session_state.input_data['v467b'] = st.selectbox("Distance to Hospital (1=Far, 2=Near)", [1, 2], index=st.session_state.input_data['v467b']-1)
    st.session_state.input_data['v414e'] = st.selectbox("Eats Eggs (0=No, 2=Daily)", [0, 1, 2], index=st.session_state.input_data['v414e'])
    st.session_state.input_data['v414f'] = st.selectbox("Eats Meat (0=No, 2=Daily)", [0, 1, 2], index=st.session_state.input_data['v414f'])

st.divider()

# --- Human-Readable Feature Name Mapping ---
FEATURE_NAMES = {
    'v012': 'Age', 'v445': 'Body Mass Index (BMI)', 'v106': 'Education Level',
    'v190': 'Wealth Index', 'v213': 'Currently Pregnant', 'v404': 'Breastfeeding',
    'v208': 'Births (Last 5 Years)', 'v024': 'State (Odisha)', 'v025': 'Residence (Urban/Rural)',
    'v130': 'Religion', 'v481': 'Health Insurance', 'v157': 'Reads Newspaper',
    'v158': 'Listens to Radio', 'v159': 'Watches TV', 'v414e': 'Egg Consumption',
    'v414f': 'Meat Consumption', 'v113': 'Drinking Water Source', 'v116': 'Toilet Facility',
    'v161': 'Cooking Fuel Type', 'v136': 'Household Size', 'v137': 'Children Under 5',
    'v201': 'Total Children Born', 'v501': 'Marital Status', 'v312': 'Contraceptive Use',
    'v467b': 'Distance to Hospital', 'v467d': 'Hospital Access Difficulty',
    'v463z': 'No Tobacco Use', 'v438': 'Height (cm)',
    'Shield_Score': 'Socio-Nutritional Shield', 'BMI_Age_Interaction': 'BMI x Age Factor',
    'Maternal_Burden': 'Maternal Burden Score', 'Healthcare_Barrier': 'Healthcare Access Barrier',
}

# --- Prediction & Explainability ---
if st.button("Evaluate Patient Risk", type="primary", use_container_width=True):
    with st.spinner("Processing Stacking Ensemble..."):
        # 1. Prepare raw data
        raw_input = st.session_state.input_data.copy()
        
        # 2. Feature Engineering (Match Preprocessing script exactly)
        raw_input['Shield_Score'] = ((raw_input['v190'] - 1) / 4 * 0.35 + (raw_input['v106']) / 3 * 0.25 + raw_input['v414e'] * 0.20 + raw_input['v414f'] * 0.20)
        raw_input['BMI_Age_Interaction'] = raw_input['v445'] * raw_input['v012'] / 1000
        raw_input['Maternal_Burden'] = (raw_input['v201'] + raw_input['v208']) / max(raw_input['v012'], 15)
        raw_input['Healthcare_Barrier'] = raw_input['v467b'] + raw_input['v467d']
        
        # Create DataFrame matching feature order
        input_df = pd.DataFrame([raw_input])[features]
        
        # 3. Get Base Learner Probabilities (pass .values to avoid sklearn warnings)
        meta_features = np.zeros((1, 3))
        for i in range(5):
            meta_features[0, 0] += base_rfs[i].predict_proba(input_df.values)[0][1] / 5
            meta_features[0, 1] += base_anns[i].predict_proba(input_df.values)[0][1] / 5
            meta_features[0, 2] += base_lgbms[i].predict_proba(input_df.values)[0][1] / 5
            
        # 4. Get Final Meta Prediction
        final_prob = meta_learner.predict_proba(meta_features)[0][1]
        is_high_risk = final_prob >= threshold

        # 5. Generate SHAP explanation
        shap_values = explainer(input_df)
        sv = shap_values[0, :, 1]  # positive class SHAP values

        # Build sorted list of (feature_name, shap_value, raw_value)
        shap_items = []
        for i, col in enumerate(features):
            friendly = FEATURE_NAMES.get(col, col)
            shap_items.append((friendly, sv.values[i], input_df.iloc[0, i]))
        shap_items.sort(key=lambda x: abs(x[1]), reverse=True)

        risk_factors = [(n, v, r) for n, v, r in shap_items if v > 0.001]
        protective_factors = [(n, v, r) for n, v, r in shap_items if v < -0.001]

        # --- Display Results ---
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if is_high_risk:
                st.error("### HIGH RISK")
                st.metric("Risk Score", f"{final_prob*100:.1f}%", delta=f"+{(final_prob - threshold)*100:.1f}% above threshold", delta_color="inverse")
                st.markdown("---")
                st.warning("**Recommendation:** Refer for hemoglobin blood test. This patient shows demographic indicators associated with moderate-to-severe anaemia.")
            else:
                st.success("### LOW RISK")
                st.metric("Risk Score", f"{final_prob*100:.1f}%", delta=f"{(final_prob - threshold)*100:.1f}% below threshold", delta_color="normal")
                st.markdown("---")
                st.info("**Recommendation:** Routine monitoring. Patient demographics do not indicate elevated risk for severe anaemia.")

            st.markdown("---")
            st.caption(f"Triage Threshold: {threshold*100:.1f}% | Model AUC: {bundle.get('test_auc', 0.585):.3f}")

        with res_col2:
            st.subheader("Why did the AI make this decision?")

            # --- Risk Factors (Red) ---
            if risk_factors:
                st.markdown("**Risk Factors** (increasing anaemia likelihood)")
                for name, shap_val, raw_val in risk_factors[:5]:
                    impact_pct = abs(shap_val) * 100
                    bar_width = min(impact_pct / 8 * 100, 100)  # Scale for visual
                    st.markdown(
                        f'<div style="margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:14px;">'
                        f'<span><b>{name}</b> = {raw_val:.1f}</span>'
                        f'<span style="color:#ff4b4b;">+{impact_pct:.1f}%</span></div>'
                        f'<div style="background:#ffdddd;border-radius:4px;height:8px;width:100%;">'
                        f'<div style="background:#ff4b4b;border-radius:4px;height:8px;width:{bar_width}%;"></div>'
                        f'</div></div>', unsafe_allow_html=True
                    )
            else:
                st.write("No significant risk factors identified.")

            st.markdown("")

            # --- Protective Factors (Green) ---
            if protective_factors:
                st.markdown("**Protective Factors** (reducing anaemia likelihood)")
                for name, shap_val, raw_val in protective_factors[:5]:
                    impact_pct = abs(shap_val) * 100
                    bar_width = min(impact_pct / 8 * 100, 100)
                    st.markdown(
                        f'<div style="margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:14px;">'
                        f'<span><b>{name}</b> = {raw_val:.1f}</span>'
                        f'<span style="color:#21c354;">-{impact_pct:.1f}%</span></div>'
                        f'<div style="background:#ddffdd;border-radius:4px;height:8px;width:100%;">'
                        f'<div style="background:#21c354;border-radius:4px;height:8px;width:{bar_width}%;"></div>'
                        f'</div></div>', unsafe_allow_html=True
                    )
            else:
                st.write("No significant protective factors identified.")

        # --- Technical SHAP chart in collapsible section for the panel ---
        with st.expander("Technical View: Raw SHAP Waterfall Chart (for researchers)"):
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(sv, show=False)
            st.pyplot(fig)
