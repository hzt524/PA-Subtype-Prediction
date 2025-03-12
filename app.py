import streamlit as st
import xgboost as xgb
import pandas as pd

model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

features = ["Hypertension grade", "Lowest potassium", "Upright ARR", 
            "Supine to Upright Aldo change", "Nodule at imaging"]

st.title("PA Subtype Prediction")
hypertension_grade = st.selectbox("Hypertension grade", [1, 2, 3]) 
lowest_potassium = st.selectbox("Lowest potassium (0:>3.5mmol/L, 1:3~3.5mmol/L, 2:<3.0mmol/L)", [0, 1, 2])
upright_arr = st.number_input("Upright ARR (ng/dL)/(mU/L)", min_value=0.0, max_value=500.0)
supine_aldosterone_change = st.number_input("Supine to Upright Aldo change", min_value=-1.0, max_value=50.0)
nodule_at_imaging = st.selectbox("Nodule at imaging (0=No, 1=Yes)", [0, 1]) 

input_data = pd.DataFrame({
    "Hypertension grade": [hypertension_grade],
    "Lowest potassium": [lowest_potassium],
    "Upright ARR": [upright_arr],
    "Supine to Upright Aldo change": [supine_aldosterone_change],
    "Nodule at imaging": [nodule_at_imaging]
})

input_data = input_data[features]
probabilities = model.predict_proba(input_data)
lateralization_probability = probabilities[0][1]
st.write(f"UPA probability：{lateralization_probability:.4f}")

