# app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Readmission Prediction",
    page_icon="🩺",
    layout="wide"
)

# Use caching to load model and data only once
@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model pipeline and SHAP data."""
    model_pipeline = joblib.load('model_pipeline.pkl')
    shap_data = joblib.load('shap_data.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model_pipeline, shap_data, model_columns

model_pipeline, shap_data, model_columns = load_artifacts()
explainer = shap_data['explainer']
shap_expected_value = shap_data['expected_value']
shap_feature_names = shap_data['feature_names']


# --- App Title and Description ---
st.title("🩺 AI-Powered Diabetes Readmission Prediction")
st.markdown("""
This tool predicts the risk of a diabetic patient being readmitted to the hospital within 30 days.
Enter the patient's details in the sidebar to get a risk score and an explanation of the key contributing factors.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Patient Details")

def get_user_input():
    """Creates sidebar widgets and returns user inputs as a DataFrame."""
    # These values are based on the unique values from the original dataset
    # In a real-world scenario, these would come from a configuration file or database
    race_options = ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']
    gender_options = ['Female', 'Male']
    age_options = ['[70-80)', '[60-70)', '[80-90)', '[50-60)', '[40-50)', '[30-40)', '[90-100)', '[20-30)', '[10-20)', '[0-10)']
    diag_options = ['Circulatory', 'Other', 'Respiratory', 'Digestive', 'Diabetes', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms']
    
    # Input fields
    race = st.sidebar.selectbox("Race", race_options)
    gender = st.sidebar.selectbox("Gender", gender_options)
    age = st.sidebar.selectbox("Age Group", age_options)
    time_in_hospital = st.sidebar.slider("Time in Hospital (Days)", 1, 14, 3)
    num_lab_procedures = st.sidebar.slider("Number of Lab Procedures", 1, 132, 45)
    num_medications = st.sidebar.slider("Number of Medications", 1, 81, 16)
    number_inpatient = st.sidebar.slider("Number of Inpatient Visits (Prev. Year)", 0, 21, 0)
    number_emergency = st.sidebar.slider("Number of Emergency Visits (Prev. Year)", 0, 76, 0)
    number_outpatient = st.sidebar.slider("Number of Outpatient Visits (Prev. Year)", 0, 42, 0)
    diag_1 = st.sidebar.selectbox("Primary Diagnosis Category", diag_options)

    # Create a dictionary of the inputs
    input_dict = {
        'race': race,
        'gender': gender,
        'age': age,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient,
        'diag_1': diag_1
    }

    # Create a DataFrame from the dictionary, ensuring column order matches the model
    # We add placeholders for the other columns the model expects, even if we don't have inputs for them
    # In a real app, you would have inputs for all relevant features
    
    # Start with a dictionary of NaNs for all model columns
    full_input_dict = {col: [None] for col in model_columns}

    # Update with user inputs
    for key, value in input_dict.items():
        if key in full_input_dict:
            full_input_dict[key] = [value]
    
    return pd.DataFrame.from_dict(full_input_dict)


input_df = get_user_input()

# --- Prediction and Explanation ---
st.header("Prediction Results")

# Display the user's selected data
st.write("**Patient Input Data:**")
st.dataframe(input_df)

if st.sidebar.button("Predict Readmission Risk", type="primary"):
    # Make prediction
    probability = model_pipeline.predict_proba(input_df)[:, 1][0]
    
    # Display prediction with a gauge
    st.subheader("Readmission Risk Score")
    if probability > 0.5:
        st.metric(label="Risk", value=f"{probability:.1%}", delta="High Risk", delta_color="inverse")
    else:
        st.metric(label="Risk", value=f"{probability:.1%}", delta="Low Risk", delta_color="normal")
        
    st.progress(probability)
    st.markdown(f"""
    The model predicts a **{probability:.1%} probability** of this patient being readmitted within 30 days.
    """)
    
    # --- SHAP Explanation ---
    st.subheader("Explanation of Prediction (SHAP Analysis)")
    
    # We need to transform the input data just like we did for training
    input_transformed = model_pipeline.named_steps['preprocessor'].transform(input_df)
    
    # Calculate SHAP values for the single prediction
    shap_values = explainer.shap_values(input_transformed)
    
    # Create the SHAP explanation object for the positive class (readmission)
    explanation = shap.Explanation(
        values=shap_values[1][0],
        base_values=shap_expected_value,
        data=input_transformed[0],
        feature_names=shap_feature_names
    )
    
    # Create and display the waterfall plot
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(explanation, max_display=10, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("""
    **How to read this chart:** The plot above shows the factors that contributed to the final prediction.
    - **Red bars** represent features that *increased* the predicted risk of readmission.
    - **Blue bars** represent features that *decreased* the risk.
    - The `E[f(x)]` value at the bottom is the baseline prediction. Each feature's impact pushes the prediction from this baseline to the final score `f(x)` at the top.
    """)
else:
    st.info("Click the 'Predict Readmission Risk' button in the sidebar to see the results.")