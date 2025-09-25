# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Import configurations and reporting function
from config import DIAGNOSIS_MAP, COLS_TO_DROP, TARGET_MAP
from reporting import create_pdf_report

print("Project Execution Started...")

# --- 1. Data Loading and Initial Cleaning ---
print("Step 1: Loading and cleaning data...")
df = pd.read_csv('diabetic_data.csv')

# Replace '?' with NaN for proper missing value handling
df.replace('?', np.nan, inplace=True)

# Drop columns with too many missing values or identifiers
df.drop(columns=COLS_TO_DROP, inplace=True)

# Drop rows with missing race, gender, or primary diagnosis
df.dropna(subset=['race', 'gender', 'diag_1'], inplace=True)

# Remove 'Unknown/Invalid' gender entries
df = df[df['gender'] != 'Unknown/Invalid']

# --- 2. Feature Engineering ---
print("Step 2: Performing feature engineering...")

# Map diagnosis codes to categories
def map_diagnosis(diag_code):
    if pd.isna(diag_code):
        return 'Other'
    try:
        # Handle codes like '250.83' by taking the integer part
        code = float(diag_code)
    except (ValueError, TypeError):
        # For non-numeric codes (like 'V57'), classify as Other
        return 'Other'

    for category, code_range in DIAGNOSIS_MAP.items():
        if int(code) in code_range:
            return category
    return 'Other'

df['diag_1'] = df['diag_1'].apply(map_diagnosis)

# Binarize the target variable 'readmitted'
df['readmitted'] = df['readmitted'].map(TARGET_MAP)

# --- 3. Preprocessing Setup ---
print("Step 3: Setting up preprocessing pipelines...")

# Explicitly define categorical and numerical features after feature engineering
categorical_features = [
    'race', 'gender', 'diag_1', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
    'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
    'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin',
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed'
]
numerical_features = [
    'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
    'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses'
]

# Create a preprocessing pipeline
# OneHotEncoder handles categorical variables, StandardScaler scales numerical ones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- 4. Data Splitting ---
print("Step 4: Splitting data into training and testing sets...")
X = df.drop('readmitted', axis=1)
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Model Training ---
print("Step 5: Training predictive models...")

# Define the Random Forest model with class weights for imbalance
rf_model = RandomForestClassifier(random_state=42, n_estimators=150, class_weight='balanced', max_depth=10)

# Create the full pipeline including SMOTE for oversampling the minority class
# SMOTE is applied ONLY to the training data to prevent data leakage
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

# Train the model
model_pipeline.fit(X_train, y_train)

# --- 6. Model Evaluation ---
print("Step 6: Evaluating model performance...")
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

# Print key metrics
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Store metrics for the report
metrics = {
    "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
    "report_dict": classification_report(y_test, y_pred, output_dict=True)
}

# --- 7. Explainability (XAI) with SHAP ---
print("Step 7: Generating SHAP explanations...")

# Create an inference pipeline (preprocessor + classifier only)
inference_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])
inference_pipeline.fit(X_train, y_train)  # Fit on training data

# Transform test data
X_test_transformed = preprocessor.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

# Use TreeExplainer for RandomForestClassifier
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_transformed)  # shape: (n_classes, n_samples, n_features) or (n_samples, n_features)

# Get feature names after preprocessing
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names = np.concatenate([numerical_features, cat_feature_names])

# Print shapes for debugging
print("type(shap_values):", type(shap_values))
if isinstance(shap_values, list):
    print("shap_values[1] shape:", np.array(shap_values[1]).shape)
else:
    print("shap_values shape:", np.array(shap_values).shape)
print("X_test_transformed shape:", X_test_transformed.shape)
print("feature_names length:", len(feature_names))

# Use the correct SHAP values for class 1
if isinstance(shap_values, list):
    shap_summary = shap_values[1]  # shape: (n_samples, n_features)
else:
    # If shap_values is an ndarray with shape (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        shap_summary = shap_values[:, :, 1]  # class 1
    else:
        shap_summary = shap_values  # shape: (n_samples, n_features)

# Generate and save SHAP summary plot for class 1 (readmitted)
plt.figure()
shap.summary_plot(shap_summary, X_test_transformed, feature_names=feature_names, show=False, max_display=15)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.close()
print("SHAP summary plot saved as shap_summary_plot.png")

# --- 8. Identify High-Risk Patients and Generate Report ---
print("Step 8: Identifying high-risk patients and generating PDF report...")

# Create a DataFrame with test results and predictions
results_df = X_test.copy()
results_df['true_readmitted'] = y_test
results_df['predicted_probability'] = y_pred_proba

# Identify top 10 patients with the highest predicted readmission risk
top_10_high_risk = results_df.sort_values(by='predicted_probability', ascending=False).head(10)

## Generate and save individual SHAP plots for the top 3 high-risk patients
for i in range(1, 4):  # i = 1, 2, 3 for shap_patient_1.png, shap_patient_2.png, shap_patient_3.png
    patient_index = top_10_high_risk.index[i-1]
    loc_in_X_test = X_test.index.get_loc(patient_index)
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_summary[loc_in_X_test],
            base_values=explainer.expected_value[1],
            data=X_test_transformed[loc_in_X_test],
            feature_names=feature_names
        ),
        show=False,
        max_display=10
    )
    plt.tight_layout()
    plt.savefig(f'shap_patient_{i}.png')
    plt.close()

# Create the final PDF report

create_pdf_report(top_10_high_risk, metrics)
print("Step 9: Saving model and SHAP explainer for deployment...")

# Save the entire model pipeline
joblib.dump(model_pipeline, 'model_pipeline.pkl')

# Save the SHAP explainer and feature names
# We wrap them in a dictionary for easy loading
shap_data = {
    'explainer': explainer,
    'expected_value': explainer.expected_value[1],
    'feature_names': feature_names
}
joblib.dump(shap_data, 'shap_data.pkl')

# Also save the columns in the order the model expects them
joblib.dump(list(X.columns), 'model_columns.pkl')

print("Model and supporting artifacts saved successfully.")


print("\nProject Execution Finished Successfully!")
print("Check the generated file: 'Diabetes_Readmission_Report.pdf'")