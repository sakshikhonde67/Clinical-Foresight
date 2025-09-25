# Clinical-Foresight
This project is an AI tool that predicts hospital readmission risk for diabetic patients. It uses an interactive dashboard to not only show the risk level but also to clearly explain the specific factors that contribute to it, helping doctors make proactive care decisions.

# 🩺 AI-Powered Diabetes Readmission Prediction

> This project develops an AI-powered predictive system designed to forecast the risk of hospital readmission for diabetic patients within 30 days of discharge. By leveraging a machine learning model trained on a comprehensive clinical dataset, the system not only predicts risk but also provides clear, interpretable explanations for its conclusions using SHAP (SHapley Additive exPlanations). The final solution is delivered as an interactive Streamlit dashboard, enabling clinicians to input patient data and receive an instant risk assessment and a breakdown of the contributing factors, facilitating proactive and data-driven patient care.

---

## 📋 Project Overview

### The Problem 🏥
Hospital readmissions are a major challenge in healthcare, leading to increased costs, strained resources, and often indicating a gap in patient care. For chronic conditions like diabetes, the risk of readmission is particularly high. A key difficulty for healthcare providers is accurately identifying which patients are most at risk before they leave the hospital, as the contributing factors can be numerous and complex.

### The Solution 💡
This project tackles the problem by building a robust, data-driven tool that provides two key functions: **prediction** and **explanation**.

* **Prediction:** At its core, the project uses a Random Forest Classifier, a powerful machine learning algorithm, to analyze dozens of patient attributes—such as age, time in hospital, number of medications, and primary diagnosis—to calculate a precise probability of readmission.

* **Explanation (XAI):** Knowing the "what" (the risk score) is not enough; clinicians need to know the "why." Using the SHAP framework, the system makes the model's "thinking" transparent. It pinpoints exactly which factors are pushing a specific patient's risk up (e.g., "multiple prior inpatient visits") or pulling it down (e.g., "A1C test result is normal"), turning the black-box model into a trustworthy clinical decision-support tool.

### Technical Approach ⚙️
The project was executed in a systematic pipeline:

* **Data Foundation:** The `diabetic_data.csv` dataset was rigorously cleaned. This involved handling missing values, removing irrelevant identifiers, and simplifying complex features like medical diagnosis codes into broader, more meaningful categories.

* **Handling Imbalance:** A critical challenge was the severe class imbalance (far more patients are not readmitted than are). This was solved by implementing **SMOTE** (Synthetic Minority Oversampling Technique), which intelligently creates synthetic examples of at-risk patients in the training data, ensuring the model learns to recognize them effectively.

* **Deployment:** The final, trained model and its SHAP explainer were saved and integrated into a **Streamlit web application**. This dashboard provides an intuitive interface where users can input patient information and receive an immediate, actionable risk analysis.

## 🔑 Key Features

* **🤖 Predictive Modeling:** Utilizes a Random Forest Classifier to predict the likelihood of a patient being readmitted within 30 days.
* **⚖️ Class Balancing:** Implements the SMOTE (Synthetic Minority Oversampling Technique) to address the severe class imbalance inherent in the dataset, leading to a more robust model.
* **🧠 Explainable AI (XAI):** Integrates the SHAP (SHapley Additive exPlanations) library to explain both global feature importance and individual patient predictions.
* **📑 Automated PDF Reports:** Generates comprehensive PDF reports summarizing model performance, high-risk patient lists, and visual explanations.
* **🌐 Interactive Dashboard:** Includes a user-friendly web application built with Streamlit for real-time risk assessment and decision support.

## 📊 Project Workflow

The project follows a structured machine learning lifecycle, broken down into the following stages:

| Stage                        | Process                                                                                                                                | Tools / Libraries Used                                 |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- |
| **1. Data Preparation** | <ul><li>Load the `diabetic_data.csv` dataset.</li><li>Handle missing values (`?`) and drop irrelevant columns.</li></ul>                      | `Pandas`, `NumPy`                                      |
| **2. Feature Engineering** | <ul><li>Map thousands of specific diagnosis codes into broader clinical categories.</li><li>Convert the target variable to binary (1/0).</li></ul> | `Pandas`                                               |
| **3. Preprocessing & Balancing** | <ul><li>Scale numerical features using `StandardScaler`.</li><li>Encode categorical features using `OneHotEncoder`.</li><li>Apply **SMOTE** to the training data to create a balanced dataset.</li></ul> | `Scikit-learn`, `Imblearn`                             |
| **4. Model Training & Evaluation** | <ul><li>Train a Random Forest Classifier on the preprocessed, balanced data.</li><li>Evaluate the model using ROC-AUC, Precision, Recall, and F1-Score.</li></ul> | `Scikit-learn`                                         |
| **5. Explainability (XAI)** | <ul><li>Generate a **global** SHAP summary plot to identify key risk drivers overall.</li><li>Create **local** SHAP waterfall plots to explain predictions for individual high-risk patients.</li></ul> | `SHAP`, `Matplotlib`                                   |
| **6. Reporting & Deployment**| <ul><li>Save trained model, preprocessor, and SHAP explainer using `joblib`.</li><li>Generate a summary PDF report.</li><li>Serve the model via an interactive **Streamlit** dashboard.</li></ul> | `FPDF2`, `Joblib`, `Streamlit` |

## 💻 Tech Stack

* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, Imblearn
* **Explainable AI (XAI):** SHAP
* **Data Visualization:** Matplotlib, Seaborn
* **Web App/Dashboard:** Streamlit
* **PDF Reporting:** FPDF2
* **Serialization:** Joblib

## 🚀 Setup and Installation

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/sakshikhonde67/Clinical-Foresight.git]
    cd your-repo-name
    ```

2.  **Create a Python Virtual Environment**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    All required libraries are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ How to Run

The project has two main components that can be executed.

### 1. Run the Full ML Pipeline

This script will perform all steps from data cleaning to model training, evaluation, and generating the PDF report. It will also save the necessary model artifacts (`.pkl` files) for the web app.

```bash
python main.py
```
After execution, a `Diabetes_Readmission_Report.pdf` will be generated in the root directory.

### 2. Launch the Interactive Dashboard

This command will start the Streamlit web server and open the interactive dashboard in your browser.

```bash
streamlit run app.py
```

## 📂 Project File Structure

```
Diabetes_Project/
|
|-- venv/                     # Virtual environment folder
|-- diabetic_data.csv         # The raw dataset
|-- model_pipeline.pkl        # Saved model for the app
|-- shap_data.pkl             # Saved SHAP explainer for the app
|-- model_columns.pkl         # Saved model column order
|
|-- app.py                    # Script for the Streamlit dashboard
|-- config.py                 # Configuration file (mappings, etc.)
|-- main.py                   # Main script to run the entire ML pipeline
|-- reporting.py              # Module for generating the PDF report
|-- requirements.txt          # List of required Python libraries
|-- README.md                 # Project documentation (this file)
|
|-- shap_summary_plot.png     # Generated SHAP plot (example)
|-- Diabetes_Readmission_Report.pdf # Generated PDF report (example)
```

## 💡 Future Improvements

-   **Hyperparameter Tuning:** Implement `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the Random Forest model.
-   **Model Calibration:** Use techniques like Isotonic Regression or Platt Scaling to ensure the model's predicted probabilities are well-calibrated.
-   **Advanced Feature Engineering:** Explore interactions between features and create more sophisticated predictors.
-   **Alternative Models:** Experiment with other models like Gradient Boosting (XGBoost, LightGBM) or a simple Neural Network to compare performance.

---
