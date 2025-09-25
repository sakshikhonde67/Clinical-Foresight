# reporting.py

from fpdf import FPDF

# A simple function to add a section title to the PDF
def add_section_title(pdf, title):
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, title, 0, 1, 'L')
    pdf.ln(5)

# A simple function to add body text to the PDF
def add_body_text(pdf, text):
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 5, text)
    pdf.ln(5)

def create_pdf_report(top_10_patients, metrics):
    """Generates a comprehensive PDF report of the project findings."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Header ---
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, 'Diabetes Readmission Risk Prediction Report', 0, 1, 'C')
    pdf.ln(10)

    # --- Model Performance Section ---
    add_section_title(pdf, '1. Model Performance KPIs')
    roc_auc = metrics['ROC-AUC']
    report_dict = metrics['report_dict']
    performance_text = (
        f"The Random Forest model was evaluated on the unseen test set.\n\n"
        f"-> ROC-AUC Score: {roc_auc:.4f}\n\n"
        f"Class 1 (Readmitted) Performance:\n"
        f"  - Precision: {report_dict['1']['precision']:.2f}\n"
        f"  - Recall: {report_dict['1']['recall']:.2f}\n"
        f"  - F1-Score: {report_dict['1']['f1-score']:.2f}"
    )
    add_body_text(pdf, performance_text)
    pdf.ln(10)

    # --- High-Risk Patients Section ---
    add_section_title(pdf, '2. Top 10 High-Risk Patients')
    add_body_text(pdf, "The table below lists the 10 patients with the highest predicted probability of readmission.")

    pdf.set_font('Arial', '', 9)
    # Table Header
    pdf.cell(20, 10, 'Prob (%)', 1)
    pdf.cell(20, 10, 'Age', 1)
    pdf.cell(50, 10, 'Diagnosis', 1)
    pdf.cell(30, 10, '# Inpatient', 1)
    pdf.cell(30, 10, '# Meds', 1)
    pdf.ln()

    # Table Rows
    for index, row in top_10_patients.iterrows():
        prob_percent = f"{row['predicted_probability']*100:.1f}"
        pdf.cell(20, 10, prob_percent, 1)
        pdf.cell(20, 10, str(row['age']), 1)
        pdf.cell(50, 10, str(row['diag_1']), 1)
        pdf.cell(30, 10, str(row['number_inpatient']), 1)
        pdf.cell(30, 10, str(row['num_medications']), 1)
        pdf.ln()
    pdf.ln(10)

    # --- Global Feature Importance Section ---
    add_section_title(pdf, '3. Global Feature Importance (SHAP)')
    add_body_text(pdf, "This plot shows the top features that impact the model's predictions across all patients. Longer bars indicate greater importance.")
    # Add the SHAP summary plot image
    pdf.image('shap_summary_plot.png', x=10, w=180)
    pdf.ln(5)

    # --- Local Explanations Section ---
    pdf.add_page()
    add_section_title(pdf, '4. Local Patient-Specific Explanations')
    add_body_text(pdf, "The following waterfall plots explain the prediction for the top 3 high-risk patients. Red bars increase risk, blue bars decrease it.")

    # Add individual SHAP plots
    for i in range(1, 4):
        pdf.image(f'shap_patient_{i}.png', x=10, w=180)
        pdf.ln(5)

    # Save the PDF
    pdf.output('Diabetes_Readmission_Report.pdf')