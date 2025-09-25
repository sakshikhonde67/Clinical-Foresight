# config.py

# This file contains configuration variables and mappings for the project.

# Mapping of ICD-9 diagnosis codes to broader categories.
# This simplifies the feature by grouping thousands of specific codes.
DIAGNOSIS_MAP = {
    'Circulatory': (list(range(390, 460)) + [785]),
    'Respiratory': (list(range(460, 520)) + [786]),
    'Digestive': (list(range(520, 580)) + [787]),
    'Diabetes': ([250]),
    'Injury': (list(range(800, 1000))),
    'Musculoskeletal': (list(range(710, 740))),
    'Genitourinary': (list(range(580, 630)) + [788]),
    'Neoplasms': (list(range(140, 240))),
    'Other': list(range(1, 140)) + list(range(241, 250)) + list(range(251, 389)) + list(range(631, 709)) + list(range(741, 784)) + list(range(790, 800))
}

# List of columns to be dropped due to high missing values or lack of predictive power.
COLS_TO_DROP = [
    'encounter_id',
    'patient_nbr',
    'weight',
    'payer_code',
    'medical_specialty',
    'diag_2', # We focus on the primary diagnosis for simplicity
    'diag_3'
]

# Defines the target variable for binary classification.
# '<30' is our positive class (readmitted).
TARGET_MAP = {'<30': 1, '>30': 0, 'NO': 0}