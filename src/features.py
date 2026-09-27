"""Clinical feature metadata.

Names, units and category definitions used to explain each feature in "clinician language".
Core of the XAI demo: report back in clinical terms what the model based its decision on.
"""

# UCI Heart Disease (Cleveland): 13 input features
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "num"  # 0 = no disease, 1-4 = disease -> binarized

# human-readable names
DISPLAY_NAME = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar >120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "Exercise ST slope",
    "ca": "Major vessels (fluoroscopy)",
    "thal": "Thalassemia test",
}

UNIT = {
    "age": " yrs",
    "trestbps": "mmHg",
    "chol": "mg/dL",
    "thalach": " bpm",
    "oldpeak": "",
}

# categorical value -> clinical meaning
CATEGORY_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "fbs": {0: "normal", 1: "high (>120 mg/dL)"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "left ventricular hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}


def humanize(feature: str, value) -> str:
    """Convert a feature value to a clinical expression, e.g. cp=4 -> 'Chest pain type: asymptomatic'."""
    name = DISPLAY_NAME.get(feature, feature)
    if feature in CATEGORY_LABELS:
        label = CATEGORY_LABELS[feature].get(int(round(value)), str(value))
        return f"{name}: {label}"
    unit = UNIT.get(feature, "")
    v = round(float(value), 1)
    return f"{name}: {v}{unit}".rstrip()
