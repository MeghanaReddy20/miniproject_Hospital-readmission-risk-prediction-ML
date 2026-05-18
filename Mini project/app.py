from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

app = Flask(__name__)

# COMPREHENSIVE PATIENT DATA
DATA = {
    "patient_name": "Johnathan Smith",
    "condition": "Post-Myocardial Infarction (Heart Attack) & Type 2 Diabetes",
    "history": "History of hypertension since 2015. Previous stent placement in 2019. High cholesterol managed with statins. Non-smoker. Family history of coronary artery disease.",
    "meds": [
        {"name": "Lisinopril", "time": "08:00 AM", "purpose": "Blood Pressure"},
        {"name": "Metformin", "time": "09:00 AM", "purpose": "Blood Sugar"},
        {"name": "Clopidogrel", "time": "01:00 PM", "purpose": "Anti-platelet"},
        {"name": "Atorvastatin", "time": "08:00 PM", "purpose": "Cholesterol"},
        {"name": "Furosemide", "time": "08:00 AM", "purpose": "Water Retention"},
        {"name": "Potassium Chloride", "time": "01:00 PM", "purpose": "Electrolytes"},
        {"name": "Multivitamin", "time": "09:00 PM", "purpose": "General Health"}
    ],
    "tests": [
        {"test": "ECG (Electrocardiogram)", "remind": "Monday, 10:00 AM"},
        {"test": "HbA1c Blood Test", "remind": "Wednesday, 08:00 AM"},
        {"test": "Echocardiogram", "remind": "Friday, 02:00 PM"}
    ],
    "weekly_schedule": [
        {"day": "Monday", "task": "Fast for 8 hours before Blood Work; 15 min light walking."},
        {"day": "Tuesday", "task": "Check blood pressure at home; High-fiber diet focus."},
        {"day": "Wednesday", "task": "Glucose monitoring session; Upper body light stretching."},
        {"day": "Thursday", "task": "Hydration focus: 3L water; Review medication supply."},
        {"day": "Friday", "task": "Cardiology follow-up call; Lower body light exercise."},
        {"day": "Saturday", "task": "Rest day; Low-sodium meal prep for coming week."},
        {"day": "Sunday", "task": "Prepare pill organizer for next week; Evening meditation."}
    ],
    "organ_health": [
        {"organ": "Heart", "food": "Walnuts, Salmon, Leafy Greens"},
        {"organ": "Lungs", "food": "Apples, Turmeric, Ginger"},
        {"organ": "Kidneys", "food": "Blueberries, Red Grapes, Cauliflower"},
        {"organ": "Liver", "food": "Oatmeal, Green Tea, Olive Oil"},
        {"organ": "Brain", "food": "Avocados, Broccoli, Dark Chocolate"}
    ]
}

def train():
    np.random.seed(42)
    X = np.random.randint(20, 90, (1000, 5))
    y = (X[:, 0] * 0.2 + X[:, 1] * 0.5 > 35).astype(int)
    model = RandomForestClassifier().fit(X, y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

@app.route('/')
def index():
    return render_template('index.html', d=DATA)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    features = np.array([[int(data['age']), int(data['stay']), int(data['prev']), int(data['hr']), int(data['bp'])]])
    prob = model.predict_proba(features)[0][1]
    return jsonify({'prob': round(prob * 100, 2), 'risk': "High" if prob > 0.5 else "Low"})

if __name__ == '__main__':
    train()
    app.run(debug=True)