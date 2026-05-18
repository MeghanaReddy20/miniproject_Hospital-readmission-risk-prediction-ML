import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Create a dummy dataset
np.random.seed(42)
data_size = 1500
data = {
    'age': np.random.randint(18, 95, data_size),
    'gender': np.random.randint(0, 2, data_size),
    'stay_duration': np.random.randint(1, 20, data_size),
    'prev_admissions': np.random.randint(0, 6, data_size),
    'comorbidity_index': np.random.randint(0, 10, data_size),
}
df = pd.DataFrame(data)
# Simple logic for risk prediction
df['target'] = ((df['age'] * 0.2 + df['stay_duration'] * 0.8 + df['prev_admissions'] * 3) > 25).astype(int)

# 2. Train the model
X = df.drop('target', axis=1)
y = df['target']
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# 3. Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ SUCCESS: Model trained and saved as model.pkl")