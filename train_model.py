import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

np.random.seed(42)
N = 2000

# Simulate realistic health data with 3 classes
# Class 0 = No Diabetes, Class 1 = Type 1, Class 2 = Type 2
def generate_patient(label):
    if label == 0:  # No Diabetes
        age = np.random.normal(35, 12)
        glucose = np.random.normal(88, 10)
        bmi = np.random.normal(24, 3)
        bp = np.random.normal(70, 8)
        insulin = np.random.normal(80, 25)
        hba1c = np.random.normal(5.0, 0.3)
        family_history = np.random.choice([0, 1], p=[0.8, 0.2])
        physical_activity = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])
    elif label == 1:  # Type 1 (autoimmune, usually younger, low insulin)
        age = np.random.normal(22, 10)
        glucose = np.random.normal(195, 30)
        bmi = np.random.normal(22, 3)
        bp = np.random.normal(75, 10)
        insulin = np.random.normal(12, 8)   # very low - body doesn't produce it
        hba1c = np.random.normal(8.5, 1.2)
        family_history = np.random.choice([0, 1], p=[0.7, 0.3])
        physical_activity = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
    else:  # Type 2 (metabolic, older, insulin resistant)
        age = np.random.normal(52, 12)
        glucose = np.random.normal(175, 35)
        bmi = np.random.normal(31, 5)
        bp = np.random.normal(82, 12)
        insulin = np.random.normal(160, 50)  # high - insulin resistant
        hba1c = np.random.normal(7.8, 1.0)
        family_history = np.random.choice([0, 1], p=[0.4, 0.6])
        physical_activity = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])

    return {
        'age': max(1, age),
        'glucose': max(50, glucose),
        'bmi': max(10, bmi),
        'blood_pressure': max(40, bp),
        'insulin': max(0, insulin),
        'hba1c': max(3.5, hba1c),
        'family_history': family_history,
        'physical_activity': physical_activity,
        'label': label
    }

# Imbalanced: Type 1 is rarer
counts = {0: 900, 1: 300, 2: 800}
records = []
for label, n in counts.items():
    for _ in range(n):
        records.append(generate_patient(label))

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

features = ['age', 'glucose', 'bmi', 'blood_pressure', 'insulin', 'hba1c', 'family_history', 'physical_activity']
X = df[features]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights for imbalance handling
classes = np.array([0, 1, 2])
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {0: cw[0], 1: cw[1], 2: cw[2]}

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
print(f"CV F1 Macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Type 1', 'Type 2']))

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
