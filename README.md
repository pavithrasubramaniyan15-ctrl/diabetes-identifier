# Diabetes Type Identifier

A machine learning web application that classifies patients as having **No Diabetes**, **Type 1 Diabetes**, or **Type 2 Diabetes** based on health metrics.

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates model.pkl and scaler.pkl)
python train_model.py

# Start the web app
python app.py
```

Then open **http://localhost:5000** in your browser.

## Input Fields

| Field             | Unit     | Example |
|-------------------|----------|---------|
| Age               | years    | 45      |
| Glucose           | mg/dL    | 130     |
| BMI               | kg/m²    | 28.5    |
| Blood Pressure    | mmHg     | 80      |
| Insulin           | μU/mL    | 100     |
| HbA1c             | %        | 6.5     |
| Family History    | Yes/No   | Yes     |
| Physical Activity | Low–High | Moderate|

## Output

The app returns only:

```
Result: Type 2 Diabetes
```

## ML Details

- **Algorithm**: Random Forest (200 estimators) with balanced class weights
- **Imbalance handling**: `compute_class_weight('balanced')` from scikit-learn
- **Validation**: 5-fold Stratified Cross-Validation
- **Train/Test split**: 80/20 stratified
