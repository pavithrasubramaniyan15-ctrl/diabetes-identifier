import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

LABEL_MAP = {0: 'No Diabetes', 1: 'Type 1 Diabetes', 2: 'Type 2 Diabetes'}

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Diabetes Type Identifier</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }

    .card {
      background: #1a1d2e;
      border: 1px solid #2d3154;
      border-radius: 20px;
      padding: 40px 44px;
      width: 100%;
      max-width: 580px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.5);
    }

    .header {
      text-align: center;
      margin-bottom: 36px;
    }

    .icon {
      font-size: 48px;
      margin-bottom: 12px;
    }

    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      color: #f0f4ff;
      letter-spacing: -0.3px;
    }

    .subtitle {
      font-size: 0.875rem;
      color: #6b7db3;
      margin-top: 6px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .field.full { grid-column: span 2; }

    label {
      font-size: 0.78rem;
      font-weight: 600;
      color: #8899cc;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }

    input, select {
      background: #0f1117;
      border: 1px solid #2d3154;
      border-radius: 10px;
      color: #e2e8f0;
      padding: 11px 14px;
      font-size: 0.95rem;
      transition: border-color 0.2s, box-shadow 0.2s;
      outline: none;
      width: 100%;
    }

    input:focus, select:focus {
      border-color: #5b6ef5;
      box-shadow: 0 0 0 3px rgba(91,110,245,0.15);
    }

    select option { background: #1a1d2e; }

    .hint {
      font-size: 0.72rem;
      color: #4a5580;
    }

    button {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #5b6ef5, #7c3aed);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      margin-top: 8px;
      letter-spacing: 0.3px;
      transition: opacity 0.2s, transform 0.1s;
    }

    button:hover { opacity: 0.92; }
    button:active { transform: scale(0.99); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    #result-box {
      display: none;
      margin-top: 28px;
      border-radius: 14px;
      padding: 22px 24px;
      text-align: center;
      border: 1.5px solid;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    #result-box.no-diabetes   { background: #0d2318; border-color: #22c55e; }
    #result-box.type1-diabetes { background: #1f1430; border-color: #a855f7; }
    #result-box.type2-diabetes { background: #1c1508; border-color: #f59e0b; }

    .result-label {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      margin-bottom: 8px;
      opacity: 0.7;
    }

    .result-value {
      font-size: 1.55rem;
      font-weight: 800;
    }

    .no-diabetes   .result-value { color: #4ade80; }
    .type1-diabetes .result-value { color: #c084fc; }
    .type2-diabetes .result-value { color: #fbbf24; }

    .spinner { display: none; }
    .loading .spinner { display: inline-block; animation: spin 0.8s linear infinite; }
    .loading .btn-text { display: none; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<div class="card">
  <div class="header">
    <div class="icon">🩺</div>
    <h1>Diabetes Type Identifier</h1>
    <p class="subtitle">Enter patient health details to classify diabetes type</p>
  </div>

  <form id="form">
    <div class="grid">
      <div class="field">
        <label>Age <span class="hint">(years)</span></label>
        <input type="number" id="age" min="1" max="110" placeholder="e.g. 45" required />
      </div>
      <div class="field">
        <label>Glucose <span class="hint">(mg/dL)</span></label>
        <input type="number" id="glucose" min="50" max="400" placeholder="e.g. 130" required />
      </div>
      <div class="field">
        <label>BMI <span class="hint">(kg/m²)</span></label>
        <input type="number" id="bmi" min="10" max="70" step="0.1" placeholder="e.g. 28.5" required />
      </div>
      <div class="field">
        <label>Blood Pressure <span class="hint">(mmHg)</span></label>
        <input type="number" id="bp" min="40" max="200" placeholder="e.g. 80" required />
      </div>
      <div class="field">
        <label>Insulin <span class="hint">(μU/mL)</span></label>
        <input type="number" id="insulin" min="0" max="900" placeholder="e.g. 100" required />
      </div>
      <div class="field">
        <label>HbA1c <span class="hint">(%)</span></label>
        <input type="number" id="hba1c" min="3" max="15" step="0.1" placeholder="e.g. 6.5" required />
      </div>
      <div class="field">
        <label>Family History</label>
        <select id="family_history">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>
      <div class="field">
        <label>Physical Activity</label>
        <select id="physical_activity">
          <option value="0">Low</option>
          <option value="1">Moderate</option>
          <option value="2">High</option>
        </select>
      </div>
    </div>

    <button type="submit" id="btn">
      <span class="spinner">⟳</span>
      <span class="btn-text">Predict</span>
    </button>
  </form>

  <div id="result-box">
    <div class="result-label">Result</div>
    <div class="result-value" id="result-text"></div>
  </div>
</div>

<script>
  document.getElementById('form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('btn');
    btn.classList.add('loading');
    btn.disabled = true;

    const payload = {
      age: parseFloat(document.getElementById('age').value),
      glucose: parseFloat(document.getElementById('glucose').value),
      bmi: parseFloat(document.getElementById('bmi').value),
      blood_pressure: parseFloat(document.getElementById('bp').value),
      insulin: parseFloat(document.getElementById('insulin').value),
      hba1c: parseFloat(document.getElementById('hba1c').value),
      family_history: parseInt(document.getElementById('family_history').value),
      physical_activity: parseInt(document.getElementById('physical_activity').value),
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      const box = document.getElementById('result-box');
      const text = document.getElementById('result-text');

      box.className = '';
      box.classList.add(data.result.toLowerCase().replace(/ /g, '-').replace('diabetes', 'diabetes'));

      const classMap = {
        'No Diabetes': 'no-diabetes',
        'Type 1 Diabetes': 'type1-diabetes',
        'Type 2 Diabetes': 'type2-diabetes'
      };
      box.className = classMap[data.result] || 'no-diabetes';
      text.textContent = data.result;
      box.style.display = 'block';
    } catch (err) {
      alert('Prediction failed. Please try again.');
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  });
</script>
</body>
</html>'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['age'],
        data['glucose'],
        data['bmi'],
        data['blood_pressure'],
        data['insulin'],
        data['hba1c'],
        data['family_history'],
        data['physical_activity']
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    return jsonify({'result': LABEL_MAP[int(prediction)]})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
