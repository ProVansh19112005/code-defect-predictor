from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
model_nlp = pickle.load(open("model_nlp.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

model_numeric = pickle.load(open("model_numeric.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


# -------------------------
# NLP Prediction
# -------------------------
@app.route('/predict_text', methods=['POST'])
def predict_text():
    message = request.form['message']

    vector = vectorizer.transform([message])
    prediction = model_nlp.predict(vector)[0]
    prob = model_nlp.predict_proba(vector)[0][1]

    result = "Bug Fix" if prediction == 1 else "Not a Bug Fix"

    return render_template(
        'result.html',
        result=result,
        probability=round(prob * 100, 2),
        input_data=message
    )


# -------------------------
# Numeric Prediction
# -------------------------
@app.route('/predict_numeric', methods=['POST'])
def predict_numeric():
    features = [
        float(request.form['nloc']),
        float(request.form['previous_changes']),
        float(request.form['developer_count']),
        float(request.form['code_churn']),
        float(request.form['net_change']),
        float(request.form['lines_added']),
        float(request.form['lines_deleted'])
    ]

    final_features = np.array([features])

    prediction = model_numeric.predict(final_features)[0]
    prob = model_numeric.predict_proba(final_features)[0][1]

    result = "Bug Fix" if prediction == 1 else "Not a Bug Fix"

    return render_template(
        'result.html',
        result=result,
        probability=round(prob * 100, 2),
        input_data="Numeric Features"
    )


if __name__ == "__main__":
    app.run(debug=True)