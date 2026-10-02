from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import pickle

# Load model, encoders, and scaler
with open('best_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('encoder.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler_data = pickle.load(scaler_file)

app = Flask(__name__)
DATA_PATH = Path(__file__).resolve().parent / 'data' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
REPORT_SEGMENTS = {
    'Contract': 'Contract',
    'Internet service': 'InternetService',
    'Payment method': 'PaymentMethod',
    'Partner status': 'Partner',
    'Dependents': 'Dependents',
    'Senior citizen': 'senior_group',
    'Gender': 'gender',
    'Tenure': 'tenure_group',
}

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0, 1]
    return "Churn" if prediction == 1 else "No Churn", probability

@app.route('/report')
def report():
    df = pd.read_csv(DATA_PATH)
    df['ChurnFlag'] = (df['Churn'] == 'Yes').astype(int)
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 72],
        labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months'],
    )
    df['senior_group'] = df['SeniorCitizen'].map({0: 'Non-senior', 1: 'Senior citizen'})

    segment_data = {}
    for label, column in REPORT_SEGMENTS.items():
        grouped = df.groupby(column, observed=True).agg(
            customers=('ChurnFlag', 'size'),
            churners=('ChurnFlag', 'sum'),
            churn_rate=('ChurnFlag', 'mean'),
        ).reset_index().sort_values('churn_rate', ascending=False)
        segment_data[label] = [
            {
                'label': str(row[column]),
                'customers': int(row['customers']),
                'churners': int(row['churners']),
                'churn_rate': round(float(row['churn_rate']) * 100, 1),
                'churn_share': round(float(row['churners']) / int(df['ChurnFlag'].sum()) * 100, 1),
            }
            for _, row in grouped.iterrows()
        ]

    tenure = df.groupby('Churn')['tenure'].mean()
    summary = {
        'customers': len(df),
        'churners': int(df['ChurnFlag'].sum()),
        'churn_rate': round(float(df['ChurnFlag'].mean()) * 100, 1),
        'churn_tenure': round(float(tenure['Yes']), 1),
        'retained_tenure': round(float(tenure['No']), 1),
        'churn_monthly_charges': round(float(df.loc[df['ChurnFlag'] == 1, 'MonthlyCharges'].sum()), 2),
    }
    contract_risk = max(segment_data['Contract'], key=lambda item: item['churn_rate'])
    internet_risk = max(segment_data['Internet service'], key=lambda item: item['churn_rate'])
    payment_risk = max(segment_data['Payment method'], key=lambda item: item['churn_rate'])
    tenure_risk = max(segment_data['Tenure'], key=lambda item: item['churn_rate'])
    insights = [
        f"{contract_risk['label']} customers have the highest contract-segment churn rate ({contract_risk['churn_rate']}%).",
        f"{internet_risk['label']} customers have the highest internet-service churn rate ({internet_risk['churn_rate']}%).",
        f"{payment_risk['label']} customers have the highest payment-method churn rate ({payment_risk['churn_rate']}%).",
        f"Customers who churned averaged {summary['churn_tenure']} months of tenure, versus {summary['retained_tenure']} months for retained customers.",
    ]
    actions = [
        {
            'rank': '01',
            'title': 'Strengthen first-year retention',
            'metric': f"{tenure_risk['churn_rate']}% churn · {tenure_risk['churners']:,} churned",
            'detail': f"The {tenure_risk['label']} group accounts for {tenure_risk['churn_share']}% of all churned customers. Test onboarding and early-tenure support, then measure against a holdout.",
        },
        {
            'rank': '02',
            'title': 'Review month-to-month offers',
            'metric': f"{contract_risk['churn_rate']}% churn · {contract_risk['churners']:,} churned",
            'detail': 'Compare retention offers or contract options for this group. Its size makes it a high-volume segment, but results should be tested rather than assumed.',
        },
        {
            'rank': '03',
            'title': 'Investigate electronic-check experience',
            'metric': f"{payment_risk['churn_rate']}% churn · {payment_risk['churners']:,} churned",
            'detail': 'Check billing and payment friction for electronic-check customers before testing a payment-method intervention.',
        },
    ]
    return render_template(
        'report.html',
        summary=summary,
        segment_data=segment_data,
        insights=insights,
        actions=actions,
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
        }

        prediction, probability = make_prediction(input_data)

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
