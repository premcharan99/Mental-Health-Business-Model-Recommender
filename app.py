from flask import Flask, request, render_template, send_file
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import json
from io import StringIO

app = Flask(__name__)

# Load model and feature columns
with open('business_model_rf.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Load dataset
df = pd.read_json('mental_health_business_2000.json')

# Dropdown options
options = {
    'service_type': ['Outpatient', 'Teletherapy', 'Inpatient', 'Wellness App', 'Community Program'],
    'ownership': ['Private For-Profit', 'Private Nonprofit', 'Public'],
    'target_age_group': ['Children', 'Teens', 'Young Adults', 'Adults', 'Seniors', 'All Ages'],
    'location': ['Urban US', 'Rural US', 'Urban Global', 'Rural Global'],
    'market_demand': ['Low', 'Medium', 'High'],
    'delivery_mode': ['In-Person', 'Online', 'Hybrid'],
    'payment_cash': ['No', 'Yes'],
    'payment_insurance': ['No', 'Yes'],
    'payment_grants': ['No', 'Yes'],
    'payment_subscription': ['No', 'Yes'],
    'payment_free': ['No', 'Yes'],
    'accessibility_goal': ['Low', 'Medium', 'High']
}


# Hybrid prediction function
def predict_hybrid(input_data, model, feature_columns):
    input_df = pd.DataFrame([input_data])
    input_X = pd.get_dummies(input_df)
    for col in feature_columns:
        if col not in input_X.columns:
            input_X[col] = 0
    input_X = input_X[feature_columns]
    probs = model.predict_proba(input_X)[0]
    classes = model.classes_
    top_indices = probs.argsort()[-3:][::-1]
    recommendations = [(classes[i], round(probs[i] * 100, 2)) for i in top_indices if probs[i] > 0.1]
    return recommendations


# Generate visualization for predictions
def generate_prediction_charts(recommendations):
    models = [r[0] for r in recommendations]
    confidences = [r[1] for r in recommendations]

    # Bar chart
    bar_fig = go.Figure(data=[
        go.Bar(x=models, y=confidences, marker_color='#4A90E2')
    ])
    bar_fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Business Model",
        yaxis_title="Confidence (%)",
        template="plotly_white"
    )
    bar_json = bar_fig.to_json()

    # Pie chart
    pie_fig = go.Figure(data=[
        go.Pie(labels=models, values=confidences, hole=0.3, marker_colors=['#4A90E2', '#50C878', '#F7DC6F'])
    ])
    pie_fig.update_layout(
        title="Model Probability Distribution",
        template="plotly_white"
    )
    pie_json = pie_fig.to_json()

    return bar_json, pie_json


# Generate dataset visualizations
def generate_dataset_visualizations():
    # Service type distribution
    service_counts = df['service_type'].value_counts()
    service_fig = px.bar(
        x=service_counts.index,
        y=service_counts.values,
        labels={'x': 'Service Type', 'y': 'Count'},
        title="Service Type Distribution",
        color=service_counts.index,
        color_discrete_sequence=['#4A90E2', '#50C878', '#F7DC6F', '#E57373', '#AB47BC']
    )
    service_fig.update_layout(template="plotly_white")
    service_json = service_fig.to_json()

    # Business model pie chart
    model_counts = df['business_model'].value_counts()
    model_fig = go.Figure(data=[
        go.Pie(labels=model_counts.index, values=model_counts.values, hole=0.3,
               marker_colors=['#4A90E2', '#50C878', '#F7DC6F', '#E57373', '#AB47BC', '#FF9800'])
    ])
    model_fig.update_layout(
        title="Business Model Distribution",
        template="plotly_white"
    )
    model_json = model_fig.to_json()

    return service_json, model_json


@app.route('/', methods=['GET', 'POST'])
def dashboard():
    recommendations = None
    bar_json = None
    pie_json = None
    if request.method == 'POST':
        input_data = {
            'service_type': request.form['service_type'],
            'ownership': request.form['ownership'],
            'budget_k': int(request.form['budget_k']),
            'target_age_group': request.form['target_age_group'],
            'location': request.form['location'],
            'market_demand': request.form['market_demand'],
            'delivery_mode': request.form['delivery_mode'],
            'payment_cash': 1 if request.form['payment_cash'] == 'Yes' else 0,
            'payment_insurance': 1 if request.form['payment_insurance'] == 'Yes' else 0,
            'payment_grants': 1 if request.form['payment_grants'] == 'Yes' else 0,
            'payment_subscription': 1 if request.form['payment_subscription'] == 'Yes' else 0,
            'payment_free': 1 if request.form['payment_free'] == 'Yes' else 0,
            'accessibility_goal': request.form['accessibility_goal']
        }
        recommendations = predict_hybrid(input_data, model, feature_columns)
        bar_json, pie_json = generate_prediction_charts(recommendations)
    return render_template('index.html', options=options, recommendations=recommendations,
                           bar_json=bar_json, pie_json=pie_json)


@app.route('/dataset')
def dataset():
    dataset_html = df.to_html(classes='table table-striped', index=False, max_rows=100)
    return render_template('dataset.html', dataset_html=dataset_html)


@app.route('/download_dataset')
def download_dataset():
    return send_file('mental_health_business_2000.json', as_attachment=True)


@app.route('/visualizations')
def visualizations():
    service_json, model_json = generate_dataset_visualizations()
    return render_template('visualizations.html', service_json=service_json, model_json=model_json)


@app.route('/instructions')
def instructions():
    return render_template('instructions.html')


if __name__ == '__main__':
    app.run(debug=True)