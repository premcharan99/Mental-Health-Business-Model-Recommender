from flask import Flask, render_template, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import pickle
import numpy as np
from functools import lru_cache
import plotly.express as px
import plotly
import json
import os

app = Flask(__name__)

# Load model, scaler, and feature columns
with open('models/hybrid_business_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load dataset
df = pd.read_json('data/mental_health_business_2000.json')

# Rate limiting
limiter = Limiter(app, key_func=get_remote_address, default_limits=["50 per hour"])

# Prediction function with caching
@lru_cache(maxsize=128)
def predict_business_model(input_tuple):
    input_data = pd.DataFrame([dict(zip([
        'Service Type', 'Ownership', 'Budget', 'Target Age Group', 'Location',
        'Market Demand', 'Delivery Mode', 'Payment Methods', 'Accessibility Goal'
    ], input_tuple))])
    input_encoded = pd.get_dummies(input_data)
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]
    input_encoded['Budget'] = scaler.transform(input_encoded[['Budget']])
    probs = model.predict_proba(input_encoded)[0]
    top_indices = np.argsort(probs)[-3:][::-1]
    return [(le.classes_[i], probs[i] * 100) for i in top_indices]

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def dashboard():
    if request.method == 'POST':
        input_data = (
            request.form['service_type'],
            request.form['ownership'],
            float(request.form['budget']),
            request.form['target_age_group'],
            request.form['location'],
            request.form['market_demand'],
            request.form['delivery_mode'],
            request.form['payment_methods'],
            request.form['accessibility_goal']
        )
        predictions = predict_business_model(input_data)
        # Create bar chart for predictions
        fig = px.bar(
            x=[p[0] for p in predictions],
            y=[p[1] for p in predictions],
            labels={'x': 'Business Model', 'y': 'Confidence (%)'},
            title='Prediction Confidence'
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('dashboard.html', predictions=predictions, graph_json=graph_json)
    return render_template('dashboard.html')

@app.route('/dataset')
def dataset():
    dataset_sample = df.head(100).to_dict(orient='records')
    return render_template('dataset.html', dataset=dataset_sample)

@app.route('/download_dataset')
def download_dataset():
    return send_file('data/mental_health_business_2000.json', as_attachment=True)

@app.route('/visualizations')
def visualizations():
    # Service type distribution
    fig1 = px.histogram(df, x='Service Type', title='Service Type Distribution')
    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graph1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    # Business model distribution
    fig2 = px.histogram(df, x='Business Model', title='Business Model Distribution')
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graph2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('visualizations.html', graph1_json=graph1_json, graph2_json=graph2_json)

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(debug=True)