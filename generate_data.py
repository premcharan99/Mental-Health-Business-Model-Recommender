import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

# Define attributes and their possible values
attributes = {
    'Service Type': ['Teletherapy', 'Outpatient', 'Inpatient', 'Crisis Intervention', 'Support Group'],
    'Ownership': ['Private For-Profit', 'Nonprofit', 'Government'],
    'Budget': list(range(5000, 500001, 5000)),
    'Target Age Group': ['Children', 'Young Adults', 'Adults', 'Seniors'],
    'Location': ['Urban Global', 'Urban Local', 'Rural Local'],
    'Market Demand': ['Low', 'Medium', 'High'],
    'Delivery Mode': ['Online', 'In-Person', 'Hybrid'],
    'Payment Methods': ['Cash', 'Insurance', 'Grants', 'Subscription', 'Free'],
    'Accessibility Goal': ['Low', 'Medium', 'High'],
    'Business Model': ['Subscription', 'Nonprofit', 'Freemium', 'Pay-Per-Use', 'Government-Funded', 'Hybrid']
}

# Generate synthetic dataset with conditional probabilities
n_records = 2000
data = {attr: [] for attr in attributes}
for _ in range(n_records):
    service_type = np.random.choice(attributes['Service Type'], p=[0.4, 0.3, 0.15, 0.1, 0.05])
    data['Service Type'].append(service_type)

    ownership = np.random.choice(attributes['Ownership'], p=[0.5, 0.3, 0.2])
    data['Ownership'].append(ownership)

    budget = np.random.choice(attributes['Budget'])
    data['Budget'].append(budget)

    target_age = np.random.choice(attributes['Target Age Group'], p=[0.2, 0.4, 0.3, 0.1])
    data['Target Age Group'].append(target_age)

    location = np.random.choice(attributes['Location'], p=[0.5, 0.3, 0.2])
    data['Location'].append(location)

    market_demand = np.random.choice(attributes['Market Demand'], p=[0.2, 0.5, 0.3])
    data['Market Demand'].append(market_demand)

    delivery_mode = np.random.choice(attributes['Delivery Mode'], p=[0.5, 0.3, 0.2])
    data['Delivery Mode'].append(delivery_mode)

    payment_methods = np.random.choice(attributes['Payment Methods'], p=[0.2, 0.6, 0.1, 0.05, 0.05])
    data['Payment Methods'].append(payment_methods)

    accessibility_goal = np.random.choice(attributes['Accessibility Goal'], p=[0.3, 0.4, 0.3])
    data['Accessibility Goal'].append(accessibility_goal)

    # Conditional business model based on service type and market demand
    if service_type == 'Teletherapy' and market_demand == 'High' and delivery_mode == 'Online':
        business_model = np.random.choice(attributes['Business Model'], p=[0.6, 0.1, 0.15, 0.1, 0.05, 0.0])
    elif ownership == 'Nonprofit':
        business_model = np.random.choice(attributes['Business Model'], p=[0.1, 0.6, 0.1, 0.1, 0.05, 0.05])
    else:
        business_model = np.random.choice(attributes['Business Model'], p=[0.35, 0.2, 0.15, 0.15, 0.1, 0.05])
    data['Business Model'].append(business_model)

# Create DataFrame
df = pd.DataFrame(data)

# Save to JSON
os.makedirs('data', exist_ok=True)
df.to_json('data/mental_health_business_2000.json', orient='records', lines=True)

print(f"Dataset generated and saved to data/mental_health_business_2000.json with {n_records} records.")
print("Business Model distribution:")
print(df['Business Model'].value_counts())