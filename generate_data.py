import random
import json

# Define options
service_types = ["Outpatient", "Teletherapy", "Inpatient", "Wellness App", "Community Program"]
ownerships = ["Private For-Profit", "Private Nonprofit", "Public"]
age_groups = ["Children", "Teens", "Young Adults", "Adults", "Seniors", "All Ages"]
locations = ["Urban US", "Rural US", "Urban Global", "Rural Global"]
demands = ["Low", "Medium", "High"]
deliveries = ["In-Person", "Online", "Hybrid"]
access_goals = ["Low", "Medium", "High"]
business_models = ["Subscription", "Freemium", "Pay-Per-Use", "Insurance-Based", "Nonprofit", "B2B Partnership"]

# Generate 2000 records
data = []
for i in range(2000):
    service = random.choices(service_types, weights=[0.4, 0.2, 0.15, 0.15, 0.1])[0]
    ownership = random.choices(ownerships, weights=[0.4, 0.3, 0.3])[0]
    budget = random.randint(5, 500)
    age = random.choices(age_groups, weights=[0.05, 0.15, 0.25, 0.3, 0.05, 0.2])[0]
    loc = random.choices(locations, weights=[0.5, 0.2, 0.2, 0.1])[0]
    demand = random.choices(demands, weights=[0.2, 0.3, 0.5])[0]
    delivery = random.choices(deliveries, weights=[0.5, 0.3, 0.2])[0]
    access = random.choices(access_goals, weights=[0.2, 0.5, 0.3])[0]

    # Payment methods with correlations
    cash = 1 if random.random() < (
        0.6 if service in ["Outpatient", "Teletherapy"] and ownership == "Private For-Profit" else 0.3) else 0
    insurance = 1 if random.random() < (0.8 if service in ["Outpatient", "Inpatient"] else 0.2) else 0
    grants = 1 if random.random() < (0.7 if ownership in ["Private Nonprofit", "Public"] else 0.1) else 0
    subscription = 1 if random.random() < (0.7 if service in ["Teletherapy", "Wellness App"] else 0.1) else 0
    free = 1 if random.random() < (
        0.5 if ownership in ["Private Nonprofit", "Public"] and access == "High" else 0.1) else 0

    # Assign business model with balancing
    if i % 6 == 0:
        model = business_models[i % 6]
    elif service == "Wellness App" and subscription and free:
        model = "Freemium"
    elif service in ["Teletherapy", "Wellness App"] and subscription and ownership == "Private For-Profit":
        model = "Subscription"
    elif cash and not subscription and service in ["Outpatient", "Teletherapy"]:
        model = "Pay-Per-Use"
    elif insurance and service in ["Outpatient", "Inpatient"]:
        model = "Insurance-Based"
    elif grants or free and ownership in ["Private Nonprofit", "Public"]:
        model = "Nonprofit"
    else:
        model = "B2B Partnership"

    record = {
        "service_type": service,
        "ownership": ownership,
        "budget_k": budget,
        "target_age_group": age,
        "location": loc,
        "market_demand": demand,
        "delivery_mode": delivery,
        "payment_cash": cash,
        "payment_insurance": insurance,
        "payment_grants": grants,
        "payment_subscription": subscription,
        "payment_free": free,
        "accessibility_goal": access,
        "business_model": model
    }
    data.append(record)

# Save to JSON
with open('mental_health_business_2000.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Generated 2000 records in 'mental_health_business_2000.json'")