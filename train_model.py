import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
with open('mental_health_business_2000.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Preprocess
X = pd.get_dummies(df.drop('business_model', axis=1))
y = df['business_model']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and feature columns
with open('business_model_rf.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

print("Model and feature columns saved as 'business_model_rf.pkl' and 'feature_columns.pkl'")