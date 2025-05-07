import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_json('data/mental_health_business_2000.json', lines=True)

# Verify dataset balance
print("Business Model distribution:")
print(df['Business Model'].value_counts())

# Prepare features and target
X = df.drop('Business Model', axis=1)
y = df['Business Model']

# Encode categorical features
X_encoded = pd.get_dummies(X, columns=['Service Type', 'Ownership', 'Target Age Group', 'Location',
                                      'Market Demand', 'Delivery Mode', 'Payment Methods', 'Accessibility Goal'])

# Scale budget
scaler = StandardScaler()
X_encoded['Budget'] = scaler.fit_transform(X_encoded[['Budget']])

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded Business Model classes:", le.classes_)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Define classifiers with improved parameters
rf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=5, random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, learning_rate='adaptive',
                    early_stopping=True, alpha=0.0001, random_state=42)

# Evaluate individual classifiers
classifiers = {'Random Forest': rf, 'XGBoost': xgb, 'Gradient Boosting': gb, 'MLP': mlp}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    cv_scores = cross_val_score(clf, X_encoded, y_encoded, cv=3, scoring='accuracy')
    print(f"{name} - Train Accuracy: {train_score*100:.1f}%, Test Accuracy: {test_score*100:.1f}%, "
          f"CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('gb', gb),
        ('mlp', mlp)
    ],
    voting='soft',
    weights=[0.3, 0.3, 0.2, 0.2]
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Evaluate ensemble
train_accuracy = ensemble.score(X_train, y_train)
test_accuracy = ensemble.score(X_test, y_test)
cv_scores = cross_val_score(ensemble, X_encoded, y_encoded, cv=3, scoring='accuracy')
print(f"\nEnsemble - Train Accuracy: {train_accuracy*100:.1f}%, Test Accuracy: {test_accuracy*100:.1f}%, "
      f"CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# Save model, scaler, and feature columns
os.makedirs('models', exist_ok=True)
with open('models/hybrid_business_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X_encoded.columns.tolist(), f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model, scaler, feature columns, and label encoder saved to models/ directory.")