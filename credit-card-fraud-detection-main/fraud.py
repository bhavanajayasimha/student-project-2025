import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
print("Loading dataset...")
data = pd.read_csv("creditcard.csv")
print("Dataset loaded successfully!")
print("Shape:", data.shape)

# 2. Check class distribution
print("\nClass distribution:")
print(data['Class'].value_counts())

# 3. Split X and y
X = data.drop('Class', axis=1)
y = data['Class']

# 4. Scale Time and Amount
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# 6. Train model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
print("Model training complete!")

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
