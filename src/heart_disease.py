import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix
)
from xgboost import XGBClassifier

# Load Dataset

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

# Handle Missing Values

df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

# Convert 5 Classes into 4 Classes

df["num"] = df["num"].replace({
4: 3
})

# Disease Distribution

plt.figure(figsize=(6, 4))
sns.countplot(x="num", data=df)
plt.title("Heart Disease Severity Distribution")
plt.savefig("disease_distribution.png")
plt.show()

# Correlation Heatmap

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Features and Target

X = df.drop("num", axis=1)
y = df["num"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

# XGBoost Model

model = XGBClassifier(
objective="multi:softmax",
num_class=4,
n_estimators=200,
max_depth=4,
learning_rate=0.05,
random_state=42,
eval_metric="mlogloss"
)

# Training

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Evaluation

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance

feature_importance = pd.DataFrame({
"Feature": X.columns,
"Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
by="Importance",
ascending=False
)

plt.figure(figsize=(10, 5))
sns.barplot(
data=feature_importance,
x="Importance",
y="Feature"
)

plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.show()

# Confusion Matrix Heatmap

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
cm,
annot=True,
fmt="d",
cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

