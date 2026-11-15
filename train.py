# Train a Random Forest on heart.csv and save it with joblib.
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/heart.csv")           # put the Kaggle CSV here
X, y = df.drop(columns="target"), df["target"]  # target: 1 = disease, 0 = no disease
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_tr, y_tr)
print("Accuracy:", accuracy_score(y_te, model.predict(X_te)))

# Save model + feature order together so the API uses the same column order
joblib.dump({"model": model, "features": list(X.columns)}, "model/heart_model.joblib")
