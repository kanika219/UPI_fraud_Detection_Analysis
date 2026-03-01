import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engineering import BalanceFeatureEngineer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os


# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("AIML Dataset.csv")
# no sampling: use full dataset for mathematically correct model
print(f"Using full dataset with {len(df)} rows")


# 2. Display basic info
print(f"Dataset Shape: {df.shape}")
print("First 5 rows:")
print(df.head())
print("Column Names:", df.columns.tolist())

# Detect target column
target_col = 'isFraud'
if target_col not in df.columns:
    print(f"Error: {target_col} not found in dataset.")
    exit()

print("Class Distribution:")
print(df[target_col].value_counts())

# Print fraud ratio by transaction type
if 'type' in df.columns:
    print("Fraud ratio per transaction type:")
    fraud_by_type = (df.groupby('type')['isFraud'].mean() * 100).round(2)
    print(fraud_by_type)

# 3. Preprocessing
# Drop columns that are usually IDs or redundant
cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
X = df.drop(columns=[target_col] + [c for c in cols_to_drop if c in df.columns])
y = df[target_col]

# Handle missing values (only numeric columns)
numeric_cols = X.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

# save raw feature list for UI generation
joblib.dump(X.columns.tolist(), 'features.pkl')


# 4. Build preprocessing pipeline with feature engineering
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['type'] if 'type' in X.columns else [])
    ],
    remainder='passthrough'  # Keep numerical features unchanged
)

# Create full pipeline: feature engineering → preprocessing → model
pipeline = Pipeline(steps=[
    ('feature_engineering', BalanceFeatureEngineer()),
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=20, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1))
])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. Model Training
print("\nTraining pipeline...")
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
pipeline.fit(X_train, y_train)

# 7. Evaluation
print("Evaluating pipeline...")
y_pred = pipeline.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1-score": f1_score(y_test, y_pred)
}

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save complete pipeline
joblib.dump(pipeline, "model.pkl")
print("\nComplete pipeline saved as model.pkl")