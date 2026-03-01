import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engineering import BalanceFeatureEngineer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

# TensorFlow / Keras for ANN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("AIML Dataset.csv")
# Sample 10% for faster training (still mathematically sound with 11k+ samples)
df = df.sample(frac=0.1, random_state=42)
print(f"Using sampled dataset with {len(df)} rows for faster training")


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
# We will apply feature engineering first and then handle categorical encoding
# and numeric scaling (required by ANN). The remainder of the columns
# after the categorical transform will be scaled using StandardScaler.
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['type'] if 'type' in X.columns else [])
    ],
    remainder=StandardScaler()  # scale all other (numeric) features
)

# Combine feature engineering and preprocessing so the same pipeline
# can be saved and reused at prediction time.
pipeline_preprocess = Pipeline(steps=[
    ('feature_engineering', BalanceFeatureEngineer()),
    ('preprocessing', preprocessor)
])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. Model Training with ANN
print("\nTraining preprocessing pipeline...")
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
# fit preprocessing pipeline and transform data
pipeline_preprocess.fit(X_train)
X_train_proc = pipeline_preprocess.transform(X_train)
X_test_proc = pipeline_preprocess.transform(X_test)

# compute class weights to address imbalance
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight = {cls: w for cls, w in zip(classes, weights)}

# build ANN model (smaller for faster training)
input_dim = X_train_proc.shape[1]
print(f"Building optimized ANN with input dimension {input_dim}")
tf.random.set_seed(42)
model = Sequential([
    Dense(32, activation='relu', input_shape=(input_dim,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training ANN model...")
early_stop = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
history = model.fit(
    X_train_proc,
    y_train,
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluation
print("Evaluating ANN model...")
prob_test = model.predict(X_test_proc)
# threshold at 0.5 for metrics
y_pred = (prob_test.flatten() >= 0.5).astype(int)

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

# 8. Save preprocessing pipeline and ANN model
joblib.dump(pipeline_preprocess, "preprocessor.pkl")
print("\nPreprocessing pipeline saved as preprocessor.pkl")

model.save("model.h5")
print("ANN model saved as model.h5")