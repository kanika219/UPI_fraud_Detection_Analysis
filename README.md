# AI-Powered UPI Fraud Detection System

This project is a Streamlit-based web application that uses Machine Learning (Random Forest) to detect fraudulent UPI transactions.

## 🚀 Features
- **Real-Time Fraud Detection**: Instant transaction risk assessment.
- **AI Intelligence**: Powered by a robust ML model trained on transaction history.
- **Smart Analytics**: Modern UI with risk visualization and feature importance.
- **Model Retraining**: Built-in option to retrain the model.

## 📁 Project Structure
- `app.py`: Main Streamlit application.
- `model_training.py`: Script to train and save the ML model.
- `AIML Dataset.csv`: The core transaction dataset.
- `requirements.txt`: List of dependencies.
- `model.pkl`: Saved trained model.
- `scaler.pkl`, `label_encoder.pkl`, `features.pkl`: Preprocessing objects.

## 🛠️ Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed. Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The model needs to be trained on the `AIML Dataset.csv` before running the application.
```bash
python model_training.py
```

### 3. Run the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## 🧠 Model Details
- **Algorithm**: Random Forest Classifier
- **Target Column**: `isFraud` (0: Legitimate, 1: Fraud)
- **Features**: step, type, amount, balances before/after transaction.
- **Handling Class Imbalance**: `class_weight='balanced'`
