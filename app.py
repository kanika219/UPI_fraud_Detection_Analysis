import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
import os

from feature_engineering import BalanceFeatureEngineer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer for loading pickled pipelines
class BalanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create engineered features for balance inconsistency detection."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if {'oldbalanceOrg', 'amount', 'newbalanceOrig'}.issubset(X.columns):
            X['balanceOrigDiff'] = X['oldbalanceOrg'] - X['amount'] - X['newbalanceOrig']
        else:
            X['balanceOrigDiff'] = 0.0
        
        if {'oldbalanceDest', 'amount', 'newbalanceDest'}.issubset(X.columns):
            X['balanceDestDiff'] = X['oldbalanceDest'] + X['amount'] - X['newbalanceDest']
        else:
            X['balanceDestDiff'] = 0.0
        
        return X

# Page Config
st.set_page_config(page_title="UPI Fraud Guard", layout="wide", page_icon="🛡️")

# Custom CSS for Dark Theme and Modern UI
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .legit-card {
        background-color: #1e392a;
        border: 2px solid #28a745;
        color: #ffffff;
    }
    .fraud-card {
        background-color: #3d1b1b;
        border: 2px solid #dc3545;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def start_training_background():
    """Spawn a background thread to run model_training.py once."""
    if st.session_state.get('training_started'):
        return
    st.session_state['training_started'] = True
    # use threading to avoid blocking Streamlit
    import threading, subprocess, sys
    def _train():
        try:
            # run with same interpreter and working directory
            subprocess.run([sys.executable, "model_training.py"], cwd=os.path.dirname(__file__))
        except Exception as e:
            # we can't use st.error inside thread, but log to file or ignore
            pass
    threading.Thread(target=_train, daemon=True).start()


def load_artifacts():
    """Load preprocessing pipeline and ANN model.

    Returns tuple (preprocessor, ann_model) or (None,None) if missing.
    Caches result in `st.session_state['artifacts']` so we don't reload repeatedly.
    """
    # check session cache first
    if 'artifacts' in st.session_state:
        return st.session_state['artifacts']

    # ensure files exist before attempting load
    if not os.path.isfile("preprocessor.pkl") or not os.path.isfile("model.h5"):
        # trigger training in background
        st.warning("Model artifacts are not available. Training will start automatically in the background. This may take several minutes.")
        start_training_background()
        st.session_state['artifacts'] = (None, None)
        return None, None

    try:
        preproc = joblib.load("preprocessor.pkl")
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        st.session_state['artifacts'] = (None, None)
        return None, None
    try:
        ann = load_model("model.h5")
    except Exception as e:
        st.error(f"Error loading ANN model: {e}")
        st.session_state['artifacts'] = (None, None)
        return None, None
    st.session_state['artifacts'] = (preproc, ann)
    return preproc, ann

def home_page():
    st.title("🛡️ UPI Fraud Guard")
    st.subheader("Secure Your Digital Transactions")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Welcome to **UPI Fraud Guard**, an advanced AI-powered system designed to detect and prevent 
        fraudulent UPI transactions in real-time. Using state-of-the-art machine learning algorithms, 
        we analyze transaction patterns to ensure your digital payments remain secure.
        """)
        if st.button("Get Started →"):
            st.session_state.page = "Fraud Detection"
            st.rerun()
    with col2:
        st.image("https://img.icons8.com/clouds/500/000000/safe.png", width=300)

    st.divider()
    st.header("Why Choose UPI Fraud Guard?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("⚡ Real-Time Protection")
        st.write("Instant analysis of transactions as they happen.")
    with c2:
        st.subheader("🧠 AI Intelligence")
        st.write("Powered by an Artificial Neural Network (ANN) model.")
    with c3:
        st.subheader("📊 Smart Analytics")
        st.write("Deep insights into transaction risks and patterns.")

def detection_page():
    st.title("🔍 Fraud Detection")
    
    preproc, ann_model = load_artifacts()
    if preproc is None or ann_model is None:
        # If training has been kicked off but files still missing, inform user
        if st.session_state.get('training_started') and not os.path.isfile("model.h5"):
            st.info("Model training is currently running in the background. Please wait a few minutes and refresh the page.")
        # load_artifacts already displayed a warning or error
        return

    # load the pre-saved raw features list for inputs
    try:
        features = joblib.load('features.pkl')
    except Exception as e:
        st.error(f"Could not load feature list: {e}")
        return

    engineered_features = ['balanceOrigDiff', 'balanceDestDiff']
    input_features = [f for f in features if f not in engineered_features]

    st.sidebar.header("Input Transaction Details")
    inputs = {}
    
    # prepare list of transaction types for selectbox
    type_categories = []
    try:
        ohe = preproc.named_steps['preprocessing'].named_transformers_['categorical']
        type_categories = ohe.categories_[0].tolist()
    except Exception:
        pass

    # Generate input fields for non-engineered features
    for feat in input_features:
        if feat == 'type':
            if type_categories:
                inputs[feat] = st.sidebar.selectbox("Select Transaction Type", type_categories)
            else:
                inputs[feat] = st.sidebar.text_input("Enter Transaction Type", value="TRANSFER")
        elif feat == 'step':
            inputs[feat] = st.sidebar.number_input(f"Enter {feat} (Hours since start)", min_value=1, value=1)
        else:
            inputs[feat] = st.sidebar.number_input(f"Enter {feat}", value=0.0)

    # Adjustable threshold (default 0.2)
    threshold = st.sidebar.slider("Fraud Probability Threshold", 0.0, 1.0, 0.2, step=0.01)

    if st.button("Analyze Transaction"):
        try:
            # Create DataFrame from inputs
            input_df = pd.DataFrame([inputs])
            
            # Pass raw DataFrame through preprocessing pipeline then ANN
            # Pipeline handles feature engineering and scaling/encoding.
            processed = preproc.transform(input_df)
            prob = ann_model.predict(processed)[0][0]
            fraud_probability = float(prob)
            safe_probability = 1.0 - fraud_probability
            
            # additional logic: if engineered differences are non-zero, boost fraud score
            diff_orig = input_df['oldbalanceOrg'].iloc[0] - input_df['amount'].iloc[0] - input_df['newbalanceOrig'].iloc[0]
            diff_dest = input_df['oldbalanceDest'].iloc[0] + input_df['amount'].iloc[0] - input_df['newbalanceDest'].iloc[0]
            if diff_orig != 0 or diff_dest != 0:
                # bump to near-certain fraud
                fraud_probability = max(fraud_probability, 0.99)
                safe_probability = 1 - fraud_probability
            
            # Apply threshold
            prediction = 1 if fraud_probability >= threshold else 0
            
            fraud_pct = round(fraud_probability * 100, 2)
            safe_pct = round(safe_probability * 100, 2)

            st.divider()
            
            if prediction == 1:
                st.markdown(f'<div class="card fraud-card"><h2>🚨 FRAUDULENT TRANSACTION DETECTED</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card legit-card"><h2>✅ LEGITIMATE TRANSACTION</h2></div>', unsafe_allow_html=True)

            st.subheader(f"Fraud Risk Score: {fraud_pct}%")
            
            # Donut Chart
            fig = go.Figure(data=[go.Pie(
                labels=['Safe', 'Fraud'], 
                values=[safe_pct, fraud_pct], 
                hole=.6, 
                marker_colors=['#28a745', '#dc3545']
            )])
            fig.update_layout(title_text="Risk Analysis", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")



# Navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Fraud Detection"], index=["Home", "Fraud Detection"].index(st.session_state.page))
st.session_state.page = page

if page == "Home":
    home_page()
elif page == "Fraud Detection":
    detection_page()

