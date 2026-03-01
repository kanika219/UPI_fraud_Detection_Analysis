import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
@st.cache_resource
def load_pipeline():
    """Load the trained pipeline model."""
    try:
        pipeline = joblib.load("model.pkl")
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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
        st.write("Powered by advanced Random Forest models.")
    with c3:
        st.subheader("📊 Smart Analytics")
        st.write("Deep insights into transaction risks and patterns.")

def detection_page():
    st.title("🔍 Fraud Detection")
    
    pipeline = load_pipeline()
    
    if pipeline is None:
        st.error("⚠️ Model not found! Please run 'python model_training.py' first.")
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
        ohe = pipeline.named_steps['preprocessing'].named_transformers_['categorical']
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
            
            # Pass raw DataFrame directly to pipeline
            # Pipeline handles feature engineering and preprocessing automatically
            prob = pipeline.predict_proba(input_df)[0]
            fraud_probability = prob[1]  # Probability of fraudulent class
            safe_probability = prob[0]   # Probability of legitimate class
            
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



def dashboard_page():
    st.title("📈 Model Performance Dashboard")
    
    if os.path.exists("model.pkl"):
        st.success("✅ Model is trained and loaded.")
        
        try:
            pipeline = load_pipeline()
            if pipeline is None:
                st.warning("Could not load pipeline.")
                return
            
            # Extract feature importance from the classifier in the pipeline
            classifier = pipeline.named_steps['classifier']
            feature_names = list(pipeline.feature_names_in_)
            importances = classifier.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance",
                template="plotly_dark",
                color='Importance',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not display feature importance: {e}")
        
        if st.button("Retrain Model"):
            with st.spinner("Retraining model... this may take a while."):
                import subprocess
                result = subprocess.run(["python", "model_training.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Model retrained successfully!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"Training failed: {result.stderr}")
    else:
        st.warning("Model not found. Please train the model first by running 'python model_training.py'")

# Navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Fraud Detection", "Dashboard"], index=["Home", "Fraud Detection", "Dashboard"].index(st.session_state.page))
st.session_state.page = page

if page == "Home":
    home_page()
elif page == "Fraud Detection":
    detection_page()
elif page == "Dashboard":
    dashboard_page()
