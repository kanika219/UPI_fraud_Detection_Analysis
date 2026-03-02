import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
import tensorflow as tf

# Suppress TensorFlow warnings for cleaner output
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

from feature_engineering import BalanceFeatureEngineer

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


@st.cache_resource
def load_artifacts():
    """Load preprocessing pipeline and ANN model.

    Returns tuple (preprocessor, ann_model) or (None,None) if missing.
    Uses st.cache_resource to prevent re-loading on every run.
    """
    # ensure files exist before attempting load
    if not os.path.isfile("preprocessor.pkl") or not os.path.isfile("model.h5"):
        # trigger training in background
        st.warning("Model artifacts are not available. Training will start automatically in the background. This may take several minutes.")
        start_training_background()
        return None, None

    try:
        preproc = joblib.load("preprocessor.pkl")
    except Exception as e:
        st.error(f"Error loading preprocessing pipeline: {e}")
        return None, None
    try:
        ann = load_model("model.h5")
    except Exception as e:
        st.error(f"Error loading ANN model: {e}")
        return None, None
    
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



def dashboard_page():
    st.title("📊 Dashboard & Analytics")
    
    # Load dataset for EDA
    @st.cache_data
    def load_dataset():
        return pd.read_csv('AIML Dataset.csv')
    
    try:
        df = load_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "EDA Visualizations", "Model Architecture", "How It Works"])
    
    # ==================== TAB 1: Dataset Overview ====================
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            fraud_count = df['isFraud'].sum()
            st.metric("Fraudulent Cases", f"{fraud_count:,}")
        with col3:
            fraud_pct = (df['isFraud'].sum() / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_pct:.2f}%")
        with col4:
            legitimate_count = len(df) - df['isFraud'].sum()
            st.metric("Legitimate Cases", f"{legitimate_count:,}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Statistics")
            stats_df = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("Transaction Types")
            type_counts = df['type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index, 
                        title="Distribution of Transaction Types",
                        color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: EDA Visualizations ====================
    with tab2:
        st.header("Exploratory Data Analysis")
        
        # Class Imbalance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Class Distribution")
            fraud_dist = df['isFraud'].value_counts()
            fig = go.Figure(data=[
                go.Bar(x=['Legitimate', 'Fraudulent'], 
                       y=[fraud_dist[0], fraud_dist[1]],
                       marker_color=['#28a745', '#dc3545'],
                       text=[f"{fraud_dist[0]:,}", f"{fraud_dist[1]:,}"],
                       textposition='auto')
            ])
            fig.update_layout(title_text="Transaction Class Distribution",
                            template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Fraud Percentage by Transaction Type")
            fraud_by_type = df.groupby('type')['isFraud'].apply(lambda x: (x.sum() / len(x)) * 100).sort_values(ascending=False)
            fig = px.bar(x=fraud_by_type.values, y=fraud_by_type.index,
                        orientation='h', title="Fraud Rate by Transaction Type",
                        color=fraud_by_type.values,
                        color_continuous_scale='Reds')
            fig.update_layout(template="plotly_dark", height=400, xaxis_title="Fraud Rate (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction Amount Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Amount Distribution")
            # Remove extreme outliers for better visualization
            amount_data = df[df['amount'] < df['amount'].quantile(0.99)]
            fig = px.histogram(amount_data, x='amount', nbins=50,
                              title="Distribution of Transaction Amounts",
                              color_discrete_sequence=['#1f77b4'])
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Average Amount by Type & Fraud Status")
            avg_amount = df.groupby(['type', 'isFraud'])['amount'].mean().reset_index()
            avg_amount['isFraud'] = avg_amount['isFraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
            fig = px.bar(avg_amount, x='type', y='amount', color='isFraud',
                        barmode='group', title="Average Transaction Amount",
                        color_discrete_map={'Legitimate': '#28a745', 'Fraudulent': '#dc3545'})
            fig.update_layout(template="plotly_dark", height=400, yaxis_title="Average Amount")
            st.plotly_chart(fig, use_container_width=True)
        
        # Balance Analysis
        st.subheader("Account Balance Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Legitimate vs Fraudulent - Opening Balance Distribution**")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['isFraud'] == 0]['oldbalanceOrg'],
                name='Legitimate',
                opacity=0.7,
                marker_color='#28a745',
                nbinsx=50
            ))
            fig.add_trace(go.Histogram(
                x=df[df['isFraud'] == 1]['oldbalanceOrg'],
                name='Fraudulent',
                opacity=0.7,
                marker_color='#dc3545',
                nbinsx=50
            ))
            fig.update_layout(template="plotly_dark", barmode='overlay', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Balance Difference Analysis**")
            df_viz = df.copy()
            df_viz['balanceOrigDiff'] = df_viz['oldbalanceOrg'] - df_viz['amount'] - df_viz['newbalanceOrig']
            df_viz['balanceDiffCategory'] = df_viz['balanceOrigDiff'].apply(
                lambda x: 'Balanced' if x == 0 else 'Unbalanced'
            )
            imbalance_counts = df_viz.groupby(['balanceDiffCategory', 'isFraud']).size().reset_index(name='Count')
            imbalance_counts['isFraud'] = imbalance_counts['isFraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
            fig = px.bar(imbalance_counts, x='balanceDiffCategory', y='Count', color='isFraud',
                        barmode='group', title="Balance Consistency by Fraud Status",
                        color_discrete_map={'Legitimate': '#28a745', 'Fraudulent': '#dc3545'})
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: Model Architecture ====================
    with tab3:
        st.header("Model Architecture & Details")
        
        st.subheader("🧠 Artificial Neural Network (ANN)")
        st.write("""
        The fraud detection model is built using a deep learning approach with an **Artificial Neural Network (ANN)**.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            model_info = {
                "Algorithm": "Deep Neural Network (ANN)",
                "Framework": "TensorFlow/Keras",
                "Activation Function": "ReLU (Hidden), Sigmoid (Output)",
                "Loss Function": "Binary Crossentropy",
                "Optimizer": "Adam",
                "Class Weight": "Balanced (handles imbalance)"
            }
            for key, value in model_info.items():
                st.info(f"**{key}**: {value}")
        
        with col2:
            st.subheader("Input Features")
            features_list = [
                "step - Time step of the transaction (in hours)",
                "type - Type of transaction (categorical)",
                "amount - Transaction amount",
                "oldbalanceOrg - Origin account balance before transaction",
                "newbalanceOrig - Origin account balance after transaction",
                "oldbalanceDest - Destination account balance before transaction",
                "newbalanceDest - Destination account balance after transaction"
            ]
            for i, feature in enumerate(features_list, 1):
                st.write(f"{i}. {feature}")
        
        st.divider()
        
        st.subheader("📐 Network Architecture")
        st.markdown("""
        ```
        Input Layer (7 features)
             ↓
        Dense Layer (64 units) + ReLU + Dropout(0.3)
             ↓
        Dense Layer (32 units) + ReLU + Dropout(0.3)
             ↓
        Dense Layer (16 units) + ReLU + Dropout(0.2)
             ↓
        Dense Layer (8 units) + ReLU
             ↓
        Output Layer (1 unit) + Sigmoid
             ↓
        Binary Classification (Fraud/Legitimate)
        ```
        """)
        
        st.subheader("📊 Training Details")
        training_info = {
            "Training Epochs": "20",
            "Batch Size": "32",
            "Validation Split": "20%",
            "Class Imbalance Handling": "Balanced weights",
            "Dropout Rate": "0.2-0.3 (prevents overfitting)",
            "Early Stopping": "Monitors validation loss"
        }
        for key, value in training_info.items():
            st.write(f"**{key}**: {value}")
    
    # ==================== TAB 4: How It Works ====================
    with tab4:
        st.header("How the Fraud Detection System Works")
        
        st.subheader("🔄 Step-by-Step Process")
        
        step1, step2, step3 = st.columns(3, gap="small")
        
        with step1:
            st.markdown("""
            ### 1️⃣ Data Input
            - You enter transaction details
            - Step, Type, Amount, Balance info
            - System validates all inputs
            """)
        
        with step2:
            st.markdown("""
            ### 2️⃣ Feature Engineering
            - Creates derived features
            - Calculates balance inconsistencies
            - detects unusual patterns
            """)
        
        with step3:
            st.markdown("""
            ### 3️⃣ Model Prediction
            - Neural network processes data
            - Generates fraud probability
            - Applies threshold (default: 20%)
            """)
        
        st.divider()
        
        st.subheader("🚨 Fraud Detection Logic")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Key Fraud Indicators:
            
            1. **Balance Inconsistency**
               - Amount sent ≠ balance change in origin account
               - Amount received ≠ balance change in destination
               - Strong fraud signal
            
            2. **Transaction Type Patterns**
               - CASH_OUT and TRANSFER have higher fraud rates
               - Suspicious transaction types flagged
            
            3. **Amount Anomalies**
               - Unusual transaction amounts
               - Out-of-pattern transfers
            """)
        
        with col2:
            st.markdown("""
            #### AI Decision Making:
            
            1. **Neural Network Analysis**
               - Learns complex patterns from 6.36M transactions
               - Identifies non-obvious fraud signatures
               - Adapts to evolving fraud tactics
            
            2. **Probability Scoring**
               - Returns 0-100% fraud probability
               - Customizable threshold (you can adjust)
               - Higher score = higher risk
            
            3. **Class Balance Handling**
               - Addresses data imbalance (0.13% fraud rate)
               - Prevents bias toward legitimate transactions
            """)
        
        st.divider()
        
        st.subheader("📈 Why Neural Networks?")
        
        reasons = {
            "Non-Linear Patterns": "Can detect complex fraud patterns traditional methods miss",
            "Feature Interactions": "Learns how features interact (e.g., type + amount + balance)",
            "Scalability": "Handles millions of transactions efficiently",
            "Adaptation": "Can be retrained with new fraud patterns",
            "Probabilistic Output": "Provides confidence scores, not just binary decisions"
        }
        
        for reason, explanation in reasons.items():
            st.write(f"✅ **{reason}**: {explanation}")
        
        st.divider()
        
        st.subheader("⚙️ Preprocessing Pipeline")
        
        st.markdown("""
        Before the neural network processes data:
        
        1. **Categorical Encoding** - Transaction types converted to numeric format
        2. **Feature Engineering** - Balance differences calculated
        3. **Feature Scaling** - All features normalized to similar ranges (StandardScaler)
        4. **Pipeline Caching** - Preprocessing saved for consistent predictions
        
        This ensures the model receives properly formatted, scaled data every time.
        """)
        
        st.divider()
        
        st.subheader("📋 Dataset Statistics Used")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Total Transactions**: {len(df):,}")
            st.write(f"**Fraudulent**: {df['isFraud'].sum():,} ({(df['isFraud'].sum()/len(df)*100):.2f}%)")
            st.write(f"**Legitimate**: {len(df) - df['isFraud'].sum():,}")
        
        with col2:
            st.write(f"**Transaction Types**: {df['type'].nunique()}")
            st.write(f"**Avg Amount**: ${df['amount'].mean():,.2f}")
            st.write(f"**Max Amount**: ${df['amount'].max():,.2f}")


# Navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Fraud Detection"], index=["Home", "Dashboard", "Fraud Detection"].index(st.session_state.page))
st.session_state.page = page

if page == "Home":
    home_page()
elif page == "Fraud Detection":
    detection_page()
elif page == "Dashboard":
    dashboard_page()

if __name__ == "__main__":
    pass


