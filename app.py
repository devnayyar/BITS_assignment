import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Adult Income Prediction", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("🎯 Adult Income Classification Models")
st.markdown("**Student ID:** 2025aa05627")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio("Select Page:", 
                    ["📊 Overview", "🔄 Model Training", "📈 Evaluation & Metrics", "🧪 Test Predictions"])

# ==================== PAGE 1: OVERVIEW ====================
if page == "📊 Overview":
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝 Problem Statement")
        st.write("""
        Predict whether an adult earns more than $50,000 per year based on demographic 
        and employment information from the Adult Census Income dataset.
        """)
    
    with col2:
        st.header("📊 Dataset Description")
        st.write("""
        - **Total Records:** 48,843
        - **Features:** 14
        - **Target:** Income (<=50K or >50K)
        - **Class Distribution:** Imbalanced (binary classification)
        """)
    
    st.markdown("---")
    st.header("🤖 Models Implemented")
    
    models_info = {
        "Logistic Regression": "Linear classifier based on logistic function",
        "Decision Tree": "Tree-based model that learns if-then-else rules",
        "K-Nearest Neighbors": "Instance-based learning using k nearest neighbors",
        "Naive Bayes": "Probabilistic classifier based on Bayes' theorem",
        "Random Forest": "Ensemble of decision trees with majority voting",
        "XGBoost": "Gradient boosting ensemble method"
    }
    
    for model, desc in models_info.items():
        st.info(f"**{model}:** {desc}")
    
    st.markdown("---")
    st.header("📊 Evaluation Metrics")
    st.write("""
    - **Accuracy:** Overall correctness of predictions
    - **AUC Score:** Area under ROC curve (0 to 1)
    - **Precision:** Correctly predicted positive cases / Total predicted positive
    - **Recall:** Correctly predicted positive cases / Total actual positive
    - **F1 Score:** Harmonic mean of Precision and Recall
    - **Matthews Correlation Coefficient (MCC):** Correlation between predicted and actual
    """)

# ==================== PAGE 2: MODEL TRAINING ====================
elif page == "🔄 Model Training":
    st.header("🔄 Train Models on Adult Dataset")
    
    if st.button("🚀 Load Dataset & Train All Models", key="train_button"):
        with st.spinner("Loading dataset..."):
            try:
                df = pd.read_csv('adult.csv')
                st.success(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            except:
                st.error("❌ Error loading dataset. Ensure 'adult.csv' is in the working directory.")
                st.stop()
        
        with st.spinner("Preprocessing data..."):
            # Remove missing values
            data = df[~((df == '?').any(axis=1))].copy()
            st.info(f"Removed rows with '?': {df.shape[0] - data.shape[0]} rows | Remaining: {data.shape[0]} rows")
            
            X = data.drop('income', axis=1)
            y = data['income']
            
            # Encode target
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            
            # Encode categorical
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            st.success("✓ Data preprocessing completed!")
            st.info(f"Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]} | Features: {X_train.shape[1]}")
        
        # Store in session
        st.session_state.X_test = X_test
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.y_test = y_test
        st.session_state.le_target = le_target
        st.session_state.scaler = scaler
        
        # Train models
        models_dict = {}
        results = {}
        
        def calculate_metrics(y_true, y_pred, y_pred_proba=None):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else roc_auc_score(y_true, y_pred)
            return {'Accuracy': accuracy, 'AUC': auc, 'Precision': precision, 
                    'Recall': recall, 'F1': f1, 'MCC': mcc}
        
        with st.spinner("Training Logistic Regression..."):
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)
            y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
            results['Logistic Regression'] = calculate_metrics(y_test, y_pred, y_pred_proba)
            models_dict['Logistic Regression'] = lr
            st.success("✓ Logistic Regression trained")
        
        with st.spinner("Training Decision Tree..."):
            dt = DecisionTreeClassifier(max_depth=15, random_state=42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            y_pred_proba = dt.predict_proba(X_test)[:, 1]
            results['Decision Tree'] = calculate_metrics(y_test, y_pred, y_pred_proba)
            models_dict['Decision Tree'] = dt
            st.success("✓ Decision Tree trained")
        
        with st.spinner("Training KNN..."):
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]
            results['KNN'] = calculate_metrics(y_test, y_pred, y_pred_proba)
            models_dict['KNN'] = knn
            st.success("✓ KNN trained")
        
        with st.spinner("Training Naive Bayes..."):
            nb = GaussianNB()
            nb.fit(X_train_scaled, y_train)
            y_pred = nb.predict(X_test_scaled)
            y_pred_proba = nb.predict_proba(X_test_scaled)[:, 1]
            results['Naive Bayes'] = calculate_metrics(y_test, y_pred, y_pred_proba)
            models_dict['Naive Bayes'] = nb
            st.success("✓ Naive Bayes trained")
        
        with st.spinner("Training Random Forest..."):
            rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)[:, 1]
            results['Random Forest'] = calculate_metrics(y_test, y_pred, y_pred_proba)
            models_dict['Random Forest'] = rf
            st.success("✓ Random Forest trained")
        
        with st.spinner("Training XGBoost..."):
            xgb = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.1, 
                               random_state=42, eval_metric='logloss', verbosity=0)
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_test)
            y_pred_proba = xgb.predict_proba(X_test)[:, 1]
            results['XGBoost'] = calculate_metrics(y_test, y_pred, y_pred_proba)
            models_dict['XGBoost'] = xgb
            st.success("✓ XGBoost trained")
        
        st.session_state.models = models_dict
        st.session_state.results = results
        st.success("✅ All models trained successfully!")

# ==================== PAGE 3: EVALUATION & METRICS ====================
elif page == "📈 Evaluation & Metrics":
    st.header("📈 Model Performance Evaluation")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Please train models first on the 'Model Training' page")
    else:
        results_df = pd.DataFrame(st.session_state.results).T.round(4)
        
        st.subheader("📊 Performance Metrics Table")
        st.dataframe(results_df, use_container_width=True)
        
        # Best models
        st.subheader("🏆 Best Models by Metric")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Accuracy", results_df['Accuracy'].idxmax(), 
                     f"{results_df['Accuracy'].max():.4f}")
        with col2:
            st.metric("Best F1 Score", results_df['F1'].idxmax(), 
                     f"{results_df['F1'].max():.4f}")
        with col3:
            st.metric("Best AUC", results_df['AUC'].idxmax(), 
                     f"{results_df['AUC'].max():.4f}")
        
        # Visualizations
        st.subheader("📉 Performance Comparison Charts")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = results_df[metric].sort_values(ascending=False)
            bars = ax.bar(range(len(values)), values.values, color=colors[:len(values)])
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric}')
            ax.set_ylim(0, 1)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Heatmap
        st.subheader("🔥 Performance Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(results_df, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, linewidths=0.5, ax=ax)
        plt.title('Model Performance Heatmap')
        st.pyplot(fig)

# ==================== PAGE 4: TEST PREDICTIONS ====================
elif page == "🧪 Test Predictions":
    st.header("🧪 Make Predictions on Test Data")
    
    if 'models' not in st.session_state:
        st.warning("⚠️ Please train models first on the 'Model Training' page")
    else:
        selected_model = st.selectbox("Select a Model", list(st.session_state.models.keys()))
        
        if st.button("🔮 Generate Predictions"):
            model = st.session_state.models[selected_model]
            X_test = st.session_state.X_test
            X_test_scaled = st.session_state.X_test_scaled
            y_test = st.session_state.y_test
            
            # Choose right test set based on model
            if selected_model in ['Logistic Regression', 'KNN', 'Naive Bayes']:
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Decode predictions
            le_target = st.session_state.le_target
            y_pred_labels = le_target.inverse_transform(y_pred)
            y_test_labels = le_target.inverse_transform(y_test)
            
            # Display results
            st.subheader(f"Predictions from {selected_model}")
            results_pred_df = pd.DataFrame({
                'Actual': y_test_labels[:20],
                'Predicted': y_pred_labels[:20],
                'Correct': y_test_labels[:20] == y_pred_labels[:20]
            })
            st.dataframe(results_pred_df, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['<=50K', '>50K'],
                       yticklabels=['<=50K', '>50K'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title(f'Confusion Matrix - {selected_model}')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, 
                                         target_names=['<=50K', '>50K'])
            st.text(report)

st.markdown("---")
st.markdown("**Made with ❤️ for ML Assignment 2 | BITS Pilani**")
