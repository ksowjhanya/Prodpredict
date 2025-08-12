import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import io
from utils.data_processor import DataProcessor
from utils.ml_models import ProductivityPredictor
from utils.visualizations import DashboardVisualizer
from utils.report_generator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="AI Employee Productivity Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful UI
st.markdown("""
<style>
    /* Main background styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .stAlert > div {
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric > div {
        color: white !important;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    
    .element-container:has(.stButton) {
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .css-17eq0hr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .css-1544g2n {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Sidebar content */
    .sidebar .sidebar-content {
        background: rgba(102, 126, 234, 0.1) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Card styling with enhanced backgrounds */
    .info-card {
        background: linear-gradient(135deg, rgba(255, 236, 210, 0.9) 0%, rgba(252, 182, 159, 0.9) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(252, 182, 159, 0.4);
        border-left: 5px solid #FF6B6B;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(168, 237, 234, 0.9) 0%, rgba(254, 214, 227, 0.9) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(168, 237, 234, 0.4);
        border-left: 5px solid #6BCF7F;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(255, 234, 167, 0.9) 0%, rgba(250, 177, 160, 0.9) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(255, 234, 167, 0.4);
        border-left: 5px solid #FFD93D;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Enhanced metric styling */
    .stMetric {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Navigation pills */
    .nav-pill {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(79, 172, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Main title with colorful header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI-Powered Employee Productivity Forecasting System</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar with colorful styling
st.sidebar.markdown("""
<style>
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🎛️ Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio("Select Page", [
    "📊 Dashboard", 
    "📁 Data Upload", 
    "🤖 Model Training", 
    "🔮 Predictions", 
    "📈 Analytics",
    "📋 Reports"
])

# Add status indicators in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Status")

if st.session_state.data is not None:
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.error("❌ No Data")

if st.session_state.model is not None:
    st.sidebar.success("✅ Model Trained")
else:
    st.sidebar.warning("⚠️ Model Not Trained")

if st.session_state.predictions is not None:
    st.sidebar.success("✅ Predictions Ready")
else:
    st.sidebar.info("ℹ️ No Predictions")

# Initialize components
data_processor = DataProcessor()
predictor = ProductivityPredictor()
visualizer = DashboardVisualizer()
report_generator = ReportGenerator()

if page == "📊 Dashboard":
    st.header("🎯 Dashboard Overview")
    
    if st.session_state.processed_data is None:
        # Welcome screen with colorful cards
        st.markdown("""
        <div class="info-card">
            <h3>🚀 Welcome to AI Employee Productivity Forecasting</h3>
            <p>This powerful system helps you predict employee productivity and identify potential risks. Here's what you can do:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-card">
                <h4>📁 Upload Data</h4>
                <p>Start by uploading your employee performance data in CSV or Excel format.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-card">
                <h4>🤖 Train Models</h4>
                <p>Use machine learning to build predictive models for productivity forecasting.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h4>🔮 Make Predictions</h4>
                <p>Generate future productivity forecasts and identify at-risk employees.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("### ✨ Key Features")
        
        features_col1, features_col2 = st.columns(2)
        
        with features_col1:
            st.markdown("""
            - 📊 **Interactive Analytics Dashboard**
            - 🎯 **Risk Assessment & Alerts**
            - 📈 **Performance Trend Analysis**
            - 🏢 **Department Comparisons**
            """)
        
        with features_col2:
            st.markdown("""
            - 🤖 **Advanced ML Models** (Random Forest, Gradient Boosting)
            - 📋 **Comprehensive Reports**
            - 💾 **CSV/Excel Export**
            - ⚡ **Real-time Predictions**
            """)
        
        # Quick start section
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        
        if st.button("🎲 Generate Sample Data", type="primary"):
            # Create sample data
            import random
            from datetime import datetime, timedelta
            
            # Generate sample employee data
            employees = [f"EMP_{i:03d}" for i in range(1, 51)]  # 50 employees
            departments = ["Engineering", "Sales", "Marketing", "HR", "Finance"]
            
            sample_data = []
            start_date = datetime.now() - timedelta(days=90)
            
            for employee in employees:
                dept = random.choice(departments)
                base_productivity = random.uniform(60, 95)
                
                for day in range(90):
                    current_date = start_date + timedelta(days=day)
                    
                    # Add some randomness and trends
                    daily_variation = random.uniform(-10, 10)
                    productivity = max(0, min(100, base_productivity + daily_variation))
                    
                    sample_data.append({
                        'employee_id': employee,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'productivity_score': round(productivity, 2),
                        'department': dept,
                        'hours_worked': random.uniform(6, 10),
                        'tasks_completed': random.randint(3, 15),
                        'efficiency_rating': random.uniform(1, 5)
                    })
            
            sample_df = pd.DataFrame(sample_data)
            processed_df = data_processor.preprocess_data(sample_df)
            
            if processed_df is not None:
                st.session_state.data = sample_df
                st.session_state.processed_data = processed_df
                st.success("🎉 Sample data generated successfully! Navigate to other pages to explore features.")
                st.rerun()
    
    else:
        # Dashboard with data
        df = st.session_state.processed_data
        
        st.markdown("""
        <div class="success-card">
            <h3>📊 Data Overview</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("👥 Total Employees", df['employee_id'].nunique())
        with col2:
            st.metric("📅 Total Records", len(df))
        with col3:
            st.metric("⭐ Avg Productivity", f"{df['productivity_score'].mean():.1f}")
        with col4:
            low_performers = len(df[df['productivity_score'] < 70])
            st.metric("⚠️ Low Performers", low_performers)
        with col5:
            if 'department' in df.columns:
                st.metric("🏢 Departments", df['department'].nunique())
            else:
                st.metric("🏢 Departments", "N/A")
        
        # Recent trends
        st.markdown("### 📈 Recent Performance Trends")
        trend_fig = visualizer.plot_performance_trends(df)
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Model status
        if st.session_state.model is not None:
            st.markdown("""
            <div class="success-card">
                <h4>✅ Model Status: Ready</h4>
                <p>Your ML model is trained and ready for predictions!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ Model Status: Not Trained</h4>
                <p>Please train a model to start making predictions.</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📁 Data Upload":
    st.header("Data Upload & Validation")
    
    # File upload with multiple format support
    st.markdown("### 📂 Supported File Formats")
    st.info("CSV, Excel (XLSX/XLS), JSON, XML, TSV, TXT with various delimiters")
    
    uploaded_file = st.file_uploader(
        "Upload Employee Performance Data", 
        type=['csv', 'xlsx', 'xls', 'json', 'xml', 'tsv', 'txt'], 
        help="Upload data file in supported format with employee performance data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data using advanced processor
            df = data_processor.load_file(uploaded_file)
            
            if df is None:
                st.error("Failed to load file. Please check the format and try again.")
            else:
                # Data preview
                st.subheader("📋 Data Preview")
                st.markdown('<div class="info-card">', unsafe_allow_html=True)
                st.dataframe(df.head(10))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Data validation
                st.subheader("Data Validation")
                
                # Check required columns
                required_columns = ['employee_id', 'date', 'productivity_score']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                    st.info("Required columns: employee_id, date, productivity_score")
                    st.info("Optional columns: department, hours_worked, tasks_completed, efficiency_rating, project_deadline")
                else:
                    # Process data with advanced feature engineering
                    processed_df = data_processor.preprocess_data(df)
                    
                    if processed_df is not None:
                        # Apply advanced feature engineering
                        st.info("Applying advanced feature engineering...")
                        processed_df = data_processor.advanced_feature_engineering(processed_df)
                    
                    if processed_df is not None:
                        st.session_state.data = df
                        st.session_state.processed_data = processed_df
                        
                        # Display data statistics
                        st.subheader("📊 Data Statistics")
                        st.markdown('<div class="success-card">', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Records", len(processed_df))
                        with col2:
                            st.metric("Unique Employees", processed_df['employee_id'].nunique())
                        with col3:
                            st.metric("Date Range", f"{processed_df['date'].min().strftime('%Y-%m-%d')} to {processed_df['date'].max().strftime('%Y-%m-%d')}")
                        with col4:
                            st.metric("Avg Productivity", f"{processed_df['productivity_score'].mean():.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Data quality report
                        st.subheader("📋 Data Quality Report")
                        st.markdown('<div class="info-card">', unsafe_allow_html=True)
                        quality_report = data_processor.generate_quality_report(processed_df)
                        st.json(quality_report)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

elif page == "🤖 Model Training":
    st.header("Machine Learning Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and validate data first!")
    else:
        df = st.session_state.processed_data
        
        # Advanced Model configuration
        st.subheader("🤖 Advanced Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get available models
            available_models = list(predictor.available_models.keys())
            model_display_names = [predictor.available_models[key] for key in available_models]
            
            selected_display = st.selectbox("Select Model Type", model_display_names)
            model_type = available_models[model_display_names.index(selected_display)]
            
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        
        with col2:
            use_feature_selection = st.checkbox("Feature Selection", value=True, help="Automatically select best features")
            use_scaling = st.checkbox("Feature Scaling", value=True, help="Scale features for certain models")
            hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False, help="Optimize model parameters (slower)")
        
        with col3:
            forecast_days = st.number_input("Forecast Period (days)", 1, 90, 30)
            random_state = st.number_input("Random Seed", 1, 999, 42, help="For reproducible results")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Train model with advanced options
                    model_results = predictor.train_model(
                        df, 
                        model_type=model_type,
                        test_size=test_size,
                        random_state=random_state,
                        use_feature_selection=use_feature_selection,
                        use_scaling=use_scaling,
                        hyperparameter_tuning=hyperparameter_tuning
                    )
                    
                    if model_results:
                        st.session_state.model = model_results
                        
                        # Display training results
                        st.markdown("""
                        <div class="success-card">
                            <h4>✅ Model trained successfully!</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown('<div class="info-card">', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² Score", f"{model_results['r2_score']:.3f}")
                        with col2:
                            st.metric("MAE", f"{model_results['mae']:.3f}")
                        with col3:
                            st.metric("RMSE", f"{model_results['rmse']:.3f}")
                        with col4:
                            if 'mape' in model_results:
                                st.metric("MAPE", f"{model_results['mape']:.3f}")
                            else:
                                st.metric("MAPE", "N/A")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Additional metrics for advanced models
                        if 'cv_r2_mean' in model_results:
                            st.markdown('<div class="success-card">', unsafe_allow_html=True)
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("CV R² Mean", f"{model_results['cv_r2_mean']:.3f}")
                            with col2:
                                st.metric("CV R² Std", f"{model_results['cv_r2_std']:.3f}")
                            with col3:
                                overfitting = model_results.get('overfitting_score', 0)
                                color = "🟢" if overfitting < 0.1 else "🟡" if overfitting < 0.2 else "🔴"
                                st.metric("Overfitting", f"{color} {overfitting:.3f}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Feature importance
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame({
                            'feature': model_results['feature_names'],
                            'importance': model_results['feature_importance']
                        }).sort_values('importance', ascending=False)
                        
                        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                   title="Feature Importance in Productivity Prediction")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model performance
                        st.subheader("Model Performance")
                        performance_fig = visualizer.plot_model_performance(
                            model_results['y_test'], 
                            model_results['y_pred']
                        )
                        st.plotly_chart(performance_fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")

elif page == "🔮 Predictions":
    st.header("Productivity Predictions")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
    else:
        df = st.session_state.processed_data
        model_results = st.session_state.model
        
        # Prediction configuration
        st.subheader("Prediction Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_days = st.number_input("Days to Predict", 1, 90, 30)
            selected_employees = st.multiselect(
                "Select Employees (leave empty for all)",
                options=df['employee_id'].unique().tolist()
            )
        
        with col2:
            selected_departments = st.multiselect(
                "Select Departments (leave empty for all)",
                options=df['department'].unique().tolist() if 'department' in df.columns else []
            )
        
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Generating predictions..."):
                try:
                    # Generate predictions
                    predictions = predictor.generate_predictions(
                        df, 
                        model_results['model'], 
                        model_results['feature_names'],
                        days_ahead=prediction_days,
                        employee_filter=selected_employees if selected_employees else None,
                        department_filter=selected_departments if selected_departments else None
                    )
                    
                    if predictions is not None:
                        st.session_state.predictions = predictions
                        
                        # Display predictions summary
                        st.success(f"✅ Generated {len(predictions)} predictions!")
                        
                        # Predictions overview
                        st.subheader("🔮 Predictions Overview")
                        st.markdown('<div class="success-card">', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Avg Predicted Productivity", f"{predictions['predicted_productivity'].mean():.2f}")
                        with col2:
                            risk_count = len(predictions[predictions['risk_level'] == 'High'])
                            st.metric("High Risk Employees", risk_count)
                        with col3:
                            st.metric("Prediction Period", f"{prediction_days} days")
                        with col4:
                            st.metric("Employees Analyzed", predictions['employee_id'].nunique())
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Risk alerts
                        high_risk = predictions[predictions['risk_level'] == 'High']
                        if len(high_risk) > 0:
                            st.subheader("⚠️ Risk Alerts")
                            st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                            st.error(f"Found {len(high_risk)} high-risk predictions!")
                            st.dataframe(high_risk[['employee_id', 'department', 'prediction_date', 'predicted_productivity', 'risk_level']])
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Predictions table
                        st.subheader("Detailed Predictions")
                        st.dataframe(predictions)
                        
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {str(e)}")

elif page == "📈 Analytics":
    st.header("Advanced Analytics Dashboard")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.processed_data
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_employees = st.multiselect(
                "Filter by Employee",
                options=df['employee_id'].unique().tolist(),
                default=df['employee_id'].unique().tolist()[:5]  # Show first 5 by default
            )
        
        with col2:
            if 'department' in df.columns:
                selected_departments = st.multiselect(
                    "Filter by Department",
                    options=df['department'].unique().tolist(),
                    default=df['department'].unique().tolist()
                )
            else:
                selected_departments = []
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(df['date'].min().date(), df['date'].max().date()),
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )
        
        # Apply filters
        filtered_df = df.copy()
        if selected_employees:
            filtered_df = filtered_df[filtered_df['employee_id'].isin(selected_employees)]
        if selected_departments and 'department' in df.columns:
            filtered_df = filtered_df[filtered_df['department'].isin(selected_departments)]
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= date_range[0]) & 
                (filtered_df['date'].dt.date <= date_range[1])
            ]
        
        if len(filtered_df) == 0:
            st.warning("No data matches the selected filters.")
        else:
            # Performance trends
            st.subheader("Performance Trends")
            trend_fig = visualizer.plot_performance_trends(filtered_df)
            st.plotly_chart(trend_fig, use_container_width=True)
            
            # Department comparison
            if 'department' in filtered_df.columns:
                st.subheader("Department Performance Comparison")
                dept_fig = visualizer.plot_department_comparison(filtered_df)
                st.plotly_chart(dept_fig, use_container_width=True)
            
            # Individual employee performance
            st.subheader("Individual Employee Performance")
            employee_fig = visualizer.plot_employee_performance(filtered_df)
            st.plotly_chart(employee_fig, use_container_width=True)
            
            # Correlation analysis
            if len(filtered_df.select_dtypes(include=[np.number]).columns) > 2:
                st.subheader("Feature Correlation Analysis")
                corr_fig = visualizer.plot_correlation_matrix(filtered_df)
                st.plotly_chart(corr_fig, use_container_width=True)

elif page == "📋 Reports":
    st.header("Report Generation")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ Please generate predictions first!")
    else:
        predictions = st.session_state.predictions
        df = st.session_state.processed_data
        
        st.subheader("Generate Reports")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Get all available report types
            available_reports = list(report_generator.report_types.keys())
            report_display_names = [report_generator.report_types[key] for key in available_reports]
            
            selected_display = st.selectbox("📊 Report Type", report_display_names)
            report_type = available_reports[report_display_names.index(selected_display)]
            include_charts = st.checkbox("Include Visualizations", value=True)
        
        with col2:
            date_range = st.date_input(
                "Report Date Range",
                value=(predictions['prediction_date'].min().date(), predictions['prediction_date'].max().date())
            )
        
        if st.button("📊 Generate Report", type="primary"):
            try:
                # Generate report data
                report_data = report_generator.generate_report_data(
                    predictions, 
                    df, 
                    report_type,
                    date_range
                )
                
                # CSV download
                csv_buffer = io.StringIO()
                report_data.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name=f"{report_type.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Display report preview
                st.subheader("Report Preview")
                st.dataframe(report_data)
                
                # Summary statistics
                st.subheader("Report Summary")
                summary = report_generator.generate_summary_statistics(report_data, report_type)
                for key, value in summary.items():
                    st.metric(key.replace('_', ' ').title(), value)
                
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")

else:  # Dashboard
    st.header("Executive Dashboard")
    
    if st.session_state.processed_data is None:
        st.info("👆 Upload data to get started with the AI-powered productivity forecasting system.")
        
        # Show sample data structure
        st.subheader("Expected Data Format")
        sample_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003'],
            'date': ['2024-01-01', '2024-01-01', '2024-01-01'],
            'productivity_score': [85.5, 92.3, 78.9],
            'department': ['Sales', 'Engineering', 'Marketing'],
            'hours_worked': [8, 7.5, 8.5],
            'tasks_completed': [12, 8, 15],
            'efficiency_rating': [4.2, 4.8, 3.9]
        })
        st.dataframe(sample_data)
        
        st.markdown("""
        ### 🚀 Getting Started
        1. **Upload Data**: Navigate to 'Data Upload' and upload your employee performance data
        2. **Train Model**: Go to 'Model Training' to build your AI prediction model
        3. **Generate Predictions**: Use the 'Predictions' tab to forecast future productivity
        4. **Analyze Results**: Explore insights in the 'Analytics' dashboard
        5. **Export Reports**: Download comprehensive reports from the 'Reports' section
        
        ### 📊 Key Features
        - **AI-Powered Predictions**: Advanced ML models for accurate forecasting
        - **Risk Detection**: Automatic alerts for potential productivity issues
        - **Interactive Dashboards**: Real-time visualizations and analytics
        - **Export Capabilities**: Generate detailed reports in multiple formats
        """)
    else:
        # Show dashboard overview
        df = st.session_state.processed_data
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Employees", 
                df['employee_id'].nunique(),
                delta=None
            )
        
        with col2:
            avg_productivity = df['productivity_score'].mean()
            st.metric(
                "Avg Productivity", 
                f"{avg_productivity:.1f}",
                delta=f"{avg_productivity - 80:.1f}" if avg_productivity > 80 else f"{avg_productivity - 80:.1f}"
            )
        
        with col3:
            st.metric(
                "Data Points", 
                len(df),
                delta=None
            )
        
        with col4:
            date_range = (df['date'].max() - df['date'].min()).days
            st.metric(
                "Date Range (Days)", 
                date_range,
                delta=None
            )
        
        # Recent trends
        st.subheader("Recent Performance Trends")
        recent_data = df.tail(100)  # Last 100 records
        trend_fig = px.line(
            recent_data.groupby('date')['productivity_score'].mean().reset_index(),
            x='date', 
            y='productivity_score',
            title="Average Daily Productivity Trend"
        )
        st.plotly_chart(trend_fig, use_container_width=True)
        
        # Model status
        if st.session_state.model is not None:
            st.success("✅ ML Model is trained and ready!")
            model_info = st.session_state.model
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model R² Score", f"{model_info['r2_score']:.3f}")
            with col2:
                st.metric("Model MAE", f"{model_info['mae']:.3f}")
        else:
            st.info("ℹ️ Train an ML model to enable predictions and advanced analytics.")
        
        # Predictions status
        if st.session_state.predictions is not None:
            predictions = st.session_state.predictions
            high_risk = len(predictions[predictions['risk_level'] == 'High'])
            if high_risk > 0:
                st.warning(f"⚠️ {high_risk} employees are predicted to have high productivity risk!")
            else:
                st.success("✅ No high-risk productivity predictions detected.")

# Footer
st.markdown("---")
st.markdown("🚀 **AI Employee Productivity Forecasting System** | Powered by Machine Learning")
