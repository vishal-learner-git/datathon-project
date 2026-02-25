import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import base64
import traceback

# Set page configuration
st.set_page_config(page_title="ML Sales Prediction & Warehouse Optimization", 
                   layout="wide",
                   initial_sidebar_state="expanded")

# Add custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def create_download_link(df, filename, link_text):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the uploaded data file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Convert date column to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            # Extract additional time features
            df['Day'] = df['Date'].dt.day
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Quarter'] = df['Date'].dt.quarter
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(traceback.format_exc())
        return None

def create_default_machine_specs():
    """Create default machine specifications if not uploaded"""
    machine_specs = pd.DataFrame({
        'Infrastructure_Machineries': [
            'Compactors', 
            'Loaders (Wheeled)', 
            'Skid Steer Loaders', 
            'Backhoe Loader', 
            'Excavators(crawler)', 
            'Tele Handlers'
        ],
        'Volume_Cubic_Meters': [12, 20, 8, 15, 25, 18],  
        'Storage_Cost_Per_Unit': [100, 150, 80, 120, 200, 130],  
        'Supply_Lead_Time_Days': [7, 10, 5, 8, 12, 9],  
        'Profit_Per_Unit': [500, 800, 400, 650, 1200, 700]  
    })
    return machine_specs

# Main app
def main():
    st.markdown('<div class="title">Machine Learning Sales Prediction & Warehouse Optimization</div>', unsafe_allow_html=True)
    
    # Create sidebar for file uploads and configuration
    with st.sidebar:
        st.header("Data Upload")
        
        st.subheader("1. Upload Raw Dataset")
        raw_data_file = st.file_uploader("Upload Raw Input Dataset (CSV)", type=['csv'])
        
        st.subheader("2. Upload Cleaned Dataset")
        cleaned_data_file = st.file_uploader("Upload Cleaned Dataset (CSV)", type=['csv'])
        
        st.header("Configuration")
        
        st.subheader("1. Machine Specifications")
        use_default_specs = st.checkbox("Use Default Machine Specifications", value=True)
        
        if not use_default_specs:
            specs_file = st.file_uploader("Upload Machine Specifications CSV", type=['csv'])
        else:
            specs_file = None
            st.info("Using default machine specifications")
            
        st.subheader("2. Warehouse Configuration")
        warehouse_volume = st.number_input("Warehouse Volume (cubic meters)", min_value=100, value=5000, step=100)
        
        st.subheader("3. ML Model Configuration")
        model_type = st.selectbox("Select ML Model", ["Random Forest", "Gradient Boosting"])
        
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
            max_depth = st.slider("Max Depth", 3, 30, 10, 1)
        else:
            n_estimators = st.slider("Number of Estimators", 10, 200, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            
        st.subheader("4. Prediction Configuration")
        forecast_days = st.slider("Forecast Days", 7, 90, 30, 1)
        
        run_button = st.button("Run Analysis")
    
    # Main content area
    if raw_data_file is None or cleaned_data_file is None:
        # Display landing page with instructions
        st.markdown("""
        <div class="info-container">
            <h3>Welcome to the Sales Prediction & Warehouse Optimization Tool</h3>
            <p>This application helps you predict future sales of machinery and optimize your warehouse storage to minimize costs.</p>
            <p><b>To get started:</b></p>
            <ol>
                <li>Upload your raw input dataset CSV file</li>
                <li>Upload your cleaned dataset CSV file</li>
                <li>Configure your machine specifications or use the defaults</li>
                <li>Set your warehouse volume and model parameters</li>
                <li>Click "Run Analysis" to generate predictions and optimize storage</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Display expected data formats
        st.markdown('<div class="subtitle">Expected Data Formats</div>', unsafe_allow_html=True)
        
        st.markdown("### Raw Input Dataset")
        st.markdown("""
        Your raw input dataset should contain sales data with the following columns:
        - Customer_Id: Unique identifier for customers
        - Date: Date of sales in YYYY-MM-DD format
        - Daily_Sales_Percentage: Normalized sales metric
        - Market_Share: Market share percentage
        - Political: Binary indicator for political influence
        - Marketing: Binary indicator for marketing activity
        - Budget: Numeric budget value
        - Infrastructure_Machineries: Type of machinery sold
        - Region: Geographic location
        - Predicted_Daily_Sales_Quantity: Historical sales quantity
        """)
        
        st.markdown("### Cleaned Dataset")
        st.markdown("""
        Your cleaned dataset should have the same structure as the raw input, but with:
        - Missing values handled
        - Outliers addressed
        - Any necessary transformations applied
        - The target variable (Predicted_Daily_Sales_Quantity) should be present
        """)
        
    elif run_button:
        # Run the full analysis pipeline
        with st.spinner("Loading and preprocessing data..."):
            # Load raw data
            raw_df = load_and_preprocess_data(raw_data_file)
            if raw_df is not None:
                st.success(f"Raw data loaded successfully: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
            else:
                st.error("Failed to load raw data. Please check your CSV file format.")
                return
            
            # Load cleaned data
            df = load_and_preprocess_data(cleaned_data_file)
            
            if df is not None:
                st.success(f"Cleaned data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            else:
                st.error("Failed to load cleaned data. Please check your CSV file format.")
                return
                
            # Show a sample of the data
            st.markdown('<div class="subtitle">Raw Input Data Sample</div>', unsafe_allow_html=True)
            st.dataframe(raw_df.head())
            
            st.markdown('<div class="subtitle">Cleaned Data Sample</div>', unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Load or create machine specs
            if not use_default_specs and specs_file is not None:
                try:
                    machine_specs = pd.read_csv(specs_file)
                    st.success("Machine specifications loaded successfully")
                except Exception as e:
                    st.error(f"Error loading machine specifications: {e}")
                    machine_specs = create_default_machine_specs()
                    st.warning("Using default machine specifications instead")
            else:
                machine_specs = create_default_machine_specs()
                
            st.markdown('<div class="subtitle">Machine Specifications</div>', unsafe_allow_html=True)
            st.dataframe(machine_specs)
            
            # Create tabs for different sections
            tab1, tab2 = st.tabs(["1. ML Sales Prediction", "2. Warehouse Optimization"])
            
            with tab1:
                try:
                    st.markdown('<div class="subtitle">Machine Learning Sales Prediction Model</div>', unsafe_allow_html=True)
                    
                    # Check if required columns exist
                    required_columns = ['Predicted_Daily_Sales_Quantity', 'Customer_Id', 'Date', 
                                       'Infrastructure_Machineries', 'Region']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns in your data: {', '.join(missing_columns)}")
                        return
                    
                    # Prepare data for modeling
                    X = df.drop(['Predicted_Daily_Sales_Quantity', 'Customer_Id', 'Date'], axis=1, errors='ignore')
                    y = df['Predicted_Daily_Sales_Quantity']
                    
                    # Check what columns actually exist in the data
                    available_numerical = [col for col in ['Daily_Sales_Percentage', 'Market_Share', 'Political', 
                                                         'Marketing', 'Budget', 'Day', 'Month', 'Year', 
                                                         'DayOfWeek', 'Quarter'] if col in X.columns]
                    
                    available_categorical = [col for col in ['Infrastructure_Machineries', 'Region'] 
                                           if col in X.columns]
                    
                    if not available_numerical or not available_categorical:
                        st.error("Missing some expected feature columns. Check your data format.")
                        st.write("Available numerical columns:", available_numerical)
                        st.write("Available categorical columns:", available_categorical)
                        return
                    
                    # Define preprocessor with available columns
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), available_numerical),
                            ('cat', OneHotEncoder(handle_unknown='ignore'), available_categorical)
                        ],
                        remainder='passthrough'  # Include any other columns
                    )
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Create and train model
                    with st.spinner("Training machine learning model..."):
                        if model_type == "Random Forest":
                            model = Pipeline([
                                ('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=n_estimators, 
                                                                 max_depth=max_depth, 
                                                                 random_state=42))
                            ])
                        else:
                            model = Pipeline([
                                ('preprocessor', preprocessor),
                                ('regressor', GradientBoostingRegressor(n_estimators=n_estimators, 
                                                                     learning_rate=learning_rate, 
                                                                     random_state=42))
                            ])
                        
                        model.fit(X_train, y_train)
                        
                        # Evaluate model
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        # Display metrics
                        st.markdown("### Model Performance Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                        with col2:
                            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                        with col3:
                            st.metric("R² Score", f"{r2:.4f}")
                        
                        # Generate feature importance plot
                        st.markdown("### Feature Importance")
                        try:
                            if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                                # Get feature names after one-hot encoding
                                ohe = model.named_steps['preprocessor'].transformers_[1][1]
                                numerical_names = available_numerical
                                
                                # Get categorical feature names after OHE
                                categorical_feature_names = list(ohe.get_feature_names_out(available_categorical))
                                
                                # Combine all feature names
                                feature_names = numerical_names + categorical_feature_names
                                
                                importances = model.named_steps['regressor'].feature_importances_
                                
                                # Plot top 15 features only to avoid overcrowding
                                indices = np.argsort(importances)[::-1][:15]
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.bar(range(len(indices)), importances[indices], align='center')
                                ax.set_xticks(range(len(indices)))
                                ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
                                ax.set_xlabel('Features')
                                ax.set_ylabel('Importance')
                                ax.set_title('Top 15 Feature Importance')
                                st.pyplot(fig)
                            else:
                                st.info("Feature importance not available for this model type.")
                        except Exception as e:
                            st.error(f"Error generating feature importance: {e}")
                            st.error(traceback.format_exc())
                        
                        # Generate future predictions
                        st.markdown("### Future Sales Predictions")
                        with st.spinner("Generating future sales predictions..."):
                            try:
                                # Check if Date column is properly formatted
                                if 'Date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Date']):
                                    st.error("Date column missing or not properly formatted.")
                                    return
                                
                                last_date = df['Date'].max()
                                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                                          periods=forecast_days, freq='D')
                                
                                # Initialize list before the loop
                                future_data_list = []

                                # Loop through each machine-region combo
                                for machine_type in machine_specs['Infrastructure_Machineries']:
                                    for region in df['Region'].unique():
                                        # Get the most recent data for this machine and region
                                        mask = ((df['Infrastructure_Machineries'] == machine_type) & 
                                                (df['Region'] == region))

                                        if mask.any():
                                            recent_data = df[mask].sort_values('Date', ascending=False).iloc[0:1]

                                            # Create template row by dropping the target variable
                                            template_row = recent_data.drop('Predicted_Daily_Sales_Quantity', axis=1).copy()

                                            for future_date in future_dates:
                                                future_row = template_row.copy()
                                                future_row['Date'] = future_date
                                                future_row['Day'] = future_date.day
                                                future_row['Month'] = future_date.month
                                                future_row['Year'] = future_date.year
                                                future_row['DayOfWeek'] = future_date.dayofweek
                                                future_row['Quarter'] = future_date.quarter
                                                future_data_list.append(future_row)

                            except Exception as e:
                                st.error(f"An error occurred: {e}")

                            # Check if future_data_list is empty before proceeding
                            if not future_data_list:
                                st.error("Failed to generate future data. Check your data format.")
                            else:
                                future_df = pd.concat(future_data_list, ignore_index=True)
                                # Make predictions for future data
                                future_predictions = model.predict(future_df)
                                future_df['Predicted_Daily_Sales_Quantity'] = future_predictions
                                st.markdown("### Future Predictions")
                                st.dataframe(future_df[['Date', 'Infrastructure_Machineries', 'Region', 'Predicted_Daily_Sales_Quantity']])

            with tab2:
                st.markdown('<div class="subtitle">Warehouse Optimization</div>', unsafe_allow_html=True)
                # Add your warehouse optimization logic here

if __name__ == "__main__":
    main()