import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# App Title
st.title("📈 Demand Forecasting & Inventory Optimization")

# File Uploads
cleaned_file = st.file_uploader("📂 Upload Cleaned Dataset", type=["csv"])
input_file = st.file_uploader("📂 Upload Input Dataset", type=["csv"])

# Start Processing after both files are uploaded
if cleaned_file and input_file:
    # Load datasets
    df = pd.read_csv(cleaned_file)
    df_input = pd.read_csv(input_file)
    
    # Strip whitespace from column names (safe for unexpected spaces)
    df.columns = df.columns.str.strip()
    df_input.columns = df_input.columns.str.strip()

    # Encode categorical columns in training data
    encoded_df = pd.get_dummies(df, columns=['Infrastructure_Machineries', 'Region'], drop_first=True)
    
    # Define features and label
    X = encoded_df.drop(columns=['Date', 'Daily_Sales_Quantity', 'Customer_Id'], errors='ignore')
    y = encoded_df['Daily_Sales_Quantity']
    
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # R² Score Display
    r2 = r2_score(y_test, rf_model.predict(X_test))
    st.success(f"✅ Model Trained - R² Score on Test Set: {r2:.2f}")

    # Input preprocessing
    df_input['Date'] = pd.to_datetime(df_input['Date'], errors='coerce')
    df_input = df_input.dropna(subset=['Date'])  # Drop rows with invalid dates

    # One-hot encode input
    encoded_input = pd.get_dummies(df_input, columns=['Infrastructure_Machineries', 'Region'], drop_first=True)
    
    # Add missing columns (if any)
    for col in set(X.columns) - set(encoded_input.columns):
        encoded_input[col] = 0
    encoded_input = encoded_input[X.columns]  # Ensure column order matches training

    # Predict demand
    predicted_sales = rf_model.predict(encoded_input)
    df_input['Predicted_Daily_Sales_Quantity'] = predicted_sales

    # Inventory Optimization Logic
    storage_requirements = {
        "Backhoe Loader": {"space": 15, "cost": 3000000},
        "Excavators(crawler)": {"space": 25, "cost": 5000000},
        "Loaders (Wheeled)": {"space": 20, "cost": 4000000},
        "Skid Steer Loaders": {"space": 10, "cost": 2000000},
        "Compactors": {"space": 12, "cost": 2500000},
        "Tele Handlers": {"space": 18, "cost": 3500000}
    }
    remaining_space = 5000
    avg_demand = df_input.groupby("Infrastructure_Machineries")['Predicted_Daily_Sales_Quantity'].mean()
    inventory_plan = {}

    for machine, demand in avg_demand.items():
        if machine in storage_requirements:
            space_per_unit = storage_requirements[machine]['space']
            max_units = remaining_space // space_per_unit
            optimal_units = min(int(round(demand)), max_units)
            inventory_plan[machine] = optimal_units
            remaining_space -= optimal_units * space_per_unit

    inventory_df = pd.DataFrame(list(inventory_plan.items()), columns=['Machinery', 'Optimal_Units'])

    # Output Section
    st.subheader("📊 Predicted Demand")
    st.write(df_input[['Date', 'Infrastructure_Machineries', 'Predicted_Daily_Sales_Quantity']])

    st.subheader("📦 Optimized Inventory Plan")
    st.write(inventory_df)

    # Download buttons
    st.download_button("⬇️ Download Predicted Demand", df_input.to_csv(index=False), "processed_data.csv", "text/csv")
    st.download_button("⬇️ Download Inventory Plan", inventory_df.to_csv(index=False), "inventory_plan.csv", "text/csv")

# Upload Instructions
st.info("ℹ️ Upload the **cleaned historical dataset** and **input dataset** to begin.")
