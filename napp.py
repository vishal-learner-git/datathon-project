import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("Demand Forecasting & Inventory Optimization")

# Upload datasets
cleaned_file = st.file_uploader("Upload Cleaned Dataset", type=["csv"])
input_file = st.file_uploader("Upload Input Dataset", type=["csv"])

if cleaned_file and input_file:
    df = pd.read_csv(cleaned_file)
    df_input = pd.read_csv(input_file)

    # Ensure correct data types
    df['Infrastructure_Machineries'] = df['Infrastructure_Machineries'].astype(str)
    df_input['Infrastructure_Machineries'] = df_input['Infrastructure_Machineries'].astype(str)

    # Encode categorical features properly
    df = pd.get_dummies(df, columns=['Infrastructure_Machineries', 'Region'], drop_first=True)

    # Ensure necessary columns exist
    required_columns = ['Daily_Sales_Quantity', 'Customer_Id', 'Date']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Column {col} not found in the cleaned dataset.")
            st.stop()

    X = df.drop(columns=['Date', 'Daily_Sales_Quantity', 'Customer_Id'])
    y = df['Daily_Sales_Quantity']

    # Train Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prepare Input Data
    df_input.rename(columns={'Daily_Sales _Percentage': 'Daily_Sales_Percentage'}, inplace=True)
    df_input['Date'] = pd.to_datetime(df_input['Date'], errors='coerce')

    df_input_encoded = pd.get_dummies(df_input, columns=['Infrastructure_Machineries', 'Region'], drop_first=True)

    # Ensure only correct columns are used (prevent date-related issues)
    df_input_encoded = df_input_encoded.loc[:, df_input_encoded.columns.str.startswith("Infrastructure_Machineries_")]

    # Ensure df_input has the same features as training data
    for col in X.columns:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0  # Add missing columns with zero

    df_input_encoded = df_input_encoded[X.columns]  # Ensure column order matches

    # Predict Demand
    predicted_sales = rf_model.predict(df_input_encoded)
    df_input['Predicted_Daily_Sales_Quantity'] = [max(0, pred) for pred in predicted_sales]  # No negative values

    # ✅ **Fix Infrastructure_Machineries Reconstruction**
    if 'Infrastructure_Machineries' in df_input.columns:
        df_input['Infrastructure_Machineries'] = df_input['Infrastructure_Machineries']
    else:
        machinery_cols = [col for col in df_input_encoded.columns if 'Infrastructure_Machineries_' in col]
        df_input['Infrastructure_Machineries'] = df_input[machinery_cols].idxmax(axis=1).str.replace('Infrastructure_Machineries_', '')

    # Inventory Optimization
    storage_requirements = {
        "Backhoe Loader": {"space": 15, "cost": 3000000},
        "Excavators(crawler)": {"space": 25, "cost": 5000000},
        "Loaders (Wheeled)": {"space": 20, "cost": 4000000},
        "Skid Steer Loaders": {"space": 10, "cost": 2000000},
        "Compactors": {"space": 12, "cost": 2500000},
        "Tele Handlers": {"space": 18, "cost": 3500000}
    }
    avg_demand = df_input.groupby("Infrastructure_Machineries")['Predicted_Daily_Sales_Quantity'].mean()
    remaining_space = 5000
    inventory_plan = {}

    for machine, demand in avg_demand.items():
        if machine in storage_requirements:
            space_per_unit = storage_requirements[machine]['space']
            max_units = remaining_space // space_per_unit
            optimal_units = min(int(demand), max_units)
            inventory_plan[machine] = optimal_units
            remaining_space -= optimal_units * space_per_unit

    inventory_df = pd.DataFrame(list(inventory_plan.items()), columns=['Machinery', 'Optimal_Units'])

    # Display Results
    st.subheader("Predicted Demand")
    st.write(df_input[['Date', 'Infrastructure_Machineries', 'Predicted_Daily_Sales_Quantity']])
    
    st.subheader("Optimized Inventory Plan")
    st.write(inventory_df)

    # Download Buttons
    st.download_button("Download Processed Data", df_input.to_csv(index=False), "processed_data.csv", "text/csv")
    st.download_button("Download Inventory Plan", inventory_df.to_csv(index=False), "inventory_plan.csv", "text/csv")

st.write("Upload the cleaned dataset and input dataset to begin processing.")
