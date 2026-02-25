import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(page_title="Demand Forecasting & Inventory Optimization", layout="wide")
st.title("📊 Demand Forecasting & Inventory Optimization")

st.caption("### Upload Datasets")
column1, column2 = st.columns(2)
cleaned_dataset = column1.file_uploader("Upload Cleaned Dataset", type=["csv"])
input_dataset = column2.file_uploader("Upload Input Dataset", type=["csv"])

if cleaned_dataset and input_dataset:
    df = pd.read_csv(cleaned_dataset)
    df_input = pd.read_csv(input_dataset)
    
    st.success("Files uploaded successfully!")
    
    
    encoded_df = pd.get_dummies(df, columns=['Infrastructure_Machineries', 'Region'], drop_first=True)
    X = encoded_df.drop(columns=['Date', 'Daily_Sales_Quantity', 'Customer_Id'])
    y = encoded_df['Daily_Sales_Quantity']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    
    df_input.rename(columns={'Daily_Sales _Percentage': 'Daily_Sales_Percentage'}, inplace=True)
    df_input['Date'] = pd.to_datetime(df_input['Date'], infer_datetime_format=True, dayfirst=False, errors='coerce')

    encoded_input = pd.get_dummies(df_input, columns=['Infrastructure_Machineries', 'Region'], drop_first=True)
    for col in set(X.columns) - set(encoded_input.columns):
        encoded_input[col] = 0
    encoded_input = encoded_input[X.columns]
    
    
    sales_predict= rf_model.predict(encoded_input)
    df_input['Predicted_Daily_Sales_Quantity'] = sales_predict
    
    
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
    
    
    st.markdown("### 📈 Predicted Demand")
    st.dataframe(df_input[['Date', 'Infrastructure_Machineries', 'Predicted_Daily_Sales_Quantity']])
    
    st.markdown("### 📦 Optimized Inventory Plan")
    st.dataframe(inventory_df)
    
    
    st.markdown("### 📂 Download Results")
    column1, column2 = st.columns(2)
    column1.download_button("Download Processed Data", df_input.to_csv(index=False), "processed_data.csv", "text/csv")
    column2.download_button("Download Inventory Plan", inventory_df.to_csv(index=False), "inventory_plan.csv", "text/csv")

else:
    st.info("Upload both the cleaned dataset and input dataset to begin processing.")