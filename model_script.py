import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load the cleaned dataset
file_path = "Cleaned_Datathon_Dataset.csv"  # Ensure the cleaned dataset is in the same folder
df = pd.read_csv(file_path)

# Encode categorical features (Infrastructure_Machineries, Region)
df_encoded = pd.get_dummies(df, columns=['Infrastructure_Machineries', 'Region'], drop_first=True)

# Define features (X) and target (y - Daily Sales Quantity)
X = df_encoded.drop(columns=['Date', 'Daily_Sales_Quantity', 'Customer_Id'])
y = df_encoded['Daily_Sales_Quantity']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model for demand forecasting
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Machine storage requirements and costs
machine_specs = {
    "Backhoe Loader": {"space": 15, "cost": 3000000},
    "Excavators(crawler)": {"space": 25, "cost": 5000000},
    "Loaders (Wheeled)": {"space": 20, "cost": 4000000},
    "Skid Steer Loaders": {"space": 10, "cost": 2000000},
    "Compactors": {"space": 12, "cost": 2500000},
    "Tele Handlers": {"space": 18, "cost": 3500000}
}

# Get average predicted demand per machinery type
df['Predicted_Demand'] = rf_model.predict(X)
avg_demand = df.groupby("Infrastructure_Machineries")['Predicted_Demand'].mean()

# Inventory optimization using a greedy approach (maximize demand fulfillment within space limit)
total_space = 5000  # Available warehouse space
inventory_plan = {}

for machine, demand in avg_demand.items():
    if machine in machine_specs:
        space_per_unit = machine_specs[machine]['space']
        max_units = total_space // space_per_unit  # Max units we can store within space constraint
        optimal_units = min(int(demand), max_units)  # Store as much as predicted demand allows
        inventory_plan[machine] = optimal_units
        total_space -= optimal_units * space_per_unit  # Reduce available space

# Print Inventory Plan
print("\nOptimized Inventory Plan:")
for machine, units in inventory_plan.items():
    print(f"{machine}: {units} units")
