import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
data = pd.read_csv(r"C:\Users\alphi\Downloads\CarPrice\car data.csv")

# FEATURE ENGINEERING
data['Brand'] = data['Car_Name'].apply(lambda x: x.split()[0])

data['Car_Age'] = 2025 - data['Year']

data.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Simulate real-world features 
data['Mileage'] = np.random.normal(18, 4, len(data))         
data['Horsepower'] = np.random.randint(70, 150, len(data))   
data['Torque'] = np.random.randint(100, 300, len(data))      

# ENCODE CATEGORICAL VARIABLES
le = LabelEncoder()
for col in ['Fuel_Type', 'Selling_type', 'Transmission', 'Brand']:
    data[col] = le.fit_transform(data[col])

# FEATURES & TARGET
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL TRAINING
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# PREDICTION
y_pred = model.predict(X_test)

# EVALUATION
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# VISUALIZATION 1: Actual vs Predicted (with corrected legend)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', lw=2, label="Ideal Fit")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# VISUALIZATION 2: Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Predicting Car Price")
plt.grid(True)
plt.tight_layout()
plt.show()
