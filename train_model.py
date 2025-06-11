

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("Car details v3.csv")
df.dropna(subset=['engine'], inplace=True)


df['engine'] = df['engine'].str.replace(' CC', '', regex=False).astype(float)

# Define features and target
features = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'engine']
target = 'selling_price'
X = df[features]
y = df[target]

# Encode categorical features(alphabet value(object))
encoders = {}
for col in ['name', 'fuel', 'seller_type', 'transmission', 'owner']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train model in which We Use RandoForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)


joblib.dump(model, 'car_price_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
df[['name', 'engine']].drop_duplicates().to_csv('car_engine_data.csv', index=False)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


y_pred = model.predict(X_test)
#result :
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print(f" Model Evaluation Metrics:")
print(f" MAE (Mean Absolute Error): {mae:.2f}")
print(f" RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f" R² Score (Coefficient of Determination): {r2:.4f}")


print(" Model and encoders with engine saved successfully!")


import matplotlib.pyplot as plt
import numpy as np


n = 20
actual = y_test[:n].values
predicted = y_pred[:n]


indices = np.arange(n)

bar_width = 0.35

plt.figure(figsize=(14, 6))


plt.bar(indices, actual, width=bar_width, color='blue', label='Actual Price')


plt.bar(indices + bar_width, predicted, width=bar_width, color='red', label='Predicted Price')


plt.xlabel('Car Sample')
plt.ylabel('Selling Price (₹)')
plt.title('Actual vs Predicted Car Prices')
plt.xticks(indices + bar_width / 2, [f'Car {i+1}' for i in range(n)], rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

