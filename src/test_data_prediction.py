import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the trained model
fpath = "C:\\House_Price_Prediction\\src\\models\\Random_Forest.joblib"
model = joblib.load(fpath)

# Load test dataset
ffpath = "C:\\House_Price_Prediction\\data\\test.csv"
test_data = pd.read_csv(ffpath)

# Prepare features and target
X_test = test_data.drop("Price", axis=1)
y_test = test_data['Price'].values 

# Make predictions
y_pred = model.predict(X_test)

# Calculate regression metrics (not accuracy)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Regression Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Get a random sample from test data with prediction
sample_index = test_data.sample(1).index[0]
sample_features = X_test.iloc[sample_index]
actual_price = y_test[sample_index]
predicted_price = y_pred[sample_index]

print("\nRandom Test Sample Prediction:")
print(f"Sample Features:\n{sample_features}")
print(f"Actual Price: {actual_price:.2f}")
print(f"Predicted Price: {predicted_price:.2f}")
print(f"Difference: {abs(actual_price - predicted_price):.2f}")