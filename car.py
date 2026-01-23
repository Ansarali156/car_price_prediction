# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
data = pd.read_csv('car_dataset.csv')

# Step 2: Data exploration and preprocessing
# Drop any rows with missing values
data.dropna(inplace=True)

# Convert categorical variables to numerical using Label Encoding (e.g., Brand, Fuel Type, Transmission)
label_encoder = LabelEncoder()
data['name'] = label_encoder.fit_transform(data['name'])
data['fuel'] = label_encoder.fit_transform(data['fuel'])
data['transmission'] = label_encoder.fit_transform(data['transmission'])

# Step 3: Define features (X) and target (y)
X = data[['name', 'year', 'fuel', 'transmission']]
y = data['selling_price']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Step 8: Visualize the actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()
