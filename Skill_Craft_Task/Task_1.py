# 📌 House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv(r"C:\Users\OMKAR A TEMKAR\Downloads\house_price_regression_dataset.csv")
print("Dataset Preview:\n", df.head(), "\n")
print("Dataset Info:\n")
print(df.info(), "\n")
print("Summary Statistics:\n", df.describe(), "\n")

# 2. Correlation heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Scatter plots for relationships
sns.pairplot(df[['price', 'sqft', 'bedrooms', 'bathrooms']])
plt.show()

# 4. Define features (X) and target (y)
X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

# 5. Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Model evaluation
print("📊 Linear Regression Results:")
print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 9. Plot actual vs predicted prices
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", edgecolors="k")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 10. Example prediction
example = pd.DataFrame([[2000, 3, 2]], columns=['sqft', 'bedrooms', 'bathrooms'])
predicted_price = model.predict(example)[0]
print(f"Predicted price for 2000 sqft, 3 bed, 2 bath = {predicted_price:.2f}")
