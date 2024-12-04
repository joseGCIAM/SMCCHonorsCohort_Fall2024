# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Problem statement: A small town has been experiencing steady population growth over the past decade. 
# Town officials need to predict future population sizes to allocate resources effectively and plan for infrastructure development. 
# Using historical data of the town's population over the last ten years, create a machine learning model that predicts population growth 
# trends. The solution should provide insights into future growth to help the town plan housing, schools, and other essential services.

# Step 1: Create a simple dataset
# Years (independent variable)
years = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
# Population (dependent variable)
population = np.array([2.3, 2.5, 2.8, 3.0, 3.3, 3.7, 4.0, 4.3, 4.6, 5.0])

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(years, population, test_size=0.2, random_state=42)

# Step 3: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Step 6: Visualize the results
plt.scatter(years, population, color='blue', label='Actual Data')
plt.plot(years, model.predict(years), color='red', label='Regression Line')
plt.xlabel('Years')
plt.ylabel('Population (millions)')
plt.title('Population Growth Prediction')
plt.legend()
plt.show()
