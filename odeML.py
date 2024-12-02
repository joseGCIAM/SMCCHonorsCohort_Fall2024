# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

# Step 1: Create synthetic data for temperature vs growth rate
# Temperatures (independent variable)
temperatures = np.array([20, 22, 24, 26, 28, 30, 32, 34, 36, 38]).reshape(-1, 1)
# Growth rates (dependent variable)
growth_rates = np.array([0.15, 0.17, 0.20, 0.23, 0.27, 0.30, 0.34, 0.39, 0.44, 0.50])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(temperatures, growth_rates, test_size=0.2, random_state=42)

# Step 3: Train a Linear Regression model to predict growth rates
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set and calculate the Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error for growth rate predictions: {mse:.4f}")

# Predict the growth rate at a new temperature (e.g., 33°C)
new_temperature = np.array([[33]])
predicted_growth_rate = model.predict(new_temperature)[0]
print(f"Predicted growth rate at {new_temperature[0][0]}°C: {predicted_growth_rate:.4f}")

# Step 4: Define the logistic growth ODE using the predicted growth rate
def logistic_growth(t, P, r, K):
    """
    Logistic growth model:
    dP/dt = r * P * (1 - P / K)
    t: time
    P: population
    r: growth rate
    K: carrying capacity
    """
    return r * P * (1 - P / K)

# Parameters for the ODE
K = 500  # Carrying capacity
P0 = 50  # Initial population
t_span = (0, 50)  # Time range
t_eval = np.linspace(0, 50, 200)  # Time points to evaluate

# Solve the ODE using the predicted growth rate
solution = solve_ivp(
    logistic_growth, 
    t_span, 
    [P0], 
    args=(predicted_growth_rate, K), 
    t_eval=t_eval
)

# Extract results
time = solution.t
population = solution.y[0]

# Step 5: Visualize the results
plt.figure(figsize=(10, 6))

# Plot the machine learning regression line
plt.subplot(2, 1, 1)
plt.scatter(temperatures, growth_rates, color="blue", label="Training Data")
plt.plot(temperatures, model.predict(temperatures), color="red", label="Regression Line")
plt.scatter(new_temperature, predicted_growth_rate, color="green", label="Prediction")
plt.xlabel("Temperature (°C)")
plt.ylabel("Growth Rate")
plt.title("Temperature vs Growth Rate (Linear Regression)")
plt.legend()

# Plot the ODE solution
plt.subplot(2, 1, 2)
plt.plot(time, population, color="purple", label="Population Growth")
plt.axhline(y=K, color="red", linestyle="--", label="Carrying Capacity")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population Growth Prediction using Logistic Model")
plt.legend()

plt.tight_layout()
plt.show()
