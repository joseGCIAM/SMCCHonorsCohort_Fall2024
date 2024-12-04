# Import necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Probelm statement:In a controlled environment,a population of animals is introduced with a limited amount of resources. 
# The population is expected to grow rapidly at first but slow down as it nears the environment's carrying capacity due to resource constraints. 
# Using the logistic growth model, predict how the population changes over time, taking into account the initial population, growth rate, and environmental carrying capacity.

# Define the logistic growth model as a differential equation
def logistic_growth(t, P, r, K):
    """
    Logistic growth model
    dP/dt = r * P * (1 - P / K)
    t: time
    P: population
    r: growth rate
    K: carrying capacity
    """
    return r * P * (1 - P / K)

# Parameters for the model
r = 0.3  # Growth rate
K = 100  # Carrying capacity
P0 = 10  # Initial population
t_span = (0, 50)  # Time interval (0 to 50)
t_eval = np.linspace(0, 50, 100)  # Points where solution is evaluated

# Solve the differential equation
solution = solve_ivp(
    logistic_growth, 
    t_span, 
    [P0], 
    args=(r, K), 
    t_eval=t_eval
)

# Extract the results
time = solution.t
population = solution.y[0]

# Plot the results
plt.plot(time, population, label="Population Growth")
plt.axhline(y=K, color="r", linestyle="--", label="Carrying Capacity (K)")
plt.title("Logistic Growth Model")
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.show()
