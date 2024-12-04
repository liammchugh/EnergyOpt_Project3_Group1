import numpy as np


import matplotlib.pyplot as plt

def calculate_CVaR(alpha, scenario_costs, beta=0.8):
    scenario_costs_expr = np.vstack(scenario_costs)
    M = scenario_costs_expr - alpha
    postM = np.maximum(M, 0)
    CVaR_loss = alpha + (1/(len(scenario_costs)*(1-beta))) * np.sum(postM)
    return CVaR_loss, M

# Generate random scenario costs between 50 and 100
np.random.seed(42)  # For reproducibility
scenario_costs = np.random.uniform(500, 10000, 10)

# Define a range of alpha values
alpha_values = np.linspace(min(scenario_costs), max(scenario_costs), 100)
CVaR_values = []
M_values = []

# Calculate CVaR and M for each alpha
for alpha in alpha_values:
    CVaR_loss, M = calculate_CVaR(alpha, scenario_costs)
    CVaR_values.append(CVaR_loss)
    M_values.append(M)

# Plot CVaR as a function of alpha
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, CVaR_values, label='CVaR')
plt.xlabel('Alpha')
plt.ylabel('CVaR')
plt.title('CVaR as a function of Alpha')
plt.legend()
plt.grid(True)
plt.show()

# Plot M as a function of alpha
plt.figure(figsize=(10, 5))
for i in range(len(scenario_costs)):
    plt.plot(alpha_values, [M[i] for M in M_values], label=f'Scenario {i+1}')
plt.xlabel('Alpha')
plt.ylabel('M')
plt.title('M as a function of Alpha')
plt.legend()
plt.grid(True)
plt.show()