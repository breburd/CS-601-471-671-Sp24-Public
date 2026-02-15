from matplotlib import pyplot as plt
from typing import List
import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def plot_function(x, func_values: List[float], deriv_values: List[float], title: str, save_fig_path: str):
    plt.clf()
    plt.title(title)
    plt.plot(x, func_values, label='function')
    plt.plot(x, deriv_values, label='derivative')
    plt.axhline(0, color='black',linewidth=0.5) # Add x-axis line
    plt.axvline(0, color='black',linewidth=0.5) # Add y-axis line
    plt.xlim(x.min(), x.max())
    plt.legend()
    plt.savefig(save_fig_path)


# 2. Generate x values (range and number of points)
x = np.linspace(-15, 15, 500)

# 3. Calculate function values
y = sigmoid(x)

# 4. Calculate the derivative numerically using np.gradient()
# The second argument 'x' provides the spacing for the derivative calculation
dfdx = np.gradient(y, x)

plot_function(x, y, dfdx, "Sigmoid = 1/(1 + e^(-x))", "sigmoid.png")

# 3. Calculate function values
y = np.maximum(0, x)

# 4. Calculate the derivative numerically using np.gradient()
# The second argument 'x' provides the spacing for the derivative calculation
dfdx = np.gradient(y, x)

plot_function(x, y, dfdx, "ReLU = max(0, x)", "relu.png")

# 3. Calculate function values
y = np.tanh(x)

# 4. Calculate the derivative numerically using np.gradient()
# The second argument 'x' provides the spacing for the derivative calculation
dfdx = np.gradient(y, x)

plot_function(x, y, dfdx, "Tanh = tanh(x)", "tanh.png")
