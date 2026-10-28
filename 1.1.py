import numpy as np
import matplotlib.pyplot as plt

prefix = 'jj1.1'

data = np.loadtxt(f"{prefix}.csv", skiprows = 1, delimiter = ',')

from scipy.optimize import curve_fit

def two_terms(x, a1, w1, f1, a2, w2, f2):
    return a1 * np.sin(w1 * x + f1) + a2 * np.sin(w2 * x + f2)


initial_guesses = [3.0, 314.0, 0.0, 3.0, 345.0, 0.0]
popt, pcov = curve_fit(two_terms, data[:,0], data[:,0], p0=initial_guesses)

# popt contains the optimized parameters [a, b, c]
print("Optimized parameters:", popt)

# 4. Generate points for a smooth fit line
x_fit = np.linspace(data[0,0], data[-1,0], 100)
y_fit = two_terms(x_fit, *popt)

plt.plot(data[:,0], data[:,1])
plt.plot(x_fit, y_fit)
plt.savefig(f"{prefix}.png")
