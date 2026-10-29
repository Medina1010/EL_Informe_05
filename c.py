import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def one_term(x, a1, f1, t1):
    return a1 * np.sin(2 * np.pi * f1 * x + t1)

def two_terms(x, a1, f1, t1, a2, f2, t2):
    return a1 * np.sin(2 * np.pi * f1 * x + t1) + a2 * np.sin(2 * np.pi * f2 * x + t2)

def three_terms(x, a1, f1, t1, a2, f2, t2, a3, f3, t3):
    return a1 * np.sin(2 * np.pi * f1 * x + t1) + a2 * np.sin(2 * np.pi * f2 * x + t2) + a3 * np.sin(2 * np.pi * f3 * x + t3)

def four_terms(x, a1, f1, t1, a2, f2, t2, a3, f3, t3, a4, f4, t4):
    return a1 * np.sin(2 * np.pi * f1 * x + t1) + a2 * np.sin(2 * np.pi * f2 * x + t2) + a3 * np.sin(2 * np.pi * f3 * x + t3) + a4 * np.sin(2 * np.pi * f4 * x + t4)


prefix = 'jjc'
data = []
for i in range(4):
    data.append(np.loadtxt(f"{prefix}{i+1}.csv", skiprows = 1, delimiter = ','))

initial_guesses = [1.5, 100, 0.0]
initial_guesses = curve_fit(one_term, data[0][:,0], data[0][:,1], p0=initial_guesses)[0]

y_fit = one_term(data[0][:,0], *initial_guesses)
plt.plot(data[0][:,0], data[0][:,1])
plt.plot(data[0][:,0], y_fit)
plt.xlabel('Time (s)')
plt.ylabel('CH1 (V)')
plt.savefig(f"{prefix}1.png")
plt.cla()

initial_guesses = np.append(initial_guesses, [1.5, 300, 0.0])
initial_guesses = curve_fit(two_terms, data[1][:,0], data[1][:,1], p0=initial_guesses)[0]

y_fit = two_terms(data[1][:,0], *initial_guesses)
plt.plot(data[1][:,0], data[1][:,1])
plt.plot(data[1][:,0], y_fit)
plt.xlabel('Time (s)')
plt.ylabel('CH1 (V)')
plt.savefig(f"{prefix}2.png")
plt.cla()

initial_guesses = np.append(initial_guesses, [1.5, 500, 0.0])
initial_guesses = curve_fit(three_terms, data[2][:,0], data[2][:,1], p0=initial_guesses)[0]

y_fit = three_terms(data[0][:,0], *initial_guesses)
plt.plot(data[2][:,0], data[2][:,1])
plt.plot(data[2][:,0], y_fit)
plt.xlabel('Time (s)')
plt.ylabel('CH1 (V)')
plt.savefig(f"{prefix}3.png")
plt.cla()

initial_guesses = np.append(initial_guesses, [1.5, 700, 0.0])
initial_guesses = curve_fit(four_terms, data[3][:,0], data[3][:,1], p0=initial_guesses, maxfev=10000)[0]

y_fit = four_terms(data[0][:,0], *initial_guesses)
plt.plot(data[3][:,0], data[3][:,1])
plt.plot(data[3][:,0], y_fit)
plt.xlabel('Time (s)')
plt.ylabel('CH1 (V)')
plt.savefig(f"{prefix}4.png")

print(initial_guesses)

y_err = 6e-3 * np.ones_like(data[3][:,0])
residuals = data[3][:,1] - y_fit
chisq = np.sum((residuals / y_err) ** 2)

print(f"chi^2={chisq}")
