import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

prefix = 'jj1.1'

data = np.loadtxt(f"data/{prefix}.csv", skiprows = 1, delimiter = ',')

def two_terms(x, a1, f1, t1, a2, f2, t2, V0):
    return a1 * np.sin(2 * np.pi * f1 * x + t1) + a2 * np.sin(2 * np.pi * f2 * x + t2) + V0


initial_guesses = [1.5, 100, 0.0, 1.5, 105, 0.0, 0.0]
popt, pcov = curve_fit(two_terms, data[:,0], data[:,1], p0=initial_guesses)

print("para frecuencias 1:1.05")
print(f"{popt[0]} sin (2 * pi * {popt[1]} * t + {popt[2]}) + {popt[3]} sin (2 * pi * {popt[4]} * t + {popt[5]}) + {popt[6]}")

y_fit = two_terms(data[:,0], *popt)

y_err = 6e-3 * np.ones_like(data[:,0])
residuals = data[:,1] - y_fit
chisq = np.sum((residuals / y_err) ** 2)

print(f"chi^2={chisq}")

plt.figure(figsize=(10,5), dpi=200)
plt.scatter(data[:,0], data[:,1], label="Datos Experimentales", s=1.5, color="blue")
plt.plot(data[:,0], y_fit, label=f"Curva Ajustada\n$ω_1≈${popt[1]:.3f}, $ω_2≈${popt[4]:.3f}", lw=0.8, color="red")
plt.legend()
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.tight_layout()
plt.savefig(f"res/{prefix}.png")
