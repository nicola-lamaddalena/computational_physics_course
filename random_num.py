import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import norm

N = int(sys.argv[1])
campione = int(sys.argv[2])
f_mean = np.zeros(N)

for i in range(N):
    rng = np.random.uniform(0, 1, campione)
    f_mean[i] = np.mean(rng)

fig, ax1 = plt.subplots()
mu = 0.5
sigma = np.sqrt(1/12) / np.sqrt(campione)  # Deviazione standard teorica
x = np.linspace(0.3, 0.7, 100)
ax1.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label="Gaussiana teorica")
ax1.hist(f_mean, bins=50, density=True, alpha=0.7, label="Distribuzioni delle medie")
ax1.legend()
plt.show()
