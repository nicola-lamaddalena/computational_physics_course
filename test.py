import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm

# Parametri
num_esperimenti = 60000
dim_campione = 100
batch_size = 1000

# Generiamo tutte le medie in anticipo
f_mean = np.zeros(num_esperimenti)
for i in range(num_esperimenti):
    campione = np.random.uniform(0, 1, dim_campione)
    f_mean[i] = np.mean(campione)

# Configurazione del plot
fig, ax = plt.subplots(figsize=(10, 6))

# Parametri teorici
mu = 0.5
sigma = np.sqrt(1/12) / np.sqrt(dim_campione)
x_theo = np.linspace(0.2, 0.8, 100)
y_theo = norm.pdf(x_theo, mu, sigma)

# Linea teorica (statica)
ax.plot(x_theo, y_theo, 'r-', linewidth=2, label="Gaussiana teorica")

# Testo del contatore
line_text = ax.text(0.98, 0.97, '', transform=ax.transAxes, 
                     ha='right', va='top', fontsize=12, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def update(frame):
    # Rimuovi l'istogramma precedente
    for patch in ax.patches:
        patch.remove()
    
    # Numero di campioni da mostrare
    n_samples = min((frame + 1) * batch_size, num_esperimenti)
    data = f_mean[:n_samples]
    
    # Crea il nuovo istogramma
    ax.hist(data, bins=50, density=True, 
            alpha=0.7, label="Distribuzioni delle medie", 
            color='skyblue', edgecolor='black')
    
    # Aggiorna il testo
    line_text.set_text(f'Campioni: {n_samples}/{num_esperimenti}')
    
    ax.set_xlabel("Media campionaria")
    ax.set_ylabel("Densità di probabilità")
    ax.set_title("Teorema del Limite Centrale (Animato)")
    ax.set_ylim(0, max(y_theo) * 1.2)
    ax.set_xlim(0.2, 0.8)
    ax.legend(loc='upper left')

# Animazione
num_frames = (num_esperimenti // batch_size) + 1
anim = FuncAnimation(fig, update, frames=num_frames, 
                     interval=50, repeat=True)

plt.tight_layout()
plt.show()
