import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

L = 1
T = 0.1
N = 200
N_t = 10000

h = L / N
k = T / N_t

x = np.linspace(0, L, N)
t = np.linspace(0, T, N_t)

z = np.zeros((N, N_t))

def funk(x):
    return np.sin(np.pi * x)

funk_val = funk(x)
z[:, 0] = funk_val

def neste(z):
    for i in range(N_t - 1):
        if k / (h**2) >= 0.5:  
            print("krav om at k/(h^2) <= 0.5 er ikke oppfylt.")
            break
        for j in range(1, N - 1):
            z[j, i + 1] = k / (h**2) * (z[j + 1, i] - 2 * z[j, i] + z[j - 1, i]) + z[j, i]

neste(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, T = np.meshgrid(x, t)
surf = ax.plot_surface(X, T, z.T, cmap=cm.seismic)  
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('posisjon')
ax.set_ylabel('tid')
ax.set_zlabel('temperatur')

plt.title('Varmeligningen eksplisitt')

plt.show()
