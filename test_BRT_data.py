import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the .mat file
mat_data = scipy.io.loadmat('BRT_data.mat')

# Extract grid info
g = mat_data['g']
g_min = g['min'][0][0].flatten()
g_max = g['max'][0][0].flatten()
g_N = g['N'][0][0].flatten()
g_dx = g['dx'][0][0].flatten()

# Extract grid axes
xs = np.linspace(g_min[0], g_max[0], g_N[0])
ys = np.linspace(g_min[1], g_max[1], g_N[1])
thetas = np.linspace(g_min[2], g_max[2], g_N[2])

# Extract v and w from safety controller
safety_ctrl = mat_data['safety_controller'][0, 0]
v = safety_ctrl['v']
w = safety_ctrl['w']

# Generate meshgrid for x-y
X, Y = np.meshgrid(xs, ys)

# Plot v(x, y) for several theta slices
theta_indices = [0, g_N[2] // 4, g_N[2] // 2, 3 * g_N[2] // 4, g_N[2] - 1]
theta_values = thetas[theta_indices]

fig = plt.figure(figsize=(18, 10))
for i, idx in enumerate(theta_indices):
    ax = fig.add_subplot(2, 3, i + 1, projection='3d')
    Z = v[:, :, idx]  # shape [Nx, Ny]
    
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f'v(x, y) at θ = {theta_values[i]:.2f} rad')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('v')
    ax.view_init(elev=30, azim=135)  # nice viewing angle

plt.suptitle('Velocity Field v(x, y) across Different θ slices', fontsize=16)
plt.tight_layout()
plt.show()
