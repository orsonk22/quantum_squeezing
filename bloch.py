import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def bloch_vector(t, omega):
    x = np.cos(omega * t)
    y = -np.sin(omega * t)
    z = 0
    return np.array([x, y, z])

def plot_bloch_sphere():
    # Create a 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='cyan', alpha=0.1)

    # Axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1.5)

    # Label axes
    ax.text(1.1, 0, 0, 'X', color='red')
    ax.text(0, 1.1, 0, 'Y', color='green')
    ax.text(0, 0, 1.1, 'Z', color='blue')

    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect('auto')
    ax.view_init(30, 30)
    return ax

def plot_trajectory(omega, t_max, num_points):
    ax = plot_bloch_sphere()

    # Time evolution
    t_values = np.linspace(0, t_max, num_points)
    trajectory = np.array([bloch_vector(t, omega) for t in t_values])

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='blue')
    ax.set_title('Time Evolution on Bloch Sphere')
    plt.show()

# Parameters
omega = 1.0  # Frequency
t_max = 2 * np.pi  # Time period for one full rotation
num_points = 300

# Plot the trajectory
plot_trajectory(omega, t_max, num_points)
