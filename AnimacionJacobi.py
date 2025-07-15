import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Definición del sistema
def gauss_jacobi_iteration(x, b):
    """
    Iteración del método de Gauss-Jacobi para el sistema dado.
    """
    x1, x2 = x
    x1_new = 6 - x2
    x2_new = 2 * x1
    return np.array([x1_new, x2_new])

# Parámetros
max_iterations = 50
tolerance = 1e-6

# Vector b
b = np.array([6, 0])

# Posiciones iniciales
initial_guesses = [
    np.array([3, 3]),  # Posición inicial 1
    np.array([2, 4]),  # Posición inicial 2
    np.array([5, 2])   # Posición inicial 3
]

# Almacenar trayectorias
trajectories = []
for initial_guess in initial_guesses:
    x = initial_guess.copy()
    trajectory = [x.copy()]
    
    for _ in range(max_iterations):
        x = gauss_jacobi_iteration(x, b)
        trajectory.append(x.copy())
    
    trajectories.append(np.array(trajectory))

# Configurar la figura
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['blue', 'green', 'red']
lines = []
points = []

# Inicializar líneas y puntos
for i in range(len(trajectories)):
    line, = ax.plot([], [], marker='o', color=colors[i], label=f'Initial Guess {i+1}')
    point, = ax.plot([], [], 'o', color=colors[i], markersize=10, alpha=0.7)
    lines.append(line)
    points.append(point)

# Punto exacto de solución
solution = np.linalg.solve(np.array([[1, 1], [-2, 1]]), b)
ax.scatter(solution[0], solution[1], color='black', s=100, label='Exact Solution')

# Configuraciones del gráfico
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Trayectorias del Método de Gauss-Jacobi')
ax.legend()
ax.grid(True)

# Función de inicialización
def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        point.set_data([], [])
    return lines + points

# Función de animación
def update(frame):
    for i, traj in enumerate(trajectories):
        # Mostrar hasta el frame actual para cada trayectoria
        if frame < len(traj):
            x_data = traj[:frame+1, 0]
            y_data = traj[:frame+1, 1]
            lines[i].set_data(x_data, y_data)
            points[i].set_data(traj[frame, 0], traj[frame, 1])
    
    # Ajustar dinámicamente los límites si es necesario
    all_points = np.concatenate([t[:min(frame+1, len(t))] for t in trajectories])
    if len(all_points) > 0:
        margin = 0.5
        x_min, x_max = all_points[:, 0].min()-margin, all_points[:, 0].max()+margin
        y_min, y_max = all_points[:, 1].min()-margin, all_points[:, 1].max()+margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    return lines + points

# Crear la animación
max_frames = max(len(traj) for traj in trajectories)
ani = FuncAnimation(fig, update, frames=max_frames,
                    init_func=init, blit=True, interval=500, repeat_delay=2000)

plt.close()
HTML(ani.to_jshtml())

ani.save('gauss_jacobi.gif', writer='pillow', fps=2)  # Para GIF
# o
ani.save('gauss_jacobi.mp4', writer='ffmpeg', fps=2)  # Para MP4