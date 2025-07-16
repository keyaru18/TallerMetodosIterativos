import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from typing import List

def gauss_jacobi(*, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
    n = A.shape[0]
    x = x0.copy()
    tray = [x.copy()]
    for k in range(1, max_iter):
        x_new = np.zeros((n, 1))
        for i in range(n):
            suma = sum([A[i, j] * x[j] for j in range(n) if j != i])
            x_new[i] = (b[i] - suma) / A[i, i]
        tray.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, tray

def gauss_seidel(*, A: np.array, b: np.array, x0: np.array, tol: float, max_iter: int):
    n = A.shape[0]
    x = x0.copy()
    tray = [x.copy()]
    for k in range(1, max_iter):
        for i in range(n):
            suma = sum([A[i, j] * x[j] for j in range(n) if j != i])
            x[i] = (b[i] - suma) / A[i, i]
        tray.append(x.copy())
        if np.linalg.norm(A @ x - b) < tol:
            break
    return x, tray

def animate_trajectory(trays: List[List[np.array]], labels: List[str], title: str):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Preparar límites de la gráfica
    all_points = [pt for tray in trays for pt in tray]
    x_vals = [p[0, 0] for p in all_points]
    y_vals = [p[1, 0] for p in all_points]
    ax.set_xlim(min(x_vals) - 1, max(x_vals) + 1)
    ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)

    # Rectas del sistema
    line_x = np.linspace(min(x_vals)-1, max(x_vals)+1, 100)
    ax.plot(line_x, 7 - line_x, 'k--', alpha=0.7, label="x₁ + x₂ = 7")
    ax.plot(line_x, (2 * line_x) / 5, 'k-.', alpha=0.7, label="-2x₁ + 5x₂ = 0")

    # Solución exacta
    solution = np.linalg.solve(A, b)
    ax.plot(solution[0], solution[1], 'g*', markersize=15, label='Solución exacta')

    # Inicializar líneas animadas
    lines = [ax.plot([], [], marker='o', markersize=8, 
                    markeredgewidth=1.5, markeredgecolor='k',
                    label=label, linewidth=2)[0] for label in labels]
    
    # Líneas de trayectoria
    paths = [ax.plot([], [], '--', alpha=0.5, linewidth=1)[0] for _ in labels]
    
    # Texto para mostrar información
    info_texts = [ax.text(0.02, 0.95 - i*0.05, '', transform=ax.transAxes, 
                         fontsize=10, bbox=dict(facecolor='white', alpha=0.7)) 
                 for i in range(len(labels))]
    
    # Almacenar coordenadas acumuladas
    trajectory_data = [[] for _ in labels]

    def init():
        for line, path in zip(lines, paths):
            line.set_data([], [])
            path.set_data([], [])
        for text in info_texts:
            text.set_text('')
        return lines + paths + info_texts

    def update(frame):
        for i, tray in enumerate(trays):
            if frame < len(tray):
                pt = tray[frame]
                trajectory_data[i].append(pt.flatten())
                
                # Actualizar punto actual
                lines[i].set_data(pt[0], pt[1])
                
                # Actualizar trayectoria
                if len(trajectory_data[i]) > 1:
                    xs, ys = zip(*trajectory_data[i])
                    paths[i].set_data(xs, ys)
                
                # Actualizar información
                info_texts[i].set_text(
                    f"{labels[i]}: Iter {frame}\n"
                    f"x₁ = {pt[0,0]:.4f}, x₂ = {pt[1,0]:.4f}\n"
                    #f"Error: {np.linalg.norm(A @ pt - b):.2e}"
                )
        
        return lines + paths + info_texts

    ani = FuncAnimation(fig, update, frames=max(len(t) for t in trays), 
                        init_func=init, interval=800, blit=False, repeat=False)
    
    # Añadir leyenda mejorada
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=10)
    
    # Ajustar diseño
    plt.tight_layout()
    plt.close(fig)
    
    return HTML(ani.to_jshtml())

# Sistema de ecuaciones
A = np.array([[1, 1], [-2, 5]], dtype=float)
b = np.array([[7], [0]], dtype=float)

# Puntos iniciales
initial_points = [
    np.array([[0], [0]], dtype=float),
    np.array([[5], [2]], dtype=float),
    np.array([[100], [-50]], dtype=float)
]

# Parámetros
tol = 1e-6
max_iter = 50

# Ejecutar para cada punto inicial
for x0 in initial_points:
    print(f"\n--- Animando desde punto inicial: {x0.T} ---")
    x_j, tray_j = gauss_jacobi(A=A, b=b, x0=x0, tol=tol, max_iter=max_iter)
    x_gs, tray_gs = gauss_seidel(A=A, b=b, x0=x0, tol=tol, max_iter=max_iter)
    display(animate_trajectory([tray_j, tray_gs], ["Jacobi", "Gauss-Seidel"], 
                             f"Trayectoria desde x0 = [{x0[0,0]}, {x0[1,0]}]"))