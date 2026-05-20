import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# -------------------------
# Panel izquierdo: F1 macro medio
# -------------------------
labels_f1 = [
    'H-Optimus1\nbaseline',
    'H-Optimus1\nclean 0.00025',
    'Virchow2\nbaseline',
    'Virchow2\nclean 0.0002',
    'Virchow2\nclean 0.00025'
]
values_f1 = [0.6118, 0.6012, 0.6011, 0.5908, 0.5895]

axes[0].bar(labels_f1, values_f1)
axes[0].set_ylabel('Val F1 macro medio')
axes[0].set_title('Comparación final de rendimiento')
axes[0].tick_params(axis='x', rotation=20)
axes[0].grid(True, axis='y', alpha=0.3)

# -------------------------
# Panel derecho: tiempos
# -------------------------
labels_time = [
    'H-Optimus1\nbaseline',
    'H-Optimus1\nlimpieza',
    'H-Optimus1\nretrain',
    'Virchow2\nbaseline',
    'Virchow2\nlimpieza',
    'Virchow2\nretrain'
]
values_time = [
    4000,      # H-Optimus1 baseline aprox.
    129600,    # 36 h
    9.6,       # retrain
    32300,     # Virchow2 baseline aprox. (~9 h)
    131400,    # 36.5 h
    10.6       # retrain alpha 0.0002, el más representativo
]

axes[1].bar(labels_time, values_time)
axes[1].set_ylabel('Tiempo (s)')
axes[1].set_title('Comparación de coste computacional')
axes[1].tick_params(axis='x', rotation=20)
axes[1].set_yscale('log')
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('imagenes/cleaning_summary.png', dpi=300, bbox_inches='tight')
plt.close()