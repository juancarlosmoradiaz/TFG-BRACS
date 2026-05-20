import matplotlib.pyplot as plt

# Datos H-Optimus1
alpha_500 = [0.05, 0.02, 0.01, 0.005]
elim_500 = [0.2, 24.6, 73.0, 97.8]

alpha_5000 = [0.02, 0.003, 0.002, 0.0015, 0.001, 0.0005, 0.0002]
elim_5000 = [0.0, 5.66, 23.46, 42.72, 73.18, 96.10, 99.78]

plt.figure(figsize=(8, 5))
plt.semilogx(alpha_500, elim_500, marker='o', linewidth=2, label='500 patches')
plt.semilogx(alpha_5000, elim_5000, marker='s', linewidth=2, label='5000 patches')

plt.gca().invert_xaxis()
plt.xlabel(r'$\alpha$')
plt.ylabel('Porcentaje eliminado (%)')
plt.title('H-Optimus1: sensibilidad del porcentaje eliminado frente a $\\alpha$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('imagenes/hoptimus1_alpha_sensitivity.png', dpi=300, bbox_inches='tight')
plt.close()