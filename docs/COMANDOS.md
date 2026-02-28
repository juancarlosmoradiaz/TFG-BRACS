# Guía rápida de trabajo – TFG BRACS

## 1. Conectarme al servidor

En el **Mac** (terminal local):

```bash
ssh bracs-ugr
```


## 2. Me voy al proyecto y activo el entorno virtual
```bash
cd ~/projects/tfg-bracs
source .venv/bin/activate
```

## 3. Abro el proyecto en VS Code y edito, entreno,...
Connect to Host... → seleccionar bracs-ugr.

## 4. MLFlow: Ver experimentos y resultados para trazabilidad
En el servidor, con el entorno activado:

```bash
cd ~/projects/tfg-bracs
source .venv/bin/activate

mlflow ui --backend-store-uri ./outputs/runs/mlruns --host 0.0.0.0 --port 5000
```

Mientras esta permanece abierta, abrimos otra terminal en nuestro local:

```bash
ssh -L 5000:localhost:5000 bracs-ugr
```

usuario: usuario_tfg
contraseña: PCI1-2..TFG2026

IP: 150.214.203.144
Puerto: 2022

Mientras esta sesion este abierta, el puerto 5000 del servidor quedará disponible en el PC como localhost:5000.


Para abrir MLFlow en el ordenador, introducimos:
```bash
http://localhost:5000
```

y ahi veremos:
	•	El experimento tfg-bracs.
	•	Cada ejecución (run) con:
	•	parámetros,
	•	métricas,
	•	artefactos (modelos, gráficos, etc.).


