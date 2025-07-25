import numpy as np
from sklearn.cluster import KMeans

ruta = "puntos_generados/puntos_tiff_eyeMasks.txt"
salida_grupo1 = "ojo_izquierdo.txt"
salida_grupo2 = "ojo_derecho.txt"

# Leer puntos desde el TXT
all_points = np.loadtxt(ruta)  # Cada línea: x y z
print(f"[OK] Archivo cargado: {ruta}")
print(f"[INFO] Total puntos detectados: {all_points.shape[0]}")

# Aplicar KMeans con 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
labels = kmeans.fit_predict(all_points)

# Guardar cada cluster en un archivo
with open(salida_grupo1, "w") as f1, open(salida_grupo2, "w") as f2:
    for (x, y, z), label in zip(all_points, labels):
        if label == 0:
            f1.write(f"{x} {y} {z}\n")
        else:
            f2.write(f"{x} {y} {z}\n")

print(f"[OK] Grupo 1: {np.sum(labels==0)} puntos -> {salida_grupo1}")
print(f"[OK] Grupo 2: {np.sum(labels==1)} puntos -> {salida_grupo2}")
