import os
import numpy as np
from sklearn.cluster import KMeans

# Carpetas
entrada = "puntos_generados"
salida = "puntos_separados_muscle"
os.makedirs(salida, exist_ok=True)

# Archivos a procesar
archivos = [
    "puntos_tiff_bloodMasks.txt", "puntos_tiff_brainMasks.txt",
    "puntos_tiff_duodenumMasks.txt", "puntos_tiff_eyeMasks.txt", "puntos_tiff_eyeRetnaMasks.txt",
    "puntos_tiff_eyeWhiteMasks.txt", "puntos_tiff_heartMasks.txt", "puntos_tiff_ileumMasks.txt",
    "puntos_tiff_kidneyMasks.txt", "puntos_tiff_lIntestineMasks.txt", "puntos_tiff_liverMasks.txt",
    "puntos_tiff_lungMasks.txt", "puntos_tiff_muscleMasks.txt", "puntos_tiff_nerveMasks.txt",
    "puntos_tiff_skeletonMasks.txt", "puntos_tiff_spleenMasks.txt", "puntos_tiff_stomachMasks.txt"
]

# Archivos que se deben dividir en 4 clusters (antes eran 2)
cuatro_grupos = {"muscleMasks"}

for archivo in archivos:
    ruta = os.path.join(entrada, archivo)
    if not os.path.exists(ruta):
        print(f"[WARNING] No se encontró {ruta}, se omite.")
        continue

    puntos = np.loadtxt(ruta)
    print(f"\n[OK] Procesando: {archivo} ({puntos.shape[0]} puntos)")

    # Extraer base name sin extensión
    base_name = archivo.replace(".txt", "")

    # Si pertenece a los que tienen 4 grupos
    if any(key in archivo for key in cuatro_grupos):
        print(f" -> Dividiendo {archivo} en 4 clusters...")
        kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
        labels = kmeans.fit_predict(puntos)

        for i in range(6):
            salida_cluster = os.path.join(salida, f"{base_name}_grupo{i+1}.txt")
            with open(salida_cluster, "w") as f:
                for (x, y, z), label in zip(puntos, labels):
                    if label == i:
                        f.write(f"{x} {y} {z}\n")
            print(f"   [OK] Grupo {i+1}: {np.sum(labels==i)} puntos -> {salida_cluster}")

    else:
        # Caso normal: solo copiar el archivo sin dividir
        salida_normal = os.path.join(salida, archivo)
        np.savetxt(salida_normal, puntos)
        print(f" -> Guardado sin dividir: {salida_normal}")
