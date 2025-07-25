import os
import numpy as np
from sklearn.cluster import KMeans

# Carpetas
entrada = "puntos_generados"
salida = "puntos_separados"
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

# Archivos que se deben dividir en dos clusters
dos_grupos = {"eyeMasks", "eyeRetnaMasks", "eyeWhiteMasks", "kidneyMasks", "lungMasks"}

for archivo in archivos:
    ruta = os.path.join(entrada, archivo)
    if not os.path.exists(ruta):
        print(f"[WARNING] No se encontró {ruta}, se omite.")
        continue

    puntos = np.loadtxt(ruta)
    print(f"\n[OK] Procesando: {archivo} ({puntos.shape[0]} puntos)")

    # Extraer base name sin extensión
    base_name = archivo.replace(".txt", "")

    # Si pertenece a los que tienen 2 grupos
    if any(key in archivo for key in dos_grupos):
        print(f" -> Dividiendo {archivo} en 2 clusters...")
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
        labels = kmeans.fit_predict(puntos)

        salida1 = os.path.join(salida, f"{base_name}_grupo1.txt")
        salida2 = os.path.join(salida, f"{base_name}_grupo2.txt")

        with open(salida1, "w") as f1, open(salida2, "w") as f2:
            for (x, y, z), label in zip(puntos, labels):
                if label == 0:
                    f1.write(f"{x} {y} {z}\n")
                else:
                    f2.write(f"{x} {y} {z}\n")

        print(f"   [OK] Grupo 1: {np.sum(labels==0)} puntos -> {salida1}")
        print(f"   [OK] Grupo 2: {np.sum(labels==1)} puntos -> {salida2}")

    else:
        # Caso normal: solo copiar el archivo sin dividir
        salida_normal = os.path.join(salida, archivo)
        np.savetxt(salida_normal, puntos)
        print(f" -> Guardado sin dividir: {salida_normal}")
