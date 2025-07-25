import tifffile
import numpy as np
from sklearn.cluster import KMeans
import os

# ---------------- CONFIGURACIÓN ----------------
path = "imagenT/"
archivos_tiff = [
    "bloodMasks.tiff",
    "brainMasks.tiff",
    "duodenumMasks.tiff",
    "eyeMasks.tiff",
    "eyeRetnaMasks.tiff",
    "eyeWhiteMasks.tiff",
    "heartMasks.tiff",
    "ileumMasks.tiff",
    "kidneyMasks.tiff",
    "lIntestineMasks",
    "liverMasks.tiff",
    "lungMasks.tiff",
    "muscleMasks.tiff",
    "nerveMasks.tiff",
    "skeletonMasks.tiff",
    "spleenMasks.tiff",
    "stomachMasks.tiff",
]  # Lista de archivos a procesar
salida = "puntos_finales1.txt"  # Archivo combinado de salida
num_clusters = 25             # Puntos por archivo (KMeans)
verbose = True
# ------------------------------------------------

with open(salida, "w") as out_file:
    for ruta in archivos_tiff:
        try:
            # Leer TIFF
            stack = tifffile.imread(path+ruta)
            print(f"[OK] Archivo cargado: {ruta}")
            print(f"Dimensiones: {stack.shape} (frames, alto, ancho)")
        except Exception as e:
            print(f"[ERROR] No se pudo leer {ruta}: {e}")
            continue

        all_points = []
        for z, frame in enumerate(stack):
            coords = np.column_stack(np.where(frame > 0))  # (y, x)
            if coords.size > 0:
                for y, x in coords:
                    all_points.append((float(x), float(y), float(z)))

        if verbose:
            print(f"[INFO] {ruta}: {len(all_points)} puntos detectados.")

        if len(all_points) == 0:
            print(f"[WARN] {ruta}: No se encontraron puntos.")
            continue

        # Reducir con KMeans
        if len(all_points) > num_clusters:
            print(f"[INFO] Aplicando KMeans para {ruta}...")
            all_points = np.array(all_points)
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
            kmeans.fit(all_points)
            centroides = kmeans.cluster_centers_
        else:
            centroides = np.array(all_points)

        # Guardar en archivo combinado
        #out_file.write(f"# Archivo: {os.path.basename(ruta)}\n")
        for x, y, z in centroides:
            out_file.write(f"{x:.4f} {y:.4f} {z:.4f}\n")

        print(f"[OK] Procesado {ruta}, {len(centroides)} puntos guardados.\n")

print(f"[FINALIZADO] Todos los resultados en '{salida}'")
