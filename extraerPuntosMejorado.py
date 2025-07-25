import tifffile
import numpy as np
import random

# ---------------- CONFIGURACIÓN ----------------
ruta = "imagenT/skeletonMasks.tiff"   # Ruta del archivo TIFF
max_points = 95000         # Número máximo de puntos a guardar
salida = "puntos_tiff_skeletonMasks_"+str(max_points)+".txt" # Archivo de salida

verbose = True             # Mostrar estadísticas
# ------------------------------------------------

try:
    # Leer todas las capas del TIFF
    stack = tifffile.imread(ruta)  # Requiere imagecodecs instalado
    print(f"[OK] Archivo cargado: {ruta}")
    print(f"Dimensiones: {stack.shape} (frames, alto, ancho)")
except Exception as e:
    print(f"[ERROR] No se pudo leer el archivo: {e}")
    exit()

num_frames = stack.shape[0]
all_points = []

for z, frame in enumerate(stack):
    # Encuentra píxeles donde hay información (>0)
    coords = np.column_stack(np.where(frame > 0))  # (y, x)
    if coords.size > 0:
        for y, x in coords:
            all_points.append((float(x), float(y), float(z)))  # (x, y, z)

    if verbose and z % 10 == 0:
        print(f"Procesado frame {z+1}/{num_frames} - puntos acumulados: {len(all_points)}")

# Si hay más puntos que el límite, muestreamos aleatoriamente
if len(all_points) > max_points:
    print(f"[INFO] Se encontraron {len(all_points)} puntos. Muestreando {max_points}...")
    all_points = random.sample(all_points, max_points)

# Guardar en archivo
with open(salida, "w") as f:
    for p in all_points:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

print(f"\n Proceso completado.")
print(f"Total puntos guardados: {len(all_points)}")
print(f"Archivo generado: {salida}")
