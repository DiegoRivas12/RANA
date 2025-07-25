import tifffile
import numpy as np
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
    "lIntestineMasks.tiff",
    "liverMasks.tiff",
    "lungMasks.tiff",
    "muscleMasks.tiff",
    "nerveMasks.tiff",
    "skeletonMasks.tiff",
    "spleenMasks.tiff",
    "stomachMasks.tiff",
]

output_dir = "puntos_generados"  # Carpeta donde se guardarán los archivos
os.makedirs(output_dir, exist_ok=True)

verbose = True
# ------------------------------------------------

for archivo in archivos_tiff:
    ruta = os.path.join(path, archivo)

    try:
        stack = tifffile.imread(ruta)
        print(f"[OK] Procesando: {archivo} | Dimensiones: {stack.shape} (frames, alto, ancho)")
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo {archivo}: {e}")
        continue

    num_frames = stack.shape[0]
    all_points = []

    # Extraer puntos (x, y, z)
    for z, frame in enumerate(stack):
        coords = np.column_stack(np.where(frame > 0))  # (y, x)
        if coords.size > 0:
            for y, x in coords:
                all_points.append((float(x), float(y), float(z)))
        if verbose and z % 10 == 0:
            print(f"   Frame {z+1}/{num_frames} - puntos acumulados: {len(all_points)}")

    # Guardar todos los puntos sin muestreo
    salida = os.path.join(output_dir, f"puntos_tiff_{archivo.replace('.tiff','')}.txt")
    with open(salida, "w") as f:
        for p in all_points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"   [OK] {archivo} -> Total puntos: {len(all_points)}")
    print(f"   Archivo generado: {salida}\n")

print("[FINALIZADO] Todos los archivos han sido procesados.")
