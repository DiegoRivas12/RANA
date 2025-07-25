from PIL import Image
import numpy as np

ruta = "brainMasks.tiff"
img = Image.open(ruta)

print(f"Número de frames (capas): {img.n_frames}")

for i in range(img.n_frames):
    img.seek(i)
    frame = np.array(img)
    print(f"Frame {i}: valores únicos {np.unique(frame)}")
