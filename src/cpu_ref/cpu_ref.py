import rasterio
import numpy as np
import time
import os

def run_cpu_baseline(red_path, nir_path, output_path):
    print("=== Profilare CPU: Calcul NDVI ===")
    start_total = time.time()

    start_io = time.time()
    with rasterio.open(red_path) as src_red:
        red = src_red.read(1).astype('float32')
        profile = src_red.profile
    with rasterio.open(nir_path) as src_nir:
        nir = src_nir.read(1).astype('float32')
    io_time = time.time() - start_io
    print(f"[I/O] Timp citire fisiere JP2: {io_time:.4f} secunde")

    start_compute = time.time()
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (nir - red) / (nir + red)
    ndvi = np.nan_to_num(ndvi, nan=-1.0)
    compute_time = time.time() - start_compute
    print(f"[CPU] Timp calcul matematic:   {compute_time:.4f} secunde")

    profile.update(dtype=rasterio.float32, count=1, driver='GTiff')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ndvi, 1)

    print(f"[Total] Timp executie complet: {time.time() - start_total:.4f} secunde")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DIR_PATH = os.path.join(BASE_DIR, "..", "..", "data")
    
    B04_FILE = os.path.join(DIR_PATH, "T34TFS_20261014T092029_B04_10m.jp2")
    B08_FILE = os.path.join(DIR_PATH, "T34TFS_20261014T092029_B08_10m.jp2")
    OUTPUT_FILE = os.path.join(DIR_PATH, "ndvi_baseline_cpu.tif")

    run_cpu_baseline(B04_FILE, B08_FILE, OUTPUT_FILE)