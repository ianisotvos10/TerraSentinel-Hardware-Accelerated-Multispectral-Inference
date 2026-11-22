import os
import glob
import numpy as np
from band_loader import (
    read_band_10m,
    read_and_resample_20m,
    extract_patches,
    to_reflectance,
    get_patch_coordinates
)

# Clase SCL invalide conform specificatiei ESA:
# 0=No Data, 1=Saturated/Defective, 3=Cloud Shadow, 8=Cloud Med, 9=Cloud High, 10=Cirrus, 11=Snow/Ice
INVALID_SCL_CLASSES = [0, 1, 3, 8, 9, 10, 11]

def compute_normalized_index(band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
    """
    Calcul standard (B1 - B2) / (B1 + B2).
    Conventie unica CPU/CUDA: cand numitorul este <= 1e-7, returneaza 0.0, apoi clip in [-1.0, 1.0].
    Implementare vectorizata; rezultat identic bit cu bit cu varianta initiala.
    """
    num = band1 - band2
    den = band1 + band2
    out = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=out, where=np.abs(den) > 1e-7)
    return np.clip(out, -1.0, 1.0, out=out)

def generate_drought_mask(b04_refl: np.ndarray, b08_refl: np.ndarray, b11_refl: np.ndarray, scl_patch: np.ndarray):
    ndvi = compute_normalized_index(b08_refl, b04_refl)
    ndmi = compute_normalized_index(b08_refl, b11_refl)

    drought = (ndvi > 0.2) & (ndmi < 0.0)

    invalid_pixels = np.isin(scl_patch, INVALID_SCL_CLASSES)
    drought[invalid_pixels] = False

    return drought.astype(np.uint8), ndvi, ndmi

def visualize_patch(drought_mask: np.ndarray, ndvi: np.ndarray, ndmi: np.ndarray):
    import matplotlib.pyplot as plt 

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axs[0].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axs[0].set_title("NDVI (Vegetatie)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(ndmi, cmap='BrBG', vmin=-1, vmax=1)
    axs[1].set_title("NDMI (Umiditate)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    axs[2].imshow(drought_mask, cmap='gray', vmin=0, vmax=1)
    pos_ratio = drought_mask.mean() * 100
    axs[2].set_title(f"Pseudo-Label Mask ({pos_ratio:.1f}% pozitiv)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data"))

    zones = sorted(d for d in os.listdir(DATA_DIR) if d.startswith("zone_") and os.path.isdir(os.path.join(DATA_DIR, d)))
    if not zones:
        raise FileNotFoundError(f"Nu exista foldere zone_* in {DATA_DIR}!")
    
    folder_path = os.path.join(DATA_DIR, zones[0])

    b04_path = sorted(glob.glob(os.path.join(folder_path, "*_B04_10m.jp2")))[0]
    b08_path = sorted(glob.glob(os.path.join(folder_path, "*_B08_10m.jp2")))[0]
    b11_path = sorted(glob.glob(os.path.join(folder_path, "*_B11_20m.jp2")))[0]
    scl_path = sorted(glob.glob(os.path.join(folder_path, "*_SCL_20m.jp2")))[0]

    b04_dn, _ = read_band_10m(b04_path)
    target_shape = b04_dn.shape
    b08_dn, _ = read_band_10m(b08_path)
    b11_dn = read_and_resample_20m(b11_path, target_shape, is_mask=False)
    scl_data = read_and_resample_20m(scl_path, target_shape, is_mask=True)

    all_coords = get_patch_coordinates(target_shape[0], target_shape[1], patch_size=512)
    
    idx_patch = min(200, len(all_coords) - 1)
    test_coord = [all_coords[idx_patch]]

    r_patch = to_reflectance(extract_patches(b04_dn, 512, test_coord)[0])
    nir_patch = to_reflectance(extract_patches(b08_dn, 512, test_coord)[0])
    swir_patch = to_reflectance(extract_patches(b11_dn, 512, test_coord)[0])
    scl_patch = extract_patches(scl_data, 512, test_coord)[0]

    mask, ndvi, ndmi = generate_drought_mask(r_patch, nir_patch, swir_patch, scl_patch)
    visualize_patch(mask, ndvi, ndmi)