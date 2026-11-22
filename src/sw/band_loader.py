import os
import numpy as np
import rasterio
from rasterio.enums import Resampling

QUANTIFICATION_VALUE = 10000.0
RADIO_ADD_OFFSET = -1000.0  

def to_reflectance(dn_array: np.ndarray) -> np.ndarray:
    """
    Contract unic CPU/GPU pentru conversia Digital Numbers (uint16) la reflectanta [0.0, 1.0].
    Formula L2A: (DN - 1000) / 10000. Pixelii NoData (DN == 0) raman 0.0.
    Implementare vectorizata (fara indexare cu masca); rezultat identic bit cu bit cu varianta initiala.
    """
    refl = (dn_array.astype(np.float32) + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE
    refl[dn_array == 0] = 0.0
    return np.clip(refl, 0.0, 1.0, out=refl)

def read_band_10m(filepath: str):
    """Citeste o banda nativa de 10m (uint16) si profilul geospatial."""
    print(f"Incarcare 10m: {os.path.basename(filepath)}")
    with rasterio.open(filepath) as src:
        return src.read(1), src.profile

def read_and_resample_20m(filepath: str, target_shape: tuple, is_mask: bool = False) -> np.ndarray:
    """Citeste o banda de 20m si face upsampling la rezolutia de 10m (target_shape)."""
    print(f"Upsampling 20m -> 10m: {os.path.basename(filepath)}")
    resampling_method = Resampling.nearest if is_mask else Resampling.bilinear
    with rasterio.open(filepath) as src:
        data = src.read(
            1,
            out_shape=target_shape,
            resampling=resampling_method
        )
    return data

def get_patch_coordinates(height: int, width: int, patch_size: int = 512) -> list:
    """
    Genereaza coordonatele (y, x, row_idx, col_idx) pentru acoperirea integrala a scenei.
    Pentru 10980x10980 cu patch_size=512, genereaza 22x22 = 484 patch-uri:
    ultimul tile de pe margine este ancorat la 10980 - 512 = 10468 (fara pierdere de pixeli).
    """
    if height < patch_size or width < patch_size:
        raise ValueError(f"Scena {height}x{width} este mai mica decat patch-ul de {patch_size}x{patch_size}")

    y_starts = list(range(0, height - patch_size + 1, patch_size))
    if y_starts[-1] + patch_size < height:
        y_starts.append(height - patch_size)

    x_starts = list(range(0, width - patch_size + 1, patch_size))
    if x_starts[-1] + patch_size < width:
        x_starts.append(width - patch_size)

    coords = []
    for r_idx, y in enumerate(y_starts):
        for c_idx, x in enumerate(x_starts):
            coords.append((y, x, r_idx, c_idx))
    return coords

def extract_patches(band_data: np.ndarray, patch_size: int = 512, coords: list = None) -> np.ndarray:
    """Extrage patch-urile 2D doar pentru coordonatele cerute (economiseste memorie RAM)."""
    if coords is None:
        coords = get_patch_coordinates(band_data.shape[0], band_data.shape[1], patch_size)
    
    patches = np.empty((len(coords), patch_size, patch_size), dtype=band_data.dtype)
    for i, (y, x, _, _) in enumerate(coords):
        patches[i] = band_data[y:y + patch_size, x:x + patch_size]
    return patches