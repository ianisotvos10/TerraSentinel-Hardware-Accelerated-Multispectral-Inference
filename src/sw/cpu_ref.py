import numpy as np
from band_loader import to_reflectance
from cpu_indices import compute_normalized_index, generate_drought_mask

def cpu_reference_pipeline(b04_dn: np.ndarray, b08_dn: np.ndarray, b11_dn: np.ndarray, scl: np.ndarray = None):
    """
    Referinta CPU: aceleasi functii ca la antrenare (va fi si referinta pentru kernelul CUDA din saptamana 10).
    Foloseste exact aceleasi functii ca pipeline-ul de antrenare:
      - to_reflectance: DN==0 -> 0.0, altfel clip((DN - 1000)/10000, 0.0, 1.0)
      - compute_normalized_index: |den| <= 1e-7 -> 0.0, altfel clip((b1 - b2)/(b1 + b2), -1.0, 1.0)
    Returneaza tensorul CHW (2, H, W) cu [NDVI, NDMI] care intra in TensorRT, plus masca (optional).
    """
    r = to_reflectance(b04_dn)
    nir = to_reflectance(b08_dn)
    swir1 = to_reflectance(b11_dn)

    if scl is not None:
        mask, ndvi, ndmi = generate_drought_mask(r, nir, swir1, scl)
    else:
        ndvi = compute_normalized_index(nir, r)
        ndmi = compute_normalized_index(nir, swir1)
        mask = None

    # Tensor CHW float32 [2, H, W]: formatul de intrare al retelei (tinta kernelului CUDA din saptamana 10)
    trt_input_chw = np.stack([ndvi, ndmi], axis=0).astype(np.float32)
    return trt_input_chw, ndvi, ndmi, mask

def self_test():
    """
    Vector de test determinist (4 pixeli) (de refolosit la testarea kernelului CUDA, saptamana 10):
      Pixel 0: B04=2500 (r=0.15), B08=5500 (nir=0.45), B11=4000 (swir=0.30) -> NDVI=0.50, NDMI=0.20
      Pixel 1: B04=0 (NoData),    B08=0 (NoData),      B11=0 (NoData)       -> NDVI=0.00, NDMI=0.00
      Pixel 2: B04=1000 (r=0.00), B08=1000 (nir=0.00), B11=1000 (swir=0.00) -> NDVI=0.00, NDMI=0.00 (numitor 0)
      Pixel 3: B04=12000 (r=1.0), B08=3000 (nir=0.20), B11=7000 (swir=0.60) -> NDVI=-0.6666667, NDMI=-0.50
    """
    b04_dn = np.array([[2500, 0], [1000, 12000]], dtype=np.uint16)
    b08_dn = np.array([[5500, 0], [1000, 3000]], dtype=np.uint16)
    b11_dn = np.array([[4000, 0], [1000, 7000]], dtype=np.uint16)

    trt_chw, ndvi, ndmi, _ = cpu_reference_pipeline(b04_dn, b08_dn, b11_dn)

    expected_ndvi = np.array([[0.5, 0.0], [0.0, -2.0 / 3.0]], dtype=np.float32)
    expected_ndmi = np.array([[0.2, 0.0], [0.0, -0.5]], dtype=np.float32)

    assert np.allclose(ndvi, expected_ndvi, atol=1e-6), f"Eroare NDVI: {ndvi} vs {expected_ndvi}"
    assert np.allclose(ndmi, expected_ndmi, atol=1e-6), f"Eroare NDMI: {ndmi} vs {expected_ndmi}"
    assert trt_chw.shape == (2, 2, 2)
    print("[CPU_REF PASS] Self-test aliniat: DN=12000 plafonat la 1.0, iar numitorul 0 returneaza 0.0.")

if __name__ == "__main__":
    self_test()