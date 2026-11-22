import os
import json
import time
import glob
import argparse
import platform
import statistics
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

from band_loader import (
    read_band_10m,
    read_and_resample_20m,
    get_patch_coordinates,
    extract_patches,
    to_reflectance
)
from cpu_indices import compute_normalized_index

try:
    import onnxruntime as _ort
    ORT_VERSION = _ort.__version__
except ImportError:
    _ort = None
    ORT_VERSION = None

PATCH_SIZE = 512
WARMUP_RUNS = 1
REPEAT_RUNS = 3


def create_session(threads=None):
    """Creeaza sesiunea ONNX Runtime (CPU). Esueaza explicit daca lipsesc modelul sau onnxruntime."""
    if _ort is None:
        raise ImportError("onnxruntime nu este instalat (pip install onnxruntime).")
    onnx_path = os.path.join(MODELS_DIR, "unet_drought.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Modelul ONNX nu exista la {onnx_path}. Ruleaza mai intai train_segmentation.py.")
    options = _ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return _ort.InferenceSession(onnx_path, sess_options=options, providers=['CPUExecutionProvider'])


def benchmark_load_bands(region_name=None):
    """Faza 1: Citirea dinamică a benzilor din subfolderele multi-regiune și resampling 20m -> 10m."""
    zones = sorted(d for d in os.listdir(DATA_DIR)
                   if d.startswith("zone_") and os.path.isdir(os.path.join(DATA_DIR, d)))
    if region_name:
        if region_name not in zones:
            raise FileNotFoundError(f"Regiunea '{region_name}' nu exista. Disponibile: {zones}")
        target_folder = region_name
    else:
        if not zones:
            raise FileNotFoundError(f"Nu exista foldere zone_* in {DATA_DIR}!")
        target_folder = zones[0]
    folder_path = os.path.join(DATA_DIR, target_folder)

    b04_files = sorted(glob.glob(os.path.join(folder_path, "*_B04_10m.jp2")))
    b08_files = sorted(glob.glob(os.path.join(folder_path, "*_B08_10m.jp2")))
    b11_files = sorted(glob.glob(os.path.join(folder_path, "*_B11_20m.jp2")))

    if not (b04_files and b08_files and b11_files):
        raise FileNotFoundError(f"Lipsesc benzile necesare în subfolderul {target_folder}!")

    print(f"  Regiune selectată pentru benchmark: {target_folder}")
    t0 = time.perf_counter()
    b04_dn, _ = read_band_10m(b04_files[0])
    b08_dn, _ = read_band_10m(b08_files[0])
    target_shape = b04_dn.shape
    b11_dn = read_and_resample_20m(b11_files[0], target_shape, is_mask=False)
    t_load = time.perf_counter() - t0

    return b04_dn, b08_dn, b11_dn, t_load, target_folder


def benchmark_preprocessing(b04_patches, b08_patches, b11_patches):
    """Faza 2: Conversie reflectanta + calcul indici NDVI/NDMI pe CPU."""
    n = b04_patches.shape[0]
    features = np.empty((n, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n):
        r = to_reflectance(b04_patches[i])
        nir = to_reflectance(b08_patches[i])
        swir1 = to_reflectance(b11_patches[i])

        features[i, 0] = compute_normalized_index(nir, r)      # NDVI
        features[i, 1] = compute_normalized_index(nir, swir1)   # NDMI
    t_pre = time.perf_counter() - t0

    return features, t_pre


def benchmark_inference_onnx(session, features):
    """Faza 3: Inferenta ONNX Runtime pe CPU, un patch pe rand (batch 1)."""
    input_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    for i in range(features.shape[0]):
        session.run(None, {input_name: features[i:i + 1]})
    return time.perf_counter() - t0


def summarize(times):
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "median": float(statistics.median(times)),
        "min": float(min(times)),
    }


def run_benchmark(num_patches=10, use_all=False, region_name=None, repeats=REPEAT_RUNS, threads=None):
    print("=" * 70)
    print("  CPU Benchmark — TerraSentinel-Edge (Multi-Region)")
    print("=" * 70)

    session = create_session(threads)
    print(f"  ONNX Runtime {ORT_VERSION}, intra_op threads: {threads if threads else 'implicit'}")

    print("\n[Faza 1] Incarcare benzi de pe disc...")
    b04_dn, b08_dn, b11_dn, t_load, active_region = benchmark_load_bands(region_name)
    h, w = b04_dn.shape
    print(f"  Dimensiune scena: {h} x {w}")
    print(f"  t_load = {t_load:.3f} s")

    all_coords = get_patch_coordinates(h, w, PATCH_SIZE)
    total_patches = len(all_coords)

    if use_all:
        coords = all_coords
    else:
        num_patches = min(num_patches, total_patches)
        idx = np.linspace(0, total_patches - 1, num_patches, dtype=int)
        coords = [all_coords[i] for i in idx]

    n = len(coords)
    print(f"\n  Patch-uri selectate: {n} din {total_patches} (patch_size={PATCH_SIZE})")

    b04_p = extract_patches(b04_dn, PATCH_SIZE, coords)
    b08_p = extract_patches(b08_dn, PATCH_SIZE, coords)
    b11_p = extract_patches(b11_dn, PATCH_SIZE, coords)

    del b04_dn, b08_dn, b11_dn

    print(f"\n[Faza 2] Preprocesare CPU (reflectanta + indici), {repeats} rulari...")
    t_pre_runs = []
    for run in range(repeats):
        features, t_pre = benchmark_preprocessing(b04_p, b08_p, b11_p)
        t_pre_runs.append(t_pre)
        print(f"  Run {run+1}: t_pre = {t_pre:.3f} s")

    pre = summarize(t_pre_runs)
    t_pre_mean = pre["mean"]
    t_pre_per_patch = t_pre_mean / n
    print(f"  Media: {pre['mean']:.3f} ± {pre['std']:.3f} s | mediana {pre['median']:.3f} s | "
          f"min {pre['min']:.3f} s ({t_pre_per_patch*1000:.2f} ms/patch)")

    print(f"\n[Faza 3] Inferenta ONNX Runtime CPU, {repeats} rulari (dupa {WARMUP_RUNS} rulare de incalzire)...")
    for _ in range(WARMUP_RUNS):
        session.run(None, {session.get_inputs()[0].name: features[:1]})

    t_inf_runs = []
    for run in range(repeats):
        t_inf = benchmark_inference_onnx(session, features)
        t_inf_runs.append(t_inf)
        print(f"  Run {run+1}: t_inf = {t_inf:.3f} s")

    inf = summarize(t_inf_runs)
    t_inf_mean = inf["mean"]
    t_inf_per_patch = t_inf_mean / n
    print(f"  Media: {inf['mean']:.3f} ± {inf['std']:.3f} s | mediana {inf['median']:.3f} s | "
          f"min {inf['min']:.3f} s ({t_inf_per_patch*1000:.2f} ms/patch)")

    
    t_compute_selected = t_pre_mean + t_inf_mean
    T_total = t_load + (t_pre_per_patch + t_inf_per_patch) * total_patches

    print("\n" + "=" * 70)
    print("  REZUMAT BASELINE CPU")
    print("=" * 70)
    print(f"  Regiune evaluată:   {active_region}")
    print(f"  Patch-uri:          {n} x {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  t_load  (I/O):      {t_load:.3f} s")
    print(f"  t_pre   (indici):   {t_pre_mean:.3f} s  ({t_pre_per_patch*1000:.2f} ms/patch)")
    print(f"  t_inf   (ONNX):     {t_inf_mean:.3f} s  ({t_inf_per_patch*1000:.2f} ms/patch)")
    print(f"  T_compute (selectie): {t_compute_selected:.3f} s  ({n} patch-uri)")
    print(f"  T_total (scena):    {T_total:.3f} s  (t_load + cost/patch x {total_patches})")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "platform": "CPU",
        "region": active_region,
        "patch_size": PATCH_SIZE,
        "num_patches": n,
        "total_patches_scene": total_patches,
        "scene_pixels": h * w,
        "cpu_count": os.cpu_count(),
        "ort_threads": threads if threads else "implicit",
        "onnxruntime_version": ORT_VERSION,
        "environment": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "repeat_runs": repeats,
        "timings_seconds": {
            "t_load": round(t_load, 4),
            "t_pre_mean": round(t_pre_mean, 4),
            "t_pre_std": round(pre["std"], 4),
            "t_pre_median": round(pre["median"], 4),
            "t_pre_min": round(pre["min"], 4),
            "t_pre_per_patch": round(t_pre_per_patch, 6),
            "t_inf_mean": round(t_inf_mean, 4),
            "t_inf_std": round(inf["std"], 4),
            "t_inf_median": round(inf["median"], 4),
            "t_inf_min": round(inf["min"], 4),
            "t_inf_per_patch": round(t_inf_per_patch, 6),
            "T_compute_selected": round(t_compute_selected, 4),
            "T_total": round(T_total, 4),
        },
        "t_pre_all_runs": [round(t, 4) for t in t_pre_runs],
        "t_inf_all_runs": [round(t, 4) for t in t_inf_runs],
    }

    out_path = os.path.join(RESULTS_DIR, "cpu_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRezultate salvate in: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CPU baseline — TerraSentinel-Edge")
    parser.add_argument("--num_patches", type=int, default=10,
                        help="Numarul de patch-uri de evaluat (implicit: 10)")
    parser.add_argument("--all", action="store_true",
                        help="Ruleaza pe toate patch-urile scenei")
    parser.add_argument("--region", type=str, default=None,
                        help="Numele folderului de regiune (ex: zone_transilvania, zone_campie, zone_mediteraneene)")
    parser.add_argument("--repeats", type=int, default=REPEAT_RUNS,
                        help=f"Numarul de repetari ale masuratorilor (implicit: {REPEAT_RUNS})")
    parser.add_argument("--threads", type=int, default=None,
                        help="Numarul de thread-uri intra-op ONNX Runtime (implicit: decide ONNX Runtime)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats trebuie sa fie >= 1")

    run_benchmark(num_patches=args.num_patches, use_all=args.all, region_name=args.region,
                  repeats=args.repeats, threads=args.threads)