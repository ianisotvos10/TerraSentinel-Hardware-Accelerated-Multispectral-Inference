import os
import glob
import argparse
import gc
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import ndimage

from band_loader import (
    read_band_10m,
    read_and_resample_20m,
    get_patch_coordinates,
    extract_patches,
    to_reflectance
)
from cpu_indices import generate_drought_mask
from train_segmentation import SimpleUNet, DroughtDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
TEST_DATA_DIR = os.path.join(REPO_ROOT, "test_data")  
MODELS_DIR = os.path.join(REPO_ROOT, "models")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PATCH_SIZE = 512


def compute_owned_offsets(coords, patch_size=PATCH_SIZE):
    """
    Pentru fiecare patch (y, x, row, col): (rand_start, col_start) din care pixelii ii apartin, adica
    nu au fost deja acoperiti de patch-ul anterior. Ultimul rand/coloana se suprapun cu cele dinainte
    (ancorate la margine); cu aceste offseturi fiecare pixel al scenei se numara exact o data.
    """
    ys = sorted({c[0] for c in coords})
    xs = sorted({c[1] for c in coords})
    offsets = []
    for y, x, r, c in coords:
        ro = max(0, ys[r - 1] + patch_size - y) if r > 0 else 0
        co = max(0, xs[c - 1] + patch_size - x) if c > 0 else 0
        offsets.append((ro, co))
    return offsets


def load_external_region(region_name: str):
    """Incarca o regiune externa de validare, eliberand benzile succesiv pentru a preveni varfurile de RAM."""
    folder_path = os.path.join(TEST_DATA_DIR, region_name)
    if not os.path.exists(folder_path):
        folder_path = os.path.join(DATA_DIR, region_name) 
        
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folderul pentru evaluare externa '{region_name}' nu există nici în {TEST_DATA_DIR} nici în {DATA_DIR}!")

    b04_files = sorted(glob.glob(os.path.join(folder_path, "*_B04_10m.jp2")))
    b08_files = sorted(glob.glob(os.path.join(folder_path, "*_B08_10m.jp2")))
    b11_files = sorted(glob.glob(os.path.join(folder_path, "*_B11_20m.jp2")))
    scl_files = sorted(glob.glob(os.path.join(folder_path, "*_SCL_20m.jp2")))

    if not (b04_files and b08_files and b11_files and scl_files):
        raise ValueError(f"Regiunea externa {region_name} nu contine toate cele 4 benzi necesare (.jp2)!")

    print(f"[Evaluare Externa] Se citesc datele din regiunea: {folder_path}...")
    
    b04_dn, _ = read_band_10m(b04_files[0])
    target_shape = b04_dn.shape
    coords = get_patch_coordinates(target_shape[0], target_shape[1], patch_size=512)
    b04_p = extract_patches(b04_dn, patch_size=512, coords=coords)
    del b04_dn
    
    b08_dn, _ = read_band_10m(b08_files[0])
    b08_p = extract_patches(b08_dn, patch_size=512, coords=coords)
    del b08_dn
    
    b11_dn = read_and_resample_20m(b11_files[0], target_shape, is_mask=False)
    b11_p = extract_patches(b11_dn, patch_size=512, coords=coords)
    del b11_dn
    
    scl_data = read_and_resample_20m(scl_files[0], target_shape, is_mask=True)
    scl_p = extract_patches(scl_data, patch_size=512, coords=coords)
    del scl_data

    n = len(coords)
    features = np.empty((n, 2, 512, 512), dtype=np.float32)
    masks = np.empty((n, 512, 512), dtype=np.uint8)

    for i in range(n):
        r = to_reflectance(b04_p[i])
        nir = to_reflectance(b08_p[i])
        swir1 = to_reflectance(b11_p[i])

        mask, ndvi, ndmi = generate_drought_mask(r, nir, swir1, scl_p[i])
        features[i, 0] = ndvi
        features[i, 1] = ndmi
        masks[i] = mask

    del b04_p, b08_p, b11_p, scl_p
    gc.collect()

    print(f"[Evaluare Externa] S-au generat {n} patch-uri de test pentru regiunea {region_name}.")
    return features, masks, compute_owned_offsets(coords)


def post_process_mask(preds: torch.Tensor, min_size: int = 15) -> torch.Tensor:
    preds_np = preds.cpu().numpy()
    cleaned_preds = np.empty_like(preds_np)
    
    for i in range(preds_np.shape[0]):
        mask = preds_np[i, 0] > 0.5
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            component_sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            too_small = component_sizes < min_size
            remove_pixel = np.isin(labeled, np.where(too_small)[0] + 1)
            labeled[remove_pixel] = 0
            cleaned_preds[i, 0] = (labeled > 0).astype(np.float32)
        else:
            cleaned_preds[i, 0] = mask.astype(np.float32)
            
    return torch.from_numpy(cleaned_preds).to(preds.device)


def compute_confusion_from_binary(preds: torch.Tensor, msks: torch.Tensor):
    preds_np = preds.detach().cpu().numpy().astype(bool)
    msks_np = msks.detach().cpu().numpy().astype(bool)
    
    tp = np.logical_and(preds_np == 1, msks_np == 1).sum()
    fp = np.logical_and(preds_np == 1, msks_np == 0).sum()
    fn = np.logical_and(preds_np == 0, msks_np == 1).sum()
    tn = np.logical_and(preds_np == 0, msks_np == 0).sum()
    return int(tp), int(fp), int(fn), int(tn)


def run_evaluation(region_name: str, min_size: int = 15):
    print("=" * 70)
    print(f"  Evaluare Externa (Out-of-Sample) — Regiunea: {region_name}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = os.path.join(MODELS_DIR, "unet_drought_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelul antrenat nu a fost gasit la {model_path}! Ruleaza mai intai train_segmentation.py.")

    model = SimpleUNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[Model] S-au incarcat ponderele din: {model_path}")

    X_test, y_test, offsets = load_external_region(region_name)
    test_loader = DataLoader(DroughtDataset(X_test, y_test), batch_size=4, shuffle=False)

    tp_sum, fp_sum, fn_sum, tn_sum = 0, 0, 0, 0

    print("\n[Inferenta & Post-procesare] Se ruleaza evaluarea pe datele oarbe...")
    sample_idx = 0
    with torch.no_grad():
        for feats, msks in test_loader:
            feats, msks = feats.to(device), msks.to(device)
            logits = model(feats)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            preds_cleaned = post_process_mask(preds, min_size=min_size)

            # Numaram doar zona "proprie" a fiecarui patch (fara suprapunerea cu patch-ul anterior)
            for j in range(feats.size(0)):
                ro, co = offsets[sample_idx + j]
                tp, fp, fn, tn = compute_confusion_from_binary(
                    preds_cleaned[j:j + 1, :, ro:, co:], msks[j:j + 1, :, ro:, co:])
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
                tn_sum += tn
            sample_idx += feats.size(0)

    iou_pos = tp_sum / max(tp_sum + fp_sum + fn_sum, 1)
    iou_neg = tn_sum / max(tn_sum + fp_sum + fn_sum, 1)
    miou = 0.5 * (iou_pos + iou_neg)
    f1_pos = (2 * tp_sum) / max(2 * tp_sum + fp_sum + fn_sum, 1)
    precision = tp_sum / max(tp_sum + fp_sum, 1)
    recall = tp_sum / max(tp_sum + fn_sum, 1)

    print("\n" + "=" * 70)
    print("  REZULTATE EVALUARE EXTERNA (fidelitate fata de pseudo-eticheta)")
    print("=" * 70)
    print(f"  Regiune testata:        {region_name}")
    print("  Nota [1]: eticheta = regula NDVI/NDMI (aproximare pseudo-label).")
    print("  Nota [2]: metrici pe scena intreaga, fiecare pixel numarat o singura data (suprapunerile de margine excluse).")
    print(f"  Prag minim componenta:  {min_size} pixeli")
    print(f"  True Positives (TP):    {tp_sum}")
    print(f"  False Positives (FP):   {fp_sum}")
    print(f"  False Negatives (FN):   {fn_sum}")
    print(f"  True Negatives (TN):    {tn_sum}")
    print("-" * 70)
    print(f"  Precision (Seceta):     {precision:.4f}")
    print(f"  Recall (Seceta):        {recall:.4f}")
    print(f"  F1-Score (Seceta):      {f1_pos:.4f}")
    print(f"  IoU (Seceta):           {iou_pos:.4f}")
    print(f"  mIoU (Mediu):           {miou:.4f}")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"evaluation_{region_name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "region": region_name, "min_size": min_size, "model": os.path.basename(model_path),
            "pixels_counted": int(tp_sum + fp_sum + fn_sum + tn_sum),
            "tp": int(tp_sum), "fp": int(fp_sum), "fn": int(fn_sum), "tn": int(tn_sum),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1_pos, 4),
            "iou_seceta": round(iou_pos, 4), "miou": round(miou, 4),
        }, f, indent=2)
    print(f"Rezultate salvate in: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluare externa model U-Net cu post-procesare — TerraSentinel-Edge")
    parser.add_argument("--region", type=str, default="val_extern",
                        help="Numele folderului cu datele externe (implicit: val_extern)")
    parser.add_argument("--min_size", type=int, default=15,
                        help="Pragul minim de pixeli pentru eliminarea zgomotului izolat (implicit: 15)")
    args = parser.parse_args()

    run_evaluation(args.region, min_size=args.min_size)