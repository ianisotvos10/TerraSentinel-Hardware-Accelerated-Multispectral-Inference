import os
import glob
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from band_loader import (
    read_band_10m,
    read_and_resample_20m,
    get_patch_coordinates,
    extract_patches,
    to_reflectance
)
from cpu_indices import generate_drought_mask

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

PATCH_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42


class SimpleUNet(nn.Module):
    """Arhitectura U-Net ușoară pentru inferență edge / CPU / GPU."""
    def __init__(self, in_channels=2, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1_1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.enc1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.enc2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bottleneck_2 = nn.Conv2d(128, 128, 3, padding=1)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec2_2 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1_1 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec1_2 = nn.Conv2d(32, 32, 3, padding=1)
        
        # Head
        self.head = nn.Conv2d(32, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        e1 = self.relu(self.enc1_1(x))
        e1 = self.relu(self.enc1_2(e1))
        p1 = self.pool1(e1)
        
        e2 = self.relu(self.enc2_1(p1))
        e2 = self.relu(self.enc2_2(e2))
        p2 = self.pool2(e2)
        
        b = self.relu(self.bottleneck_1(p2))
        b = self.relu(self.bottleneck_2(b))
        
        u2 = self.upconv2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.relu(self.dec2_1(u2))
        d2 = self.relu(self.dec2_2(d2))
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.relu(self.dec1_1(u1))
        d1 = self.relu(self.dec1_2(d1))
        
        return self.head(d1)


class DroughtDataset(Dataset):
    """
    Foloseste torch.from_numpy (zero-copy) si pastreaza mastile ca uint8 in RAM,
    convertindu-le la float32 doar pe batch-ul curent in __getitem__.
    """
    def __init__(self, features: np.ndarray, masks: np.ndarray):
        self.features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self.masks = torch.from_numpy(np.ascontiguousarray(masks, dtype=np.uint8)).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx].float()


def discover_and_load_multi_region_data():
    """
    Descarcă și procesează datele DOAR din folderele care încep cu 'zone_'.
    Include sortare deterministă și eșantionare controlată per regiune pentru protecția RAM-ului.
    """
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Directorul de date nu există la: {DATA_DIR}!")

    subfolders = sorted(
        d for d in os.listdir(DATA_DIR) 
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("zone_")
    )
    
    if not subfolders:
        raise FileNotFoundError(f"Nu s-au găsit subfoldere de antrenament (care să înceapă cu 'zone_') în {DATA_DIR}!")

    print(f"[Multi-Region] S-au găsit subfolderele pentru antrenament: {subfolders}")

    all_features = []
    all_masks = []
    max_patches_per_region = 120  
    for region in subfolders:
        folder_path = os.path.join(DATA_DIR, region)
        print(f"[Procesare] Se extrag patch-uri din regiunea: {region}...")

        b04_files = sorted(glob.glob(os.path.join(folder_path, "*_B04_10m.jp2")))
        b08_files = sorted(glob.glob(os.path.join(folder_path, "*_B08_10m.jp2")))
        b11_files = sorted(glob.glob(os.path.join(folder_path, "*_B11_20m.jp2")))
        scl_files = sorted(glob.glob(os.path.join(folder_path, "*_SCL_20m.jp2")))

        if not (b04_files and b08_files and b11_files and scl_files):
            print(f"[Atenție] Regiunea {region} nu conține toate benzile necesare. Se sare peste.")
            continue

        b04_dn, _ = read_band_10m(b04_files[0])
        target_shape = b04_dn.shape
        b08_dn, _ = read_band_10m(b08_files[0])
        b11_dn = read_and_resample_20m(b11_files[0], target_shape, is_mask=False)
        scl_data = read_and_resample_20m(scl_files[0], target_shape, is_mask=True)

        coords = get_patch_coordinates(target_shape[0], target_shape[1], PATCH_SIZE)
        
        # Eșantionare deterministă uniformă pe grila scenei
        if len(coords) > max_patches_per_region:
            indices = np.linspace(0, len(coords) - 1, max_patches_per_region, dtype=int)
            coords = [coords[i] for i in indices]

        b04_p = extract_patches(b04_dn, PATCH_SIZE, coords)
        del b04_dn
        b08_p = extract_patches(b08_dn, PATCH_SIZE, coords)
        del b08_dn
        b11_p = extract_patches(b11_dn, PATCH_SIZE, coords)
        del b11_dn
        scl_p = extract_patches(scl_data, PATCH_SIZE, coords)
        del scl_data

        region_start = len(all_masks)
        for i in range(len(coords)):
            r = to_reflectance(b04_p[i])
            nir = to_reflectance(b08_p[i])
            swir1 = to_reflectance(b11_p[i])

            mask, ndvi, ndmi = generate_drought_mask(r, nir, swir1, scl_p[i])
            
            feat = np.stack([ndvi, ndmi], axis=0)
            all_features.append(feat)
            all_masks.append(mask)

        region_pos = np.mean([m.mean() for m in all_masks[region_start:]]) * 100
        print(f"[{region}] {len(coords)} patch-uri, pseudo-eticheta pozitiva: {region_pos:.2f}%")

        del b04_p, b08_p, b11_p, scl_p
        gc.collect()

    if not all_features:
        raise RuntimeError("Nu s-au putut extrage patch-uri valide din niciun subfolder 'zone_'!")

    features_arr = np.array(all_features, dtype=np.float32)
    masks_arr = np.array(all_masks, dtype=np.uint8)
    del all_features, all_masks
    gc.collect()

    return features_arr, masks_arr


def compute_confusion_counts(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()
    
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def export_onnx_single_file(model, sample, onnx_path):
    import onnx 

    torch.onnx.export(
        model,
        sample,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    model_proto = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(model_proto, onnx_path, save_as_external_data=False)
    for leftover in glob.glob(onnx_path + "*data*"):
        os.remove(leftover)
    print(f"[ONNX] Salvat ca fisier unic (self-contained): {onnx_path}")


def check_onnx_parity(model, sample, onnx_path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[Atentie] pip install onnxruntime pentru verificarea de paritate ONNX.")
        return

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = session.run(None, {session.get_inputs()[0].name: sample.cpu().numpy()})[0]
    with torch.no_grad():
        pt_out = model(sample).cpu().numpy()

    max_err = float(np.max(np.abs(pt_out - ort_out)))
    ok = bool(np.allclose(pt_out, ort_out, atol=1e-4, rtol=1e-4))
    print(f"[Paritate PyTorch vs ONNX] eroare abs max: {max_err:.2e} | allclose(1e-4): {ok}")
    if not ok:
        raise AssertionError(f"ONNX difera de PyTorch (eroare max {max_err:.2e})")


def train_pipeline():
    print("=== Pipeline Antrenament & Validare Multi-Regiune U-Net (2 canale [NDVI, NDMI]) ===")
    
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    features, masks = discover_and_load_multi_region_data()
    total_patches = len(features)
    
    rng = np.random.default_rng(SEED)
    indices = np.arange(total_patches)
    rng.shuffle(indices)
    
    split_idx = int(0.8 * total_patches)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train = np.ascontiguousarray(features[train_idx])
    y_train = np.ascontiguousarray(masks[train_idx])
    X_val = np.ascontiguousarray(features[val_idx])
    y_val = np.ascontiguousarray(masks[val_idx])

    del features, masks
    gc.collect()

    print(f"[Dataset Universal] Total patch-uri: {total_patches} | Train: {len(X_train)} | Val: {len(X_val)}")

    pos_pixels_pct = (y_train == 1).sum() / y_train.size * 100

    if pos_pixels_pct == 0.0:
        raise RuntimeError("Pseudo-eticheta nu are niciun pixel pozitiv in Train; verifica datele si pragurile.")
    val_pos_pct = (y_val == 1).sum() / max(y_val.size, 1) * 100
    if val_pos_pct == 0.0:
        print("[Atentie] Validarea nu contine pixeli pozitivi: IoU/F1 nu sunt informative.")

    if pos_pixels_pct >= 20.0:
        pos_weight = 1.0
    else:
        pos_weight = float(np.clip((100.0 - pos_pixels_pct) / max(pos_pixels_pct, 1.0), 1.0, 5.0))
    print(f"Procent pixeli clasa pozitiva in Train: {pos_pixels_pct:.2f}% -> pos_weight = {pos_weight:.2f}")

    loader_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(DroughtDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, generator=loader_gen)
    val_loader = DataLoader(DroughtDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispozitiv antrenament: {device}")
    model = SimpleUNet(in_channels=2, out_channels=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_iou = -1.0
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_model_path = os.path.join(MODELS_DIR, "unet_drought_best.pt")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for feats, msks in train_loader:
            feats, msks = feats.to(device), msks.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, msks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        tp_s, fp_s, fn_s, tn_s = 0, 0, 0, 0
        with torch.no_grad():
            for feats, msks in val_loader:
                feats, msks = feats.to(device), msks.to(device)
                logits = model(feats)
                loss = criterion(logits, msks)
                val_loss += loss.item() * feats.size(0)
                tp, fp, fn, tn = compute_confusion_counts(logits, msks)
                tp_s += tp
                fp_s += fp
                fn_s += fn
                tn_s += tn

        val_loss /= len(val_loader.dataset)
        iou_pos = tp_s / max(tp_s + fp_s + fn_s, 1)
        iou_neg = tn_s / max(tn_s + fp_s + fn_s, 1)
        miou = 0.5 * (iou_pos + iou_neg)
        f1_pos = (2 * tp_s) / max(2 * tp_s + fp_s + fn_s, 1)

        print(f"Epoca [{epoch+1:02d}/{EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU(Seceta): {iou_pos:.4f} | mIoU: {miou:.4f} | F1: {f1_pos:.4f}")

        if iou_pos > best_iou:
            best_iou = iou_pos
            torch.save(model.state_dict(), best_model_path)

    print(f"\n[Checkpoint] Cel mai bun model PyTorch (IoU_Seceta={best_iou:.4f}) salvat in: {best_model_path}")

    onnx_path = os.path.join(MODELS_DIR, "unet_drought.onnx")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()

    ref_pool = X_val if len(X_val) > 0 else X_train
    sample = torch.from_numpy(ref_pool[:1]).float().to(device)  # (1, 2, 512, 512)
    export_onnx_single_file(model, sample, onnx_path)
    check_onnx_parity(model, sample, onnx_path)


if __name__ == "__main__":
    train_pipeline()