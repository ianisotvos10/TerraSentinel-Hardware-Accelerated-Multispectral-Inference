# TerraSentinel-Edge

Detecția secetei din imagini Sentinel-2, rulată pe o placă NVIDIA Jetson, cu o variantă de referință pe CPU și una accelerată pe GPU (CUDA C++ și TensorRT).

## Despre proiect

Proiectul ia imagini multispectrale Sentinel-2 (L2A) și produce o hartă a zonelor afectate de secetă. Din benzile **B04 (Red)**, **B08 (NIR)** și **B11 (SWIR)** calculez indicii **NDVI** și **NDMI**, folosind masca **SCL** (Scene Classification Layer) pentru filtrarea norilor, umbrelor și zăpezii. Un model de segmentare (`SimpleUNet`, bottleneck 128 filtre) primește tensorul bicanal `[2, 512, 512]` și clasifică fiecare pixel ca „secetă" sau „normal", urmat de post-procesare morfologică (eliminare componente < 15 px).

Tot pipeline-ul rulează local pe Jetson, fără cloud. Ideea este să văd cât câștig aduce GPU-ul față de CPU-ul ARM al aceleiași plăci și cât de multă precizie se pierde când trec modelul în FP16 și INT8.

**Stadiu (Săptămâna 8 finalizată):** Pipeline-ul complet Python/PyTorch, antrenarea, evaluarea *out-of-sample* (`val_extern`), exportul ONNX (`models/unet_drought.onnx`) și profilarea baseline pe CPU (`results/cpu_baseline.json`) sunt finalizate. Urmează implementarea CUDA/TensorRT (`src/hw/`).

### Rezultate preliminare — Evaluare Externă (`val_extern`, out-of-sample)

| Metrică | Valoare |
| :--- | :--- |
| **Precision (Secetă)** | `0.9397` |
| **Recall (Secetă)** | `0.9674` |
| **F1-Score (Secetă)** | `0.9533` |
| **IoU (Secetă)** | `0.9108` |
| **mIoU (Mediu)** | `0.9176` |

*(Raport complet în `results/evaluation_val_extern.json`; speedup-ul GPU și comparația FP32/FP16/INT8 vor apărea în `docs/Documentatie_Proiect.docx` după săptămâna 14.)*

## Cum este făcut

Am împărțit proiectul în două implementări ale aceluiași pipeline, ca să le pot compara corect:

| Componentă | Varianta software (CPU) | Varianta hardware (GPU) |
| :--- | :--- | :--- |
| **Indici spectrali** | NumPy (vectorizat) și referință deterministă (`cpu_ref`) | Kernel CUDA, un thread per pixel |
| **Model** | PyTorch / ONNX Runtime pe CPU | Motor TensorRT (FP32, FP16, INT8) |
| **Cod** | `src/sw/` | `src/hw/` |

Patru idei de optimizare stau la baza variantei pe GPU:

1. **Memorie fără copieri inutile**  
   Jetson are memorie unificată: CPU-ul și GPU-ul folosesc aceeași memorie fizică. Din acest motiv folosesc memorie mapată (`cudaHostAlloc` cu `cudaHostAllocMapped`, sau `cudaMallocManaged`) în loc de `cudaMemcpy`. Kernelurile citesc direct benzile încărcate de ARM. Voi compara ambele variante de alocare cu `cudaMemcpy`, ca să măsor diferența.

2. **Un singur kernel pentru preprocesare**  
   Preprocesarea se face într-un singur kernel CUDA, ca să citesc memoria globală o singură dată. Kernelul calculează indicii, normalizează valorile și rearanjează tensorul din HWC în CHW, formatul cerut de TensorRT. Folosesc shared memory pentru tile-urile de pixeli dintr-un bloc. Dacă fuziunea se dovedește greu de depanat, păstrez un kernel separat pentru fiecare indice și compar timpii.

3. **Cuantizare cu TensorRT**  
   Antrenez modelul în FP32, îl export în ONNX și construiesc motorul TensorRT în FP16, apoi în INT8 cu post-training quantization. Pentru INT8 folosesc un set de calibrare din scene reprezentative (`src/sw/make_calib_set.py`). Compar FP32, FP16 și INT8 după mIoU, F1, eroare numerică și timp. Dacă INT8 pierde prea multă precizie, rămân pe FP16.

4. **Pipeline cu CUDA Streams**  
   Scenele Sentinel-2 sunt mari (`10980×10980` pixeli), deci le tai în patch-uri (`512×512`) și le procesez în flux:
   - ARM-ul citește patch-ul `N` de pe disc;
   - GPU-ul face preprocesarea pentru patch-ul `N−1`;
   - TensorRT face inferența pe patch-ul `N−2`.

   Etapele sunt sincronizate cu CUDA Streams. Statisticile zonale (procent de teren afectat per scenă) se scriu într-un fișier JSON, iar aplicația rulează headless, fără interfață grafică.

## Ce măsor

Compar varianta CPU cu varianta GPU pe scene care nu au fost folosite la antrenare:
- timpul de preprocesare (`t_pre`) și timpul de inferență (`t_inf`);
- timpul total (`T_total`), inclusiv citirea de pe disc (`t_load`);
- speedup-ul `S = T_CPU / T_GPU`;
- calitatea hărților: IoU, F1 și mIoU;
- eroarea numerică dintre CPU și GPU (la indici și la ieșirea rețelei);
- memoria folosită și consumul de putere (`tegrastats`).
