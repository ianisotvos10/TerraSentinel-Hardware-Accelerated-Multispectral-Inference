# TerraSentinel: Hardware-Accelerated Multispectral Edge Inference

## Abstract
TerraSentinel reprezintă un pipeline avansat de Edge Computing dezvoltat pe arhitectura System-on-Chip (SoC) NVIDIA Jetson, dedicat segmentării semantice ultra-rapide a imaginilor satelitare multispectrale (Sentinel-2). Proiectul vizează detecția autonomă a secetei meteorologice și agricole, adresând limitările severe de resurse (power & thermal budget, memorie partajată) specifice dispozitivelor Edge. 

Implementarea integrează optimizări low-level la nivelul memoriei unificate (UMA), paralelism masiv prin CUDA și accelerarea inferenței rețelelor neuronale profunde prin TensorRT, transformând platforma Jetson într-un senzor IoT autonom, complet decuplat de cloud.

## Arhitectura Hardware-Software (Co-Design)

Sistemul este fundamentat pe patru optimizări arhitecturale critice, menite să maximizeze throughput-ul (cadre pe secundă) și să minimizeze latența globală (end-to-end latency):

### 1. Optimizarea Subsistemului de Memorie (Zero-Copy & Pinned Memory)
Spre deosebire de arhitecturile discrete (dGPU), SoC-ul Jetson utilizează o arhitectură de memorie unificată (UMA). Pentru a elimina bottleneck-ul clasic reprezentat de transferul de date via magistrala de sistem, TerraSentinel utilizează **Page-Locked (Pinned) Mapped Memory**. Alocarea imaginilor satelitare se face în zone de memorie nepaginabile, permițând kernelurilor CUDA și motorului TensorRT să acceseze pixelii direct din spațiul fizic alocat de procesorul ARM (Zero-Copy), anulând overhead-ul apelurilor sincrone `cudaMemcpy`.

### 2. Preprocesare CUDA Monolitică (Operator Fusion)
Algoritmii de procesare spectrală sunt mutați de pe procesorul gazdă (CPU) pe accelerator (GPU). Pentru a minimiza numărul de accesări ale memoriei globale (Global Memory) cu latență ridicată, operațiile de preprocesare sunt fuzionate într-un singur kernel CUDA monolitic. Acest kernel preia datele brute, calculează indicele de vegetație (NDVI), execută normalizarea radiometrică și rearanjează tensorii din formatul standard $HWC$ (Height-Width-Channel) în formatul planar $CHW$ necesar motorului TensorRT, utilizând intensiv **Shared Memory** la nivel de Streaming Multiprocessor (SM).

### 3. Inferență AI Cuantizată (TensorRT INT8)
Modelul de segmentare (ex. U-Net / SegFormer), antrenat inițial în precizie flotantă FP32, este optimizat folosind tehnica **Post-Training Quantization (PTQ)**. Prin implementarea unui algoritm de calibrare bazat pe divergența Kullback-Leibler (KL Divergence), distribuția ponderilor și a activărilor este mapată pe un spațiu de 8 biți (INT8). Această conversie reduce amprenta VRAM a modelului cu până la 75% și maximizează utilizarea unităților Tensor Core, menținând în același timp degradarea metricii de acuratețe (mIoU) la un nivel imperceptibil.

### 4. Paralelism Asincron la Nivel de Sistem (Producer-Consumer)
Execuția secvențială este înlocuită de un pipeline concurent, sincronizat prin **CUDA Streams**. În timp ce procesorul ARM citește pachetul de date geospațiale $N$ de pe mediul de stocare, GPU-ul execută preprocesarea CUDA pentru pachetul $N-1$ și engine-ul TensorRT realizează inferența pe pachetul $N-2$. Rezultatele inferenței (măștile de clasificare) sunt agregate automat de un modul de telemetrie asincron, care exportă statistici zonale în format JSON, sistemul rulând într-un mediu de execuție complet **headless** (fără mediu grafic X11/Wayland) pentru conservarea resurselor.

## Evaluarea Performanței (Metrici Țintă)
Pentru validarea riguroasă a sistemului, performanța va fi evaluată comparativ cu un software de referință (Baseline) executat exclusiv pe CPU, raportând următoarele metrici:
* **Latența de preprocesare ($t_{pre}$)** și **Latența de inferență ($t_{inf}$)**.
* **Timpul total de execuție ($T_{total}$)** (inclusiv operațiile I/O).
* **Speedup-ul global ($S$)**: $S = T_{CPU} / T_{GPU}$.
* Utilizarea resurselor (Memorie VRAM alocată, gradul de utilizare a memoriei partajate, Load-ul procesorului).