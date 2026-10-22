# Roadmap și Etape de Dezvoltare: TerraSentinel

Dezvoltarea proiectului adoptă o metodologie iterativă, structurată pe evaluări de performanță (profiling) la fiecare etapă, pentru a valida atingerea parametrilor optimi ceruți de standardele academice de excelență (Nota 10 SSC).

## M1. Etalonul de Referință (Baseline) și Mediu (Săptămânile 1-2)
* **Configurarea mediului de execuție:** Instalarea și configurarea toolchain-ului nativ pe NVIDIA Jetson (CUDA Toolkit, cuDNN, TensorRT SDK, PyTorch).
* **Implementarea Baseline-ului (CPU):** Dezvoltarea pipeline-ului complet în Python/C++ rulând exclusiv pe arhitectura ARM. Aceasta include citirea benzilor Sentinel-2, calculul secvențial al NDVI și executarea inferenței PyTorch (FP32).
* **Colectarea metricilor de referință:** Profilarea latenței end-to-end, a numărului de cadre procesate pe secundă (throughput) și a consumului de memorie pentru implementarea non-accelerată.

## M2. Optimizarea Subsistemului de Memorie și CUDA (Săptămânile 3-4)
* **Integrarea Zero-Copy:** Trecerea de la memoria alocată dinamic (paginabilă) la memorie *Page-Locked (Pinned)* (`cudaHostAlloc`) pentru a anula latența transferurilor explicite pe magistrală.
* **Dezvoltarea kernelului CUDA:** Implementarea kernelului unificat (Operator Fusion) în C/C++ pentru preprocesare (NDVI + Normalizare + Transpunere $HWC \rightarrow CHW$).
* **Validare:** Utilizarea **NVIDIA Nsight Compute** pentru a analiza gradul de ocupare a memoriei partajate (Shared Memory Occupancy) și verificarea echivalenței matematice a matricelor de ieșire cu cele din referința CPU.

## M3. Optimizarea Inferenței AI - INT8 PTQ (Săptămânile 5-6)
* **Tranziția modelului:** Exportul rețelei neuronale din mediul de antrenare (PyTorch) în format intermediar ONNX.
* **Calibrarea INT8:** Dezvoltarea clasei `IInt8EntropyCalibrator2` în C++ (sau echivalentul în Python TensorRT API). Rularea unui subset reprezentativ de date multispectrale pentru calculul scalelor de activare (Post-Training Quantization).
* **Compilarea Engine-ului:** Generarea și serializarea planului de execuție TensorRT optimizat specific pentru microarhitectura GPU-ului Jetson curent. Evaluarea pierderii de precizie (mIoU) comparativ cu FP32.

## M4. Pipelining Asincron și Integrare Edge (Săptămânile 7-8)
* **Arhitectura Concurentă:** Implementarea modelului multi-threading de tip Producer-Consumer în C++. Maparea operațiilor I/O pe CPU și a operațiilor de preprocesare/inferență pe fluxuri asincrone (`cudaStream_t`).
* **Telemetrie Edge-to-Cloud:** Crearea modulului de analiză a măștii de ieșire pentru extragerea matematică a procentelor zonale afectate de secetă, cu serializare directă în format JSON.
* **Configurarea Headless:** Dezactivarea managerului de display pentru izolarea resurselor sistemului de operare, alocând puterea de calcul strict aplicației C++.

## M5. Validare Finală și Documentație Academică (Săptămânile 9-10)
* **Profiling Global:** Analiza fluxului asincron folosind **NVIDIA Nsight Systems** pentru a demonstra ascunderea latențelor (latency hiding) și lipsa blocajelor (bottlenecks) în pipeline.
* **Calculul Speedup-ului:** Agregarea datelor cantitative și calculul factorului de accelerare ($S = T_{CPU} / T_{GPU}$).
* **Redactarea Raportului Tehnic:** Documentarea arhitecturii sistemului, a testelor efectuate, includerea graficelor comparative și pregătirea sistemului pentru demonstrația live.