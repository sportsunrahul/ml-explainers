# Edge-device papers for ch-speed (device half)

Lens: benchmark. Papers: 2605.24251v2 (continual AD on Jetson Orin Nano), 2606.16119v1 (EdgeZSAD),
2505.24334v1 (KairosAD), 2407.02968v1 (unified AD on Xavier NX), 2603.20288v1 (PatchCore-Lite /
PaDiM-Lite), 2410.11591v1 (PaSTe). Anomalib 2202.08341v1 is in `anomalib.md`. Claims: `notes/claims/edge.json`.
Read from each `src/` LaTeX; table numbers checked against `paper.pdf` with pdftotext.
"n.s." = not stated in the paper. "our reading" = my inference.

## 0. Every timing record, with the accuracy the same paper reports for the same config

### 0a. Jetson-class GPU

| Source, loc | Method | Backbone | Input | Device | Framework / precision | Batch, averaging | Latency ms | FPS | Size / memory | Accuracy for this exact config |
|---|---|---|---|---|---|---|---|---|---|---|
| 2605.24251v2 Tab. 4 | PatchCore* (replay) | ResNet-18, frozen | 224 | NVIDIA Jetson Orin Nano | n.s. | single image, mean +/- std of 30 runs | 119.0 +/- 2.4 | 8.4 | 9.3 M params; 5.0 MB/task | MVTec AD img AUROC 0.925, pix 0.945; LOCO img 0.743 (CONTINUAL, Tab. 1/3) |
| same | EfficientAD* (replay) | PDN (S/M n.s.), 8.1 M params | 256 | same | n.s. | same | 34.7 +/- 0.1 | 28.8 | 8.1 M params | MVTec img 0.529, pix 0.402 (CONTINUAL; collapsed) |
| same | DINOSaur | DINOv3 ViT-S/16, frozen | 224 (14x14 grid) | same | n.s. | same | 91.6 +/- 9.6 | 10.9 | 21.6 M params; 7.1 MB/task | MVTec img 0.968, pix 0.947; LOCO img 0.768 (CONTINUAL) |
| same | DNE | ViT-B/16, fine-tuned | 224 | same | n.s. | same | 60.7 +/- 4.9 | 16.5 | 86.6 M; 2.3 MB/task | MVTec img 0.500 (CONTINUAL), no pixel map |
| same | UCAD | frozen ViT + SAM | 224 | same | n.s. | same | 956.5 +/- 7.3 | 1.0 | 36.0 M; 1.2 MB/task | MVTec img 0.798, pix 0.822 (CONTINUAL) |
| same | IUF | custom ViT enc-dec | 224 | same | n.s. | same | OOM | --- | 631.1 M | MVTec img 0.508, pix 0.525 (CONTINUAL) |
| 2606.16119v1 Tab. 4 | EdgeZSAD (TinyViT + EdgeGLR) | TinyViT-21M-512 | 512 | Jetson Orin Nano Super, MAXN 25W | TensorRT FP16 | batch n.s.; mean of 100 passes after 20 warm-up | 32.83 | 30.46 | 21.27 M params, 53.77 GFLOPs (backbone only) | Device-rescored (Tab. 5): MVTec img AUROC 91.6 +/- 0.3, pix 79.3, PRO 67.8; VisA 88.2 / 92.9 / 74.2. ZERO-SHOT, source Real-IAD-DR |
| same | "CLIP ViT-L/14 family" backbone-only probe | CLIP ViT-L/14 | 336 | same | TensorRT FP16 | same | 130.26 | 7.68 | 303.51 M, 381.92 GFLOPs | NONE for the timed graph. Row's 92.2 (MVTec) / 84.7 (VisA) is host AUROC of VisualAD (CLIP), copied from hou2026visualad |
| same | "DINOv2 ViT-L/14 family" backbone-only probe | DINOv2 ViT-L/14 | 518 | same | TensorRT FP16 | same | 301.33 | 3.32 | 304.37 M, 1013.61 GFLOPs | NONE for the timed graph. Row's 90.1 / 83.1 is VisualAD (DINOv2), copied |
| 2505.24334v1 Sec. 4.4 | KairosAD | MobileSAM image encoder + MLP | n.s. | "NVIDIA Jetson NX" (as printed) | PyTorch (implementation); deploy precision n.s. | n.s. | 211 | n.s. | n.s. | NONE on device |
| same | KairosAD | same | n.s. | "NVIDIA Jetson AGX" (as printed) | n.s. | n.s. | 218 | n.s. | n.s. | NONE on device |
| 2407.02968v1 Tab. 5 | STFPM | ResNet-18 T and S | 256 | NVIDIA Jetson Xavier NX, 16 GB, 30 W mode | PyTorch FP-32 | batch n.s.; "Avg. Time" | 52.14 | n.s. | 88 MB | "Mean AUROC" 0.94 (MVTec AD, 15-class MULTI-CLASS; image vs pixel n.s.) |
| same | STFPM | same | 256 | same | TensorRT FP-16 | same | 24.98 | n.s. | 52.7 MB | 0.94 |
| same | STFPM | same | 256 | same | TensorRT INT-8 (student only; teacher FP-32) | same | 24.87 | n.s. | 48.5 MB | 0.92 |
| same | RD (reverse distillation) | WideResNet-50 encoder | 256 | same | PyTorch FP-32 | same | 166.60 | n.s. | 645.1 MB | 0.96 |
| same | RD | same | 256 | same | TensorRT FP-16 | same | 19.68 | n.s. | 526.8 MB | 0.82 |
| same | RD | same | 256 | same | TensorRT INT-8 (bottleneck + decoder) | same | 18.02 | n.s. | 313.64 MB | 0.82 |
| same | Uninformed Students (US) | 5-layer CNN teacher + 3 students | 256 | same | PyTorch FP-32 | same | 1392.97 | n.s. | 26.8 MB | 0.79 |
| same | US | same | 256 | same | TensorRT FP-16 | same | 699.24 | n.s. | 22 MB | 0.79 |
| same | US | same | 256 | same | TensorRT INT-8 | same | 591.17 | n.s. | 15.4 MB | 0.57 |

### 0b. Other devices (CPU, mobile SoC, unknown GPU)

| Source, loc | Method | Device | Framework / precision | Latency ms | FPS | Memory | Accuracy for this config |
|---|---|---|---|---|---|---|---|
| 2605.24251v2 Tab. 4 | DNE / IUF / UCAD / PatchCore* / EfficientAD* / DINOSaur | Raspberry Pi 5 (CPU) | n.s.; single image, 30 runs | 560.0 +/- 6.1 / 3668.6 +/- 69.6 / 1058.0 +/- 15.4 / 1074.5 +/- 7.9 / 1318.2 +/- 4.3 / 630.6 +/- 10.2 | 1.8 / 0.3 / 0.9 / 0.9 / 0.8 / 1.6 | as 0a | as 0a (CONTINUAL) |
| 2606.16119v1 Tab. 4 | EdgeZSAD / CLIP-L probe / DINOv2-L probe | Qualcomm RB5 Gen2 | QNN GPU FP16, `disable_node_optimizations=true` | 139.97 / 403.42 / "Resize fail" | 7.14 / 2.48 / --- | as 0a | EdgeZSAD: same as Jetson (Tab. 5, identical cells); probes: none |
| 2505.24334v1 Tab. 1 | KairosAD | n.s. (training GPU is RTX 4090; inference device for the 5 ms n.s.) | PyTorch | 5 | n.s. | 11.53 M params | MVTec I-AUROC 99.10 (SUPERVISED, see Sec. 3) |
| 2505.24334v1 Tab. 1 (copied from li2024sam = STLM paper) | DRAEM / CFLOW / PatchCore / RD4AD / SimpleNet / RD++ / FOD / STLM | n.s. | n.s. | 159 / 37 / 180 / 28 / 39 / 30 / 103 / 20 | n.s. | 97 / 94.70 / 186.55 / 150.64 / 52.88 / 154.87 / 28.83 / 16.56 M params | MVTec I-AUROC 97.60 / 97.50 / 99.20 / 98.70 / 99.57 / 99.44 / 99.20 / 99.05 (copied) |
| 2407.02968v1 Tab. 3 | US / RD / STFPM | Intel Xeon W-2265 CPU | PyTorch FP-32 | 18274 / 1365.4 / 275 | n.s. | 26.8 / 644.2 / 88 MB | 0.79 / 0.96 / 0.94 |
| 2407.02968v1 Tab. 4, 6 | US / RD / STFPM | same CPU | PyTorch PTQ INT-8 (random-normal calib.) | 13968.88 / 1018 / 234.9 | n.s. | 11.8 / 406.42 / 55.12 MB | 0.64 / 0.75 / 0.83 (train-data calib.: 0.63 / 0.50 / 0.75) |
| 2407.02968v1 Tab. 6 | US / RD / STFPM | same CPU | PyTorch QAT INT-8 | 14607.53 / 1001 / 235.2 | n.s. | 11.8 / 406.42 / 55.12 MB | 0.60 / 0.96 / 0.92 |
| 2603.20288v1 Tab. II + I | PatchCore / PatchCore-Edge / PatchCore-Lite | Intel Core i5-9600K CPU @ 3.70GHz | n.s. | 345 / 30 / 130 | n.s. | total 163 / 7.05 / 1.49 MB; peak 286 / 0.02 / 0.03 MB | I-ROC MVTec 0.98 / 0.95 / 0.86; VisA 0.92 / 0.86 / 0.68; P-ROC MVTec 0.97 / 0.96 / 0.94; VisA 0.98 / 0.91 / 0.90 |
| same | PaDiM / PaDiM-Edge / PaDiM-Lite | same | n.s. | 22553 / 35 / 24 | n.s. | total 3895 / 4.70 / 1.08 MB; peak 1.26 / 0.91 / 0.68 MB | I-ROC MVTec 0.96 / 0.94 / 0.85; VisA 0.88 / 0.84 / 0.75; P-ROC MVTec 0.98 / 0.97 / 0.94; VisA 0.94 / 0.97 / 0.93 |
| same | STFPM / STFPM-Edge / PaSTe | same | n.s. | 123 / 13 / 9 | n.s. | total 190 / 5.32 / 5.11 MB; peak 332 / 18.34 / 44.62 MB | I-ROC MVTec 0.95 / 0.90 / 0.92; VisA 0.85 / 0.84 / 0.84; P-ROC MVTec 0.97 / 0.96 / 0.95; VisA 0.95 / 0.97 / 0.97 |
| 2410.11591v1 | (no timing) | none | none | none | none | MACs and MB only | pixel F1 only |

### 0c. Same-device verdict for the chart
- No source times PatchCore, PaDiM, EfficientAD, FastFlow, DRAEM, a zero-shot CLIP method and a
  DINO-based method on one device.
- Best Jetson set: 2605.24251v2 Tab. 4. PatchCore (ResNet-18), EfficientAD, and a DINO method
  (DINOSaur, DINOv3 ViT-S/16) on one Jetson Orin Nano with one latency protocol. But every AUROC there
  is continual-protocol, and EfficientAD* collapsed (0.529). PaDiM, FastFlow, DRAEM and CLIP are absent.
- Zero-shot on Jetson: only 2606.16119v1. EdgeZSAD is the one full zero-shot method timed on a Jetson
  with device-measured AUROC. CLIP-L and DINOv2-L appear only as backbone-only probes; their AUROC is a
  copied host number of a different (full) method.
- Cleanest latency + image AUROC on one device for PatchCore, PaDiM, STFPM (heavy and light
  backbones): 2603.20288v1, but the device is a desktop CPU (i5-9600K), not a Jetson.
- FastFlow: no timing in any of these sources. DRAEM: only a copied latency (159 ms) with no device,
  in 2505.24334v1.
- Device-name caution: 2605.24251v2 says "NVIDIA Jetson Orin Nano" (power mode, RAM, JetPack n.s.);
  2606.16119v1 says "Jetson Orin Nano Super" in MAXN 25W. Our reading: same Orin Nano module family,
  so the two are close but not guaranteed identical settings.

---

## 1. 2605.24251v2 -- Rethinking Continual Anomaly Detection on the Edge

### 1. Identity
- "Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial
  Conditions". arXiv 2605.24251, read v2 (Mon 29 Jun 2026). v1 Fri 22 May 2026 (abs.html
  `citation_date` 2026/05/22).
- Authors: Chad Weatherly, Sen Lin (University of Houston).
- Venue: "Accepted to ECCV 2026" (abs.html comments).
- Repo: https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD (footnote, Sec. 1).

### 2. Core problem
Continual AD (CAD) methods learn product categories or drifting conditions in sequence. Nobody had
compared them head to head, tested gradual drift or logical anomalies, or measured them on edge hardware.

### 3. Prior methods and why that hurt (per the paper)
- DNE, IUF, UCAD were each evaluated alone with different baselines (Sec. 1, gap 3).
- They use large machinery (ViT-B, 631 M-param IUF, SAM in UCAD) with no edge analysis (gap 1).
- Benchmarks use sharp category switches only, and structural anomalies only (gap 2).

### 4. Key idea
A frozen DINOv3 ViT-S/16 plus a per-patch-location coreset memory per task, scored only against a
local neighborhood, needs no weight updates, so it cannot forget, and it is cheap enough for a Jetson.

### 5. Mechanism (DINOSaur)
1. Frozen DINOv3 ViT-S/16 (21.6 M params, 0 trainable), 224x224 input -> 196 patch tokens (14x14) of
   dim 384, plus a 384-d CLS token (App. B, "DINOSaur").
2. Per task, for each of the 196 locations, greedy coreset over that location's D training features;
   M = max(20, floor(D * rho)), rho = 10% (Sec. 4.3, App. B).
3. Task prototype = mean CLS token of the task's training images.
4. Test: route to the nearest prototype (Eq. 10), then score each patch by nearest neighbor within
   a (2r+1)^2 window, r = 3 (7x7) (Eqs. 7-8); image score = max patch score (Eq. 9).
5. Fit = one forward pass per image plus coreset selection; nothing is trained. On-device adaptation to
   a new task "in under 30 seconds" on the Jetson Orin Nano (abstract, Sec. 6).

### 6. Equations (LaTeX verbatim, printed numbers from PDF)
- Eq. (3) `eq:forgetting_per_task`: `f_j = \max_{t \in \{1, \ldots, T-1\}} a_{t,j} - a_{T,j},`
  "the forgetting for task j after training on the final task T". a_{t,j} = performance on task j after
  training on task t; f_j = best earlier score minus final score.
- Eq. (4) `eq:forgetting_measure`: `\text{FM} = \frac{1}{T-1} \sum_{j=1}^{T-1} f_j.` Average forgetting
  over all tasks except the last. Positive = forgetting; zero = none.
- Eq. (7) `eq:neighborhood`: `\mathcal{N}_{i,j}^{r} = \bigcup_{\substack{i' \in [i-r, i+r] \\ j' \in [j-r, j+r]}} \mathcal{C}_{i',j'}.`
  "the neighborhood memory as the union of patch coresets within a radius r". C_{i',j'} = coreset at
  grid cell (i',j').
- Eq. (8) `eq:patch_score`: `s_{i,j} = \min_{\mathbf{c} \in \mathcal{N}_{i,j}^{r}} \| \mathbf{p}_{i,j}^{\text{test}} - \mathbf{c} \|_2,`
  patch anomaly score = L2 distance from test patch feature p to its nearest stored feature c nearby.
- Eq. (9) `eq:image_score`: `S(\mathbf{x}_{\text{test}}) = \max_{(i,j)} \; s_{i,j}.` Image score = max patch score.
- Eq. (10) `eq:task_id`: `t^* = \arg\min_{t} \| \mathbf{z}^{\text{test}} - \boldsymbol{\mu}^{t} \|_2,`
  z = test CLS token, mu^t = task-t mean CLS token; picks which memory bank to use.
- Note: Algorithm comments in the appendix cite "Eq. 4", "Eq. 5", "Eqs. 6--8", "Eq. 9"; these are
  off by one versus the printed numbers above.

### 7. Measurement setup
- Datasets/protocols: MVTec AD as 15 sequential tasks (alphabetical), MVTec LOCO as 5 tasks, MTD with
  10 drift tasks for color, blur, geometric (Sec. 3, App. C).
- Metric: image AUROC averaged across all tasks after training on the last task (primary); pixel
  AUROC; accuracy and recall at 97.5th-percentile threshold; FM (Sec. 3.4).
- Training: one RTX 4090; Adam lr 7.5e-4, wd 1e-4, batch 12; 224x224 (EfficientAD 256x256); 300
  epochs/task for gradient methods; replay of 100 normal images per previous task for PatchCore* and
  EfficientAD* (App. B).
- PatchCore* config (App. B): ResNet-18 (9.33 M params), intermediate layers concatenated, random
  projection to 512-d, avg-pool k3 s1 p1, 28x28 feature map, greedy coreset 5% (min 500), score
  reweighted by 3 nearest neighbors. Paper notes it is mechanistically equivalent to PatchcoreCL.
- EfficientAD* config (App. B): PDN student + autoencoder, teacher pretrained 60,000 iterations,
  batch 16, ResNet34 features at 512x512 for teacher pretraining; S vs M variant not stated.
- Edge timing (Tab. 4 caption): NVIDIA Jetson Orin Nano and Raspberry Pi 5; "mean +/- std over 30
  single-image runs". Framework, precision (FP32/FP16/TensorRT), power mode, warm-up: not stated.
- Sec. 3.4 promises "peak memory usage", but Tab. 4 reports only Params and Storage/Task.
- Non-continual numbers: none. There is no single-task, joint or upper-bound row for any method
  (grep for joint/upper bound/offline/single-task found nothing).

### 8. Results worth remembering
- Tab. 4 (Jetson Orin Nano): EfficientAD* 34.7 ms (28.8 FPS) fastest; DNE 60.7; DINOSaur 91.6 (10.9
  FPS); PatchCore* 119.0 (8.4 FPS); UCAD 956.5 (1.0 FPS); IUF OOM.
- Tab. 4 (Raspberry Pi 5): DNE 560.0 fastest; DINOSaur 630.6; UCAD 1058.0; PatchCore* 1074.5;
  EfficientAD* 1318.2; IUF 3668.6.
- Text: "DINOSaur is 23% faster than PatchCore on the Jetson (91.6 vs. 119 ms)" (Sec. 5.2).
- Tab. 1 MVTec AD (continual) image AUROC: DNE 0.500, IUF 0.508, UCAD 0.798, PatchCore* 0.925,
  EfficientAD* 0.529, DINOSaur 0.968. FM: PatchCore* 0.023, EfficientAD* 0.051, DINOSaur 0.000.
- Tab. 1 MVTec LOCO image AUROC: 0.500 / 0.572 / 0.629 / 0.743 / 0.485 / 0.768 (same order).
- Tab. 3 pixel AUROC MVTec: IUF 0.525, UCAD 0.822, PatchCore* 0.945, EfficientAD* 0.402, DINOSaur 0.947.
- Tab. 2 MTD: geometric drift puts all methods near chance (0.480-0.525).
- Tab. 5: avg AUROC over five protocols for rho=10%, r=3 is 0.801; rho=20%, r=4 is 0.822.
- Tab. 7: DINOSaur storage MVTec AD total 106.8 MB (15 tasks).

### 9. Limitations (stated)
Geometric drift defeats all methods; neighborhood scoring is local, so logical anomalies need global
reasoning; no streaming without task boundaries (Sec. 6, Future Work). Not stated but relevant:
precision and framework of the Jetson runs.

### 10. Outline feedback
- The outline calls the Orin Nano toggle "(2605.24251v2 table)". Its AUROC axis is continual-protocol
  (sequential tasks, replay). It is not comparable with the desktop EfficientAD view. Badge it, or plot
  latency only and keep AUROC as a tooltip with the protocol.
- EfficientAD* at 0.529 is a continual-learning collapse, not EfficientAD's normal accuracy. A point
  at (34.7 ms, 0.529) would mislead without that label.
- This paper's PatchCore is ResNet-18 at 224 px, not the WRN-50 PatchCore of ch-patchcore.
- DINOSaur is a DINO-based nearest-neighbor method on a Jetson; it fits ch-dinov2 as a branch.

---

## 2. 2606.16119v1 -- EdgeZSAD

### 1. Identity
- "EdgeZSAD: Practical Zero-Shot Anomaly Detection on Edge Devices". arXiv 2606.16119v1, 2026-06-15.
- Authors: Taewan Cho, Andrew Jaeyong Choi (Gachon University; Plaid Labs Inc.).
- Venue: not stated on abs.html; LaTeX uses the WACV template with `\confYear{2027}` (arXiv mode).
- Repo: not given in the LaTeX.

### 2. Core problem
Zero-shot AD methods use ViT-L backbones (about 300 M params) that exceed embedded memory and operator
budgets. How much zero-shot accuracy survives a compact, exportable graph on real edge runtimes?

### 3. Prior methods and why that hurt
CLIP-L methods (WinCLIP, APRIL-GAN, CLIP-AD, AnomalyCLIP, AdaCLIP) and DINOv2-L methods (VisualAD,
UniADet) define host accuracy but do not target embedded budgets (Sec. 2.1). Compact AD lines
(EfficientAD, PatchCore, DRAEM, GLASS) need target-domain normal data, so they are not zero-shot (Sec. 2.2).

### 4. Key idea
Train one small supervised model (TinyViT-21M-512 + a global/local scoring head) on a reshaped
Real-IAD source set with synthetic defects and edge corruptions, then apply it unchanged to unseen
target datasets.

### 5. Mechanism
1. TinyViT-21M-512, four stages with widths (96, 192, 384, 576); only stage-4 map F_4 is used (Sec. 3.1).
2. GRB: GAP + linear -> s_g (Eq. 1).
3. SCB: depthwise 3x3 + GELU + 1x1 conv, bilinear upsample -> dense map P(x) (Eq. 2); top-k mean of
   sigma(P), k ratio 0.01 -> s_l (Eq. 3).
4. Fuse p(x) = (1-lambda) s_g + lambda s_l, lambda = 0.50 (Eq. 4).
5. Train on Real-IAD-DR: 216,917 images, 30 categories (136,205 original, 79,688 edge-corruption,
   1,024 local-defect) (Sec. 3.3). Loss: image BCE (weight 1) + mask BCE (weight 2) (Eq. 5).
   AdamW lr 1e-4, wd 1e-4, cosine, batch 32, 2000 steps, EMA 0.999, 512x512, fp16 mixed precision.
   Whole network trains; it is a supervised model on source data ("zero-shot" = disjoint target classes).

### 6. Equations (verbatim, printed numbers)
- (1) `s_g(x) = \sigma\!\left(h\!\left(\operatorname{GAP}(F_4)\right)\right),` "global image probability";
  h = linear head, GAP = global average pooling, F_4 = last-stage feature map.
- (2) `P(x) = m(F_4),` m = depthwise-separable conv predictor + bilinear upsampling; P = dense map.
- (3) `s_l(x)=\operatorname{TopKMean}\!\left(\sigma(P(x))\right).` Mean of the top 1% pixel probabilities.
- (4) `p(x)=(1-\lambda)\,s_g(x)+\lambda\,s_l(x),` "fuses the global and sparse local scores"; lambda = 0.50.
- (5) `\mathcal{L} = \lambda_{\text{img}}\mathcal{L}_{\text{img}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}},`
  BCE on fused image score plus BCE on dense map; lambda_img = 1, lambda_mask = 2.

### 7. Measurement setup
- Targets: MVTec AD, VisA, BTAD, KSDD2, DAGM, DTD-Synthetic (full splits), following VisualAD.
- Metrics: image AUROC, F1-max, AP; pixel AUROC, PRO. Main model: mean +/- std over 3 runs.
- Deployment: TensorRT FP16 on Jetson Orin Nano Super (MAXN 25W); QNN GPU FP16 on RB5 Gen2 (Sec. 4.1).
- Runtime: "mean of 100 inference passes after 20 warm-up iterations" (Tab. 4 caption). Batch n.s.
- Params/FLOPs are for the vision backbone only. "Large-backbone edge values are backbone-only
  feasibility probes, not full-method runtime" (Tab. 4 caption). Our reading: the TinyViT row is the
  full deployed graph (backbone + EdgeGLR), since Sec. 3 says only that graph runs at inference.
- Tab. 1 baselines are "taken from" hou2026visualad (VisualAD); not re-run.

### 8. Results worth remembering
- Tab. 4: TinyViT 32.83 ms / 30.46 FPS (Jetson), 139.97 / 7.14 (RB5); CLIP ViT-L/14 probe 130.26 /
  7.68 and 403.42 / 2.48; DINOv2 ViT-L/14 probe 301.33 / 3.32 and "Resize fail" on RB5.
- Tab. 1 / Tab. 5 EdgeZSAD (3 runs): MVTec 91.6 +/- 0.3 / 79.3 +/- 0.6 / 67.8 +/- 2.2; VisA
  88.2 +/- 1.0 / 92.9 +/- 0.6 / 74.2 +/- 2.8 (img AUROC / pix AUROC / PRO). Tab. 5 Jetson and RB5
  cells equal the host cells; "Max host-vs-device AUROC drift remains below 0.2 points".
- Tab. 1 copied baselines (from VisualAD), MVTec img/pix/PRO: WinCLIP 90.4/82.3/62.0; AnomalyCLIP
  91.6/91.0/81.7; VisualAD (CLIP) 92.2/90.8/87.5; VisualAD (DINOv2) 90.1/91.3/88.6. VisA: WinCLIP
  75.6/73.2/51.1; AnomalyCLIP 81.0/95.4/86.4; VisualAD (CLIP) 84.7/95.8/91.0; VisualAD (DINOv2)
  83.1/95.3/88.2. These differ from WinCLIP's and AnomalyCLIP's own papers; use those papers for the
  scoreboard and these only as "as re-reported by VisualAD".
- Tab. 3: source corpus Real-IAD -> Real-IAD-DR raises MVTec img AUROC 89.56 -> 92.43 (single seed).

### 9. Limitations (stated, Sec. 5.2)
Localization trails (MVTec PRO 67.8 vs 88.6 best); DAGM gap (91.2 vs 99.5); deployment evidence is
FP16 only; several ablations single-seed. Future: INT8 NPUs (Hailo-8, DeepX DX-M1).

### 10. Outline feedback
- The outline lists EdgeZSAD under "other Jetson numbers". It is on the same Orin Nano family as
  2605.24251v2 (Super variant, MAXN 25W, TensorRT FP16). It can share the Orin toggle with a caveat.
- Do not plot "zero-shot CLIP on Jetson" at (130.26 ms, 92.2): the latency is backbone-only and the
  AUROC belongs to the full VisualAD method on a host GPU. If plotted, mark both as such.
- EdgeZSAD is supervised training on a source dataset; ch-winclip/ch-anomalyclip should say
  "zero-shot" means unseen target classes, which this paper states explicitly (Sec. 4.1).

---

## 3. 2505.24334v1 -- KairosAD

### 1. Identity
- "KairosAD: A SAM-Based Model for Industrial Anomaly Detection on Embedded Devices". arXiv 2505.24334v1, 2025-05-30.
- Authors: Uzair Khan, Franco Fummi, Luigi Capogrosso (University of Verona).
- Venue: "Accepted at the 23rd International Conference on Image Analysis and Processing (ICIAP 2025)".
- Repo: https://github.com/intelligolabs/KairosAD.

### 2. Core problem
SMEs run inspection on low-cost embedded devices; most AD models are too large for them.

### 3. Prior methods and why that hurt
Teacher-student, reconstruction and memory-bank models are large and computationally heavy (Sec. 1).

### 4. Key idea
Use the MobileSAM image encoder as is, plus a small MLP, trained as a SUPERVISED image classifier.

### 5. Mechanism
1. MobileSAM encoder f_theta (5.78 M params, distilled from SAM ViT-H) maps image I to embedding E
   (Eq. 1). The paper prints "d = 3" for the embedding dimension (as printed; looks odd).
2. N_l linear layers + ReLU g_phi -> score s (Eq. 2). N_l = 2 for MVTec, 3 for VisA.
3. Train with weighted BCE (Eq. 3) using normal AND anomalous labels; positive weight = neg/pos ratio.
   Only phi is described as trained; MobileSAM encoder "as it is".
4. Adam lr 1e-2, 35 epochs (MVTec) / 50 (VisA), RTX 4090, PyTorch. Train/test split for anomalous
   images: not described.

### 6. Equations (verbatim)
- (1) `E=f_{\theta{}}(I)\;,` encoder maps image to embedding.
- (2) `s = g_{\phi}(E)\;,` MLP maps embedding to anomaly score in [0,1].
- (3) `\mathcal{L}_{\text{WBCE}} = - \frac{1}{N} \sum_{i=1}^{N} w_i \left[ y_{i} \log \sigma(s_{i}) + (1 - y_{i}) \log (1 - \sigma(s_{i})) \right]\;,`
  y_i = 0 normal / 1 anomalous; w_i = weight on the positive class; sigma = sigmoid.

### 7. Measurement setup
- Metric: image AUROC only. Latency in ms, params in M. Inference device for the 5 ms: not stated.
- Competitor rows: "Competitors' results are from" li2024sam (STLM, ACM TOMM 2025) for MVTec (Tab. 1)
  and yao2024glad (GLAD, ECCV 2024) for VisA (Tab. 2). Not re-run. Their latency hardware: not stated here.
- Jetson: "NVIDIA Jetson NX" 211 ms, "NVIDIA Jetson AGX" 218 ms (Sec. 4.4). No Xavier/Orin qualifier,
  no precision, no batch, no on-device AUROC.

### 8. Results worth remembering
- Tab. 1 MVTec I-AUROC mean / latency ms / params M: DRAEM 97.60 / 159 / 97; CFLOW 97.50 / 37 / 94.70;
  PatchCore 99.20 / 180 / 186.55; RD4AD 98.70 / 28 / 150.64; SimpleNet 99.57 / 39 / 52.88; RD++ 99.44 /
  30 / 154.87; FOD 99.20 / 103 / 28.83; STLM 99.05 / 20 / 16.56; KairosAD 99.10 / 5 / 11.53.
- Tab. 2 VisA I-AUROC: DRAEM 92.4, PatchCore 91.0, RD4AD 96.9, SimpleNet 96.2, KairosAD 92.3 (10.55 M params).

### 9. Inconsistencies (record, do not reconcile)
- Tab. 2 latency/params row looks column-shifted vs Tab. 1: PatchCore 37 / 94.70 (= CFLOW in Tab. 1),
  RD4AD 180 / 186.55 (= PatchCore in Tab. 1).
- KairosAD Hazelnut is 99.17 in the top half of Tab. 1 and 100 in the bottom half. The mean 99.10
  matches the 100 value (our arithmetic: with 99.17 the mean is 99.04).
- Text ratios: "6x reduction" vs RD++ (154.87 / 11.53 is about 13.4); teaser "35% fewer" vs STLM
  (about 30%). Quote raw numbers, not ratios. "78% fewer than SimpleNet" checks out.
- Jetson AGX (218 ms) is slower than Jetson NX (211 ms) as printed.

### 10. Limitations and outline feedback
- Stated: VisA "Capsules" class is weak (69.7).
- Outline asks "AnomalyDINO?": KairosAD does NOT compare with AnomalyDINO. It compares DRAEM, CFLOW,
  PatchCore, RD4AD, SimpleNet, RD++, FOD, STLM.
- KairosAD is supervised (uses anomalous training labels; split undescribed). Its 99.10 must not
  sit on the one-class scoreboard without a "supervised" tag.
- Its Jetson points (211, 218 ms) have no paired AUROC; on a latency-vs-AUROC chart they need the
  host AUROC with a "not measured on device" note, or should be listed, not plotted.
- Its baseline latencies (DRAEM 159, PatchCore 180) have no device; do not plot them on a Jetson toggle.

---

## 4. 2407.02968v1 -- Unified AD on Jetson Xavier NX with distillation and quantization

### 1. Identity
- "Unified Anomaly Detection methods on Edge Device using Knowledge Distillation and Quantization".
  arXiv 2407.02968v1, 2024-07-03. Comments: "20 pages".
- Authors: Sushovan Jena ... Arnav Bhavsar (IIT Mandi; Hitachi India R&D). Venue: not stated (elsarticle preprint).
- Repo: none given; uses official RD, STFPM repos and a third-party US implementation.

### 2. Core problem
One-class models need one model per product. Can one multi-class model run on an edge GPU, and how
much do PTQ and QAT at FP16/INT8 cost in AUROC?

### 3. Prior methods and why that hurt
One-class models multiply deployment cost by class count. UniAD and similar multi-class models use
attention or diffusion, which the paper judges too slow for edge use (Sec. 2.1).

### 4. Key idea
Train light knowledge-distillation detectors (US, RD, STFPM) on all 15 MVTec classes at once, then
quantize only the student side and deploy with TensorRT.

### 5. Mechanism
1. Train 15-class models at 256x256 (US batch 1, 150 ep; RD batch 16, 200 ep; STFPM batch 32, 398 ep; Tab. 1).
2. Quantize only the student (US: three students; RD: bottleneck + decoder, but ConvTranspose2d stays
   FP-32 in Torch); teacher stays FP-32.
3. PTQ calibration on training data vs on random normal data; TensorRT PTQ uses the random-normal calibration.
4. QAT: fine-tune a few epochs with fake-quantize modules (PyTorch, CPU only).

### 6. Equations
- Inline (not numbered): `x_{q}=$ round $((1 / \mathrm{s}) \cdot \mathrm{x}+\mathrm{z})` (Sec. 3.4.1,
  as printed with math-mode breaks). x = float value, s = scale factor, z = zero point, x_q = b-bit integer.

### 7. Measurement setup
- Dataset: MVTec AD. Protocol: 15-class (multi-class, one model for all classes), 256x256.
- Metric: "Mean AUROC" (mean over classes). Image-level vs pixel-level: NOT stated.
- Devices: NVIDIA Jetson Xavier NX (384-core Volta GPU, 16 GB RAM, 30 W mode); Intel Xeon W-2265 CPU.
- Inference batch size and timing method: not stated ("Avg. Inference Time").

### 8. Results worth remembering
- Tab. 2 one-class vs 15-class mean AUROC: US 0.76 / 0.79; RD 0.98 / 0.95; STFPM 0.97 / 0.94.
- Tab. 5 (Jetson): see section 0a. STFPM TRT INT-8 24.87 ms at 0.92; RD TRT INT-8 18.02 ms at 0.82.
- Tab. 6 (CPU): QAT INT-8 recovers RD to 0.96 and STFPM to 0.92; PTQ random-normal gives 0.75 / 0.83.

### 9. Inconsistencies (record, do not reconcile)
- RD 15-class AUROC: 0.95 in Tab. 2, 0.96 in Tabs. 3, 5, 7.
- RD Jetson FP-32 time: 166.60 in Tabs. 3 and 5, "66.60" in Tab. 7.
- US PTQ INT-8 model size: 11.8 in Tabs. 4 and 6, 10.4 in Tab. 7. RD FP-32 size 644.2 (Tab. 3) vs 645.1 (Tab. 5).
- Text "26 times and 73 times" compares CPU FP-32 to Jetson TRT, not Jetson FP-32 to TRT.
  "510 times lower" is a typo as printed. Sec. 4.2 says "256-core GPU"; Sec. 4.1.1 says 384-core.
- Stated limitation: PyTorch quantized models cannot run on CUDA, so QAT/PTQ Torch results are CPU only.

### 10. Outline feedback
- This is a multi-class protocol, so its points fit ch-uniad context better than the one-class
  scoreboard. On the speed chart, label them "multi-class, AUROC level n.s.".
- Good illustration that TensorRT FP16 halves STFPM latency (52.14 -> 24.98 ms) at the same 0.94,
  while RD FP16 drops to 0.82.

---

## 5. 2603.20288v1 -- PatchCore-Lite and PaDiM-Lite

### 1. Identity
- "Efficient Visual Anomaly Detection at the Edge: Enabling Real-Time Industrial Inspection on
  Resource-Constrained Devices". arXiv 2603.20288v1, 2026-03-18.
- Authors: Arianna Stropeni, Fabrizio Genilotti (equal), ... Gian Antonio Susto (University of Padua).
- Venue: not stated (IEEEtran journal template). Code: in the MoViAD library (arXiv 2507.12049).

### 2. Core problem
PaDiM's covariance matrices and PatchCore's memory bank are too heavy for edge memory and compute,
even after the PaSTe-style switch to MobileNetV2.

### 3. Prior methods and why that hurt
PaSTe made only the teacher-student family edge-ready; memory-bank methods were not adapted (Sec. 2.2).

### 4. Key idea
PaDiM-Lite: diagonal covariance, so Mahalanobis becomes an element-wise sum. PatchCore-Lite:
product-quantize the coreset, search coarse in code space, then exactly on a small decoded subset.

### 5. Mechanism
1. Backbone MobileNetV2 (ImageNet). PaDiM-Edge/-Lite use layers 7, 10, 13 (224-d concatenated).
   PatchCore-Edge/-Lite use layers 4, 7, 10. STFPM-Edge and PaSTe use layers 3, 8, 14 (Sec. 4.2).
2. PaDiM-Lite: per position store mean and variance vector (+ eps = 0.01); score Eq. (5).
3. PatchCore-Lite: standard coreset, then PQ with m = 8 subspaces, b = 8 bits (256 centroids).
   Coarse search returns k candidates ("typically k = 500 to 1000 out of K = 10000"); decode only those;
   exact L2 min. Image score = max patch score (no PatchCore reweighting).
4. Nothing is trained by gradient (fit = statistics, coreset, k-means codebooks).

### 6. Equations (verbatim, printed numbers)
- (1) `M(\mathbf{x}_{ij}) = (\mathbf{x}_{ij} - \boldsymbol{\mu}_{ij})^T \boldsymbol{\Sigma}_{ij}^{-1} (\mathbf{x}_{ij} - \boldsymbol{\mu}_{ij})`
  PaDiM Mahalanobis score at patch position (i,j).
- (5) `M(\mathbf{x}_{ij}) = \sum_{k=1}^{d} \frac{(x_{ij,k} - \mu_{ij,k})^2}{\sigma_{ij,k}^2}` "The Mahalanobis
  distance simplifies to" this under the diagonal assumption; k runs over the d feature dimensions.
- (6) `s(\mathbf{x}) = \min_{\mathbf{m} \in \mathcal{M}} \|\mathbf{x} - \mathbf{m}\|_2,` PatchCore patch score.
- (10) `\hat{d}(\mathbf{x}, \mathbf{m}_i) = \sqrt{\sum_{j=1}^{m} \|\mathbf{\tilde{x}_{j}} - \mathbf{c}_j^{q_j(\mathbf{m}_i)}\|_2^2}`
  "asymmetric distance" in code space; c_j^{q} = centroid of subspace j assigned to memory vector m_i.
  Our reading: as printed, x~_j is the quantized test sub-vector, which would make it symmetric.
- (12) `s(\mathbf{x}) = \min_{\tilde{\mathbf{m}}_i \in \text{candidates}} \|\mathbf{x} - \tilde{\mathbf{m}}_i\|_2`
  final score over the decoded candidates.

### 7. Measurement setup
- Datasets: MVTec AD, VisA. Metrics: I-ROC (image AUROC), P-ROC (pixel AUROC), memory, inference time,
  peak inference memory.
- Device: "Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz to simulate edge deployment conditions" (Sec. 4.3).
  Not a Jetson. Framework, precision, input resolution, dataset used for timing: not stated.
  Inference time = "average time required to process a single test image".
- Baseline rows (PatchCore, PaDiM, STFPM heavy; -Edge; PaSTe) cite barusco2025paste (PaSTe, CVPRW
  2025). Whether re-run or copied is not stated. The PaSTe arXiv v1 we have reports no AUROC and no ms,
  so these numbers are at least not from that version. Heavy backbone not named (95 MB in Tab. II).
- Protocol: not stated; our reading is one model per category (PaSTe/MoViAD line).

### 8. Results worth remembering
- See section 0b. PaDiM 22553 ms -> PaDiM-Edge 35 -> PaDiM-Lite 24 ms on the CPU.
- PatchCore 345 -> PatchCore-Edge 30 -> PatchCore-Lite 130 ms (Lite is 4x slower than Edge, smaller memory).
- Image AUROC cost: PatchCore-Lite 0.86 vs Edge 0.95 (MVTec); VisA 0.68 vs 0.86.
- Abstract: PatchCore-Lite "79% reduction in total memory footprint"; PaDiM-Lite "77% reduction in
  total memory and a 31% decrease in inference time" (both vs the -Edge versions; checks out).

### 9. Limitations / oddities
- Performance "highly sensitive" to k (Sec. 5.2). Peak inference memory for PatchCore-Edge 0.02 MB and
  PatchCore-Lite 0.03 MB vs PatchCore 286 MB looks implausibly small; copy as printed.

### 10. Outline feedback
- The outline lists PatchCore-Lite as a branch of the Jetson view. Its device is a desktop CPU. Put it
  on a "CPU" toggle or a footnote, not the Jetson toggle.
- It is the only source here with PatchCore AND PaDiM AND STFPM latency + image AUROC on one device,
  on both MVTec and VisA.

---

## 6. 2410.11591v1 -- PaSTe

### 1. Identity
- "PaSTe: Improving the Efficiency of Visual Anomaly Detection at the Edge". arXiv 2410.11591v1, 2024-10-15.
  Comments "13 pages, 6 figures". Authors: Manuel Barusco ... Gian Antonio Susto (Univ. Padova; FBK).
- Venue: not on abs.html. Later papers cite it as CVPR 2025 Workshops (VAND), pp. 4026-4035
  (2603.20288v1 bib; 2605.24251v2 bib). Secondary source.
- Supplementary: https://bitbucket.org/tinyad24/paste-sup/src/main/ (not in sources).

### 2. Core problem
VAD literature ignores memory, compute and latency on edge devices.

### 3. Prior methods and why that hurt
WideResNet50 feature extractors are heavy; memory banks grow with feature-map size; STFPM stores and
backpropagates two full networks (Sec. 3).

### 4. Key idea
Swap the heavy backbone for an edge backbone; for STFPM, share the first teacher layers with the
student (frozen) so they are stored and computed once (PaSTe).

### 5. Mechanism
1. Backbones: MobileNetV2, MCUNet-in3, PhiNet, MicroNet-m1, all ImageNet-pretrained, trimmed after the
   last used layer.
2. Methods: PatchCore, PaDiM, CFA, STFPM with default hyperparameters; layer groups in Tab. 1.
3. PaSTe on MobileNetV2: layers [7, 10, 14]; layers up to 6 shared and frozen.

### 6. Equations
None in the paper.

### 7. Measurement setup
- Dataset: MVTec AD only. Metric: pixel-level F1 (main). No image AUROC. No pixel AUROC.
- Cost: total memory (MB) and inference MACs. NO wall-clock latency and NO hardware device.
  Fig. 1 caption says "x-axis the inference time", but the plotted x-axis is "Size (MB)".
- The abstract's "decreases the inference time by 25%" is the MAC reduction (454.4M -> 341.2M, 24.9%).

### 8. Results worth remembering (Tab. 3, Tab. 2)
- WideResNet50: PatchCore 300 MB, 10,42G MAC (comma as printed), F1 0.57; PaDiM 3.72G MB, 101,44G, 0.57;
  CFA 141, 36,89G, 0.60; STFPM 189.7, 18.3G, 0.51.
- MobileNetV2: PatchCore 31.11 MB, 235.6M, 0.53; PaDiM 49.4, 451M, 0.49; CFA 6.16, 2.8G, 0.55;
  STFPM 5.32, 454.4M, 0.52; PaSTe 5.11, 341.2M, 0.53.
- Tab. 2: training MAC 297.5M -> 198.4M; training RAM 96.15 -> 22.9 MB (76.2%).

### 9. Inconsistencies
- Text: PaDiM memory "from 3.72 GB to just 31.1 MB"; Tab. 3 gives 49.4 MB (31.11 is PatchCore). The
  text's "75.4-fold" matches 49.4 MB.
- Text: STFPM "4-fold reduction in inference", PatchCore "4.4 times"; Tab. 3 gives about 40x and 44x.

### 10. Outline feedback
- PaSTe cannot be a point on a latency-vs-AUROC chart: no ms and no AUROC in v1. Use it in prose
  ("swap the backbone") or plot memory vs pixel F1. For ms + AUROC of PaSTe, use 2603.20288v1 (CPU).
