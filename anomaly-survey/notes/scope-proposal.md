# Scope proposal (checkpoint 0, 2026-10-09)

Reader: basic deep learning only; knows DINO and CLIP from earlier explainers. Explain from scratch.
Core question: how do you find defects when you have almost no defect images?

## arXiv scan

`notes/arxiv-scan-2026-10-09.json`: 4 paginated queries, 1,165 unique records. Queries: `abs:MVTec` (406),
`abs:"industrial anomaly"` (189), `ti:"anomaly detection" AND (abs:defect OR abs:defects OR
abs:manufacturing OR abs:industrial)` (927), `abs:VisA AND abs:anomaly` (153).

Industrial visual subset (heuristic): cs.CV or image words, AND MVTec/VisA/industrial/defect/
manufacturing/surface/inspection, minus video/surveillance papers that name neither benchmark: 780.
By year: 2018 5, 2019 9, 2020 18, 2021 43, 2022 56, 2023 101, 2024 156, 2025 201, 2026 (to Oct 9) 191.
All counts are lower bounds; arXiv search misses papers that use none of the terms.
Hand-added records: `notes/arxiv-named-and-hand-added-2026-10-09.json`.

## Spine (full read and extraction)

| Family | Paper | arXiv | First posted |
|---|---|---|---|
| Benchmark | MVTec AD (CVPR 2019) | not on arXiv; CVF open access | 2019-06 (CVPR) |
| Benchmark | VisA, in "SPot-the-Difference" (ECCV 2022) | 2207.14315v1 | 2022-07-28 |
| Feature embedding | SPADE | 2005.02357v3 | 2020-05-05 |
| Feature embedding | PaDiM (ICPR 2020 workshop) | 2011.08785v1 | 2020-11-17 |
| Feature embedding | PatchCore (CVPR 2022) | 2106.08265v2 | 2021-06-15 |
| Normalizing flow | FastFlow | 2111.07677v2 | 2021-11-15 |
| Synthetic anomaly | CutPaste (CVPR 2021) | 2104.04015v1 | 2021-04-08 |
| Synthetic anomaly | DRAEM (ICCV 2021) | 2108.07610v2 | 2021-08-17 |
| Student-teacher | EfficientAD (WACV 2024) | 2303.14535v3 | 2023-03-25 |
| Unified | UniAD (NeurIPS 2022) | 2206.03687v3 | 2022-06-08 |
| Zero/few-shot | WinCLIP (CVPR 2023) | 2303.14814v1 | 2023-03-26 |
| Zero/few-shot | AnomalyCLIP (ICLR 2024) | 2310.18961v12 (updated 2026-01-01) | 2023-10-29 |
| Edge speed | PaSTe | 2410.11591v1 | 2024-10-15 |

AnomalyCLIP is at v12; numbers are cited from the version read and may differ from the ICLR camera-ready.
PaSTe is the only edge paper whose abstract says it benchmarks the major methods on edge hardware.
Check at extraction that its table covers PatchCore, PaDiM, a student-teacher method and EfficientAD.
If it does not, ch-speed uses per-paper latencies, each labelled by device.

Optional spine additions (ask): AnomalyDINO 2405.14529v3 and Dinomaly 2405.14325v5 (DINOv2 delta);
MVTec LOCO (IJCV 2022) and MVTec AD 2 2503.21622v1 (harder benchmarks after MVTec AD saturated).

## Branches (abstract plus one key table, one line each on the family tree)

- Precursors: AE-SSIM 1807.02011, Uninformed Students 1911.02357, DN2 2002.10445, Gaussian AD 2005.14140, Patch SVDD 2006.16067
- Feature embedding: CFA 2206.04325, PNI 2211.12634, ReConPatch 2305.16713
- Flows: DifferNet 2008.12577, CFLOW-AD 2107.12571, CS-Flow 2110.02855, MSFlow 2308.15300
- Synthetic: NSA 2109.15222, MemSeg 2205.00908, DSR 2208.01521, SimpleNet 2303.15140, GLASS 2407.09359
- Student-teacher: STPM 2103.04257, Reverse Distillation 2201.10703
- Unified: HVQ-Trans 2310.14228, DiAD 2312.06607, MambaAD 2404.06564, Dinomaly 2405.14325
- Zero/few-shot: APRIL-GAN 2305.17382, AnomalyGPT 2308.15366, MuSc 2401.16753, InCTRL 2403.06495, PromptAD 2404.05231, AnomalyDINO 2405.14529, AdaCLIP 2407.15795, UniVAD 2412.03342
- Benchmarks and tools: MVTec 3D-AD 2112.09045, Real-IAD 2403.12580, MVTec AD 2 2503.21622, Anomalib 2202.08341, ADer 2406.03262, IM-IAD 2301.13359, survey 2301.11514
- Edge: Unified AD on Jetson Xavier NX 2407.02968, PatchCore-Lite/PaDiM-Lite 2603.20288, continual AD on Jetson Orin Nano 2605.24251 (ECCV 2026), EdgeZSAD 2606.16119

## Chapters (13, about 25 min if narrated)

ch-problem comes before ch-benchmark on purpose: a from-scratch reader needs "learn normal, score
distance" before the metrics mean anything. The benchmark is still the first paper-backed chapter.

1. ch-problem: learn "normal", flag distance from it. Toy 2D patch-feature cloud, drag a test point, threshold slider.
2. ch-benchmark: MVTec AD, VisA; image AUROC vs pixel AUROC vs PRO. Toy score map; a threshold sweep draws the ROC.
3. ch-map: computed from the scan: papers per year, keyword family split (heuristic), family tree.
4. ch-spade: pretrained features plus kNN; image level, then pixel level. Architecture walkthrough.
5. ch-padim: one Gaussian per patch position; Mahalanobis distance. 3D bell per patch.
6. ch-patchcore: memory bank, locally aware patches, greedy coreset. 3D embedding cloud, coreset slider.
7. ch-fastflow: invertible map to a Gaussian; likelihood as the score. 3D density warp.
8. ch-synthetic: CutPaste and DRAEM; fake defects make the problem supervised. Perlin mask generator.
9. ch-efficientad: student-teacher plus autoencoder; logical anomalies; its own latency table as the hook.
10. ch-uniad: one model for all classes; the "identical shortcut" and how masking stops it. ADer multi-class comparison.
11. ch-winclip: zero/few-shot with CLIP; prompt ensemble, sliding windows.
12. ch-anomalyclip: object-agnostic prompts learned once; cross-dataset transfer.
13. ch-speed: latency vs image AUROC for every method above; desktop GPU numbers and Jetson numbers, each labelled by device.
Close: scoreboard (image/pixel AUROC on MVTec AD and VisA), when to use which, where to go deeper.

## Scoreboard policy

- Main view: each paper's own reported number, with its table and row.
- Reproduction view for one-class-per-category methods: Anomalib 2202.08341, if its paper tabulates them; else omit this view and say so.
- ADer 2406.03262 benchmarks the multi-class setting, so it appears only in ch-uniad, never beside one-class numbers.
