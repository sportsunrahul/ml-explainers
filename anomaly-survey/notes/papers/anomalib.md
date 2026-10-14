# Anomalib (arXiv 2202.08341v1)

Lens: benchmark / library. Read from `sources/anomalib/src/` (main.tex inputs tex/0-6) and checked
against `paper.pdf` (5 pages) with pdftotext. Job here: does the paper give a shared-protocol MVTec AD
table usable as the scoreboard's reproduction view?

Answer: PARTLY YES. Table 1 gives image AUROC for DFM, STFPM, PaDiM, CFLOW, PatchCore; Table 2 gives
pixel AUROC for STFPM, PaDiM, CFLOW, PatchCore; all on the 15 MVTec AD categories, produced by one tool.
FastFlow, DRAEM, EfficientAD, GANomaly are NOT in the tables (GANomaly's row is commented out in the
LaTeX). No PRO. No VisA. No timing and no hardware in the compiled paper.

## 1. Identity
- "Anomalib: A Deep Learning Library for Anomaly Detection". arXiv 2202.08341, version read v1,
  submitted 2022-02-16 (abs.html). Only v1 exists in the downloaded abs page.
- Authors: Samet Akcay ... Utku Genc (Intel). Venue: not stated (ICASSP-style `spconf.sty` template).
- Repo: https://github.com/openvinotoolkit/anomalib (abstract footnote).

## 2. Core problem
Anomaly detection lacked a shared library to reproduce and compare algorithms on the same data
pipeline, and to deploy them (Sec. 1).

## 3. What prior tools did and why that hurt (per the paper)
Existing AD libraries focus on single algorithms or lack deployment; supervised tasks have
mmdetection/detectron, but unsupervised AD had no equivalent (Sec. 1).

## 4. The one key idea
One modular PyTorch Lightning library: dataset adapters, model components, shared post-processing
(normalization, thresholding), metrics, benchmarking and OpenVINO export, so results come from one pipeline.

## 5. Mechanism (library, as described)
1. Data adapters (MVTec, BTAD, others), albumentations transforms, tiling.
2. Models: density estimation (DFM, PaDiM, CFLOW, PatchCore), reconstruction (GANomaly), distillation (STFPM) (Sec. 3.3).
3. Post-processing: min-max normalization of scores w.r.t. validation set; adaptive threshold by F1 on validation.
4. Metrics via TorchMetrics: AUROC, F1, PRO.
5. Deploy: OpenVINO and NNCF for optimization/quantization; inference from .ckpt or OpenVINO .bin/.xml.
6. `benchmark.py` runs grid searches and logs to TensorBoard, wandb or CSV (Sec. 4).

## 6. Equations
None.

## 7. Measurement setup (what the paper states)
- "image-level and pixel-level AUROC scores benchmarked and averaged on MVTec categories ... obtained
  by using anomalib's benchmarking tool" (Sec. 5).
- Design principle: implementations "aim to reproduce the results reported in the original
  publications" (Sec. 2, Reproducibility).
- NOT stated: backbone per model, input resolution, coreset ratio, epochs, seeds, GPU, batch size.
  The only backbone in the text is a code listing example, `Patchcore(backbone="resnet18", ...)`
  (Sec. 4 listing); it is an API example, not the benchmark config.
- Protocol: our reading is one model per category (the table is per category, then averaged); the
  paper does not say so explicitly. Full-shot (all normal training images), our reading.
- Timing: the compiled paper has none. A throughput table exists only as COMMENTED-OUT LaTeX in
  tex/5.benchmarks.tex (STFPM 6.369 FPS PyTorch on RTX 3090, 12.038 FPS OpenVINO on Core i9-10980XE;
  PADIM 15.103 / 20.720). It is not in the PDF; do not cite it on the page.

## 8. Results worth remembering (full tables, values exactly as printed)

Table 1 (PDF p. 3), "Image Level AUROC Scores for MVTec dataset categories":

| Model | Carpet | Grid | Leather | Tile | Wood | Bottle | Cable | Capsule | Hazelnut | Metalnut | Pill | Screw | Toothbrush | Transistor | Zipper | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DFM | 0.847 | 0.496 | 0.930 | 0.981 | 0.957 | 0.997 | 0.927 | 0.876 | 0.985 | 0.946 | 0.816 | 0.761 | 0.964 | 0.936 | 0.945 | 0.891 |
| STFPM | 0.921 | 0.983 | 0.921 | 0.961 | 0.994 | 0.998 | 0.946 | 0.717 | 0.996 | 0.984 | 0.492 | 0.550 | 0.638 | 0.852 | 0.837 | 0.853 |
| PADIM | 0.945 | 0.857 | 0.982 | 0.950 | 0.976 | 0.994 | 0.843 | 0.901 | 0.750 | 0.961 | 0.863 | 0.759 | 0.889 | 0.920 | 0.780 | 0.891 |
| CFLOW | 0.981 | 0.963 | 1.000 | 0.997 | 0.991 | 1.000 | 0.969 | 0.970 | 1.000 | 0.991 | 0.981 | 0.767 | 0.911 | 0.950 | 0.986 | 0.964 |
| PatchCore | 0.979 | 0.962 | 1.000 | 1.000 | 0.992 | 1.000 | 0.993 | 0.976 | 1.000 | 0.996 | 0.934 | 0.947 | 0.947 | 1.000 | 0.982 | 0.981 |

Table 2 (PDF p. 3), "Pixel Level AUROC Scores for MVTec dataset categories":

| Model | Carpet | Grid | Leather | Tile | Wood | Bottle | Cable | Capsule | Hazelnut | Metalnut | Pill | Screw | Toothbrush | Transistor | Zipper | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| STFPM | 0.973 | 0.987 | 0.976 | 0.964 | 0.958 | 0.968 | 0.940 | 0.957 | 0.979 | 0.965 | 0.867 | 0.938 | 0.148 | 0.759 | 0.983 | 0.891 |
| PADIM | 0.984 | 0.918 | 0.994 | 0.934 | 0.947 | 0.983 | 0.965 | 0.984 | 0.978 | 0.970 | 0.957 | 0.978 | 0.988 | 0.968 | 0.979 | 0.968 |
| CFLOW | 0.985 | 0.969 | 0.995 | 0.966 | 0.926 | 0.984 | 0.965 | 0.988 | 0.988 | 0.982 | 0.987 | 0.982 | 0.985 | 0.930 | 0.980 | 0.974 |
| PatchCore | 0.988 | 0.969 | 0.991 | 0.961 | 0.935 | 0.985 | 0.988 | 0.988 | 0.987 | 0.989 | 0.981 | 0.989 | 0.988 | 0.982 | 0.983 | 0.980 |

Notes on the tables:
- Printed as fractions (0.981), not percentages.
- PaDiM's image mean here (0.891) is below PaDiM's own WR50 number (95.3, per `padim.md`, PaDiM
  Table V). STFPM's image mean is 0.853, and its toothbrush pixel AUROC (0.148) pulls its pixel mean
  to 0.891. STFPM's own paper is not in my sources, so I did not compare it. Our reading: early-library
  reproductions with unstated settings. Present them as "Anomalib v1 reproduction", never as the
  method's own number.
- PaDiM hazelnut image AUROC 0.750 is unusually low (as printed).

## 9. Limitations (stated)
Future work: audio, video, 3D (Sec. 6). The paper points to the repo for "more benchmark results"
(Sec. 5 footnote); those are code-only and not in `sources/`.

## 10. Outline feedback
- The scoreboard policy ("reproduction view ... from Anomalib 2202.08341 only if its paper tabulates
  them") holds for 5 methods (image) and 4 methods (pixel). It cannot supply FastFlow, DRAEM,
  EfficientAD, SPADE, or any VisA or PRO cell.
- The reproduction view will show PaDiM at 0.891 image AUROC, below its own paper's 95.3, and STFPM
  at 0.853. Caption it "library reproduction, 2022, settings not stated in the paper".
- Anomalib gives no latency, so it adds nothing to ch-speed.
