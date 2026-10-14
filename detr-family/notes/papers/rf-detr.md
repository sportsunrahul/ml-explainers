# RF-DETR: Neural Architecture Search for Real-Time Detection Transformers

arXiv 2511.09554v2 (v2 posted 2026-02-03; v1 2025-11-12). Robinson, Robicheaux, Popov (Roboflow), Ramanan, Peri (CMU). ICLR 2026 template.
Sources read: `tex/iclr2026_conference.tex`, `tex/supplement.tex` (appendix). Table numbers verified against `paper.pdf` (pdftotext). Claim ids refer to `notes/claims/rf-detr.json`. Row-level ids follow `rfdetr-t<N>-<table>-<model><size>`; each carries the full row tuple.

Official repo: github.com/roboflow/rf-detr, README on branch `develop`, commit 990f5d9 (README last changed 2026-09-15). README numbers are tagged `"paper": "RF-DETR README"` and are kept separate below.

## Delta from predecessors (the problem)

- Lineage: Deformable DETR (deformable cross-attention) -> RT-DETR (real-time DETR; decoder layers prunable at test time because each is supervised) -> LW-DETR (plain ViT backbone, CAEv2, windowed + global attention interleave, Objects-365 pretraining) -> RF-DETR. The paper says RF-DETR "modernizes LW-DETR" (Sec 3).
- The problem the paper names: specialist real-time detectors (YOLOv8/11, D-FINE, RT-DETR) are implicitly overfit to COCO through bespoke architectures, learning-rate schedules and augmentation schedules, so they transfer poorly to real-world datasets with different statistics (objects per image, number of classes, dataset size). Fine-tuned open-vocabulary VLMs (GroundingDINO, LLMDet) transfer well but are slow (309.9 ms on RF100-VL, `rfdetr-t4-rf100-groundingdinot`).
- A second problem: each hardware target / latency budget normally needs a separately designed and trained model size (D-FINE N/S/M/L/X are hand-designed).
- Concrete changes vs LW-DETR:
  1. Backbone CAEv2 (10 layers, patch 16) -> DINOv2 (12 layers) with pretrained weights (`rfdetr-caev2-layers`). Slower backbone; latency is recovered by NAS.
  2. Batch norm -> layer norm in the multi-scale projector, so gradient accumulation works on consumer GPUs.
  3. Weight-sharing NAS over five knobs, trained end to end (new).
  4. "Scheduler-free" training: no cosine schedule; lower LR 1e-4 vs 4e-4 (`rfdetr-lr`), per-layer LR decay 0.8 (`rfdetr-layerdecay`), grad clip 0.1 (`rfdetr-gradclip`), EMA, no warm-up.
  5. Augmentations reduced to horizontal flip + random crop; resize at the batch level (not per image) to cut padding.
  6. Resolution becomes a NAS knob instead of an augmentation; multi-scale range 0.5 to 1.5 (LW-DETR 0.7 to 1.4) (`rfdetr-msscale`).
  7. Windowed blocks placed at layers {0,1,3,4,6,7,9,10} vs LW-DETR {0,1,3,6,7,9} (`rfdetr-winlayers`); DINOv2's class token is duplicated per window.
  8. Optional instance-segmentation head (RF-DETR-Seg), MaskDINO-inspired, pretrained on Objects-365 pseudo-labeled with SAM2 masks.

## The one key idea

Train ONE DINOv2-based DETR in which every training step uses a randomly sampled architecture (patch size, resolution, windows per attention block; decoder depth and query count are only varied at inference). All sub-networks share the same weights. After training on the target dataset, grid-search thousands of sub-network configurations on a validation set, measure each one's latency, and read off an accuracy-latency Pareto curve. Picking a latency budget = picking a point on that curve; no retraining. The paper also claims the random-architecture sampling acts as a regularizer ("architecture augmentation", compared to dropout ensembles).

## Mechanism step by step

### 1. Base network (Figure 2)
- Pretrained ViT backbone (DINOv2-S for N/S/M/L; DINOv2-B for XL/2XL/Max, `rfdetr-t7-detcfg-*`) extracts features; windowed and non-windowed (global) attention blocks are interleaved.
- Multi-scale projector (layer norm) -> deformable cross-attention decoder (Deformable DETR style). Deformable cross-attention and the seg head both bilinearly interpolate the projector output.
- Two-stage DETR: initial query proposals come from the encoder output; detection (and seg) losses are applied at every decoder layer so layers can be dropped.
- Seg head: bilinearly upsample the encoder feature map, lightweight projector -> pixel embedding map; mask = dot product of each (FFN-transformed) query embedding with the pixel embeddings (prototype-style, like YOLACT). No multi-scale backbone features, for latency. Seg features are at 1/4 of input resolution (Appendix L).

### 2. The five tunable knobs (Figure 3, Sec 3)
| Knob | What it trades | How it is made weight-shareable | Train-time values | Inference-time values |
|---|---|---|---|---|
| Patch size | smaller patch = more tokens = more accurate, slower | FlexiViT-style interpolation of the patch-embedding weights | 8, 10, 12, 16, 20, 24, 32 (`rfdetr-train-patch`) | adds 14 (`rfdetr-inf-patch`) |
| Image resolution | higher = better small objects, slower | pre-allocate positional embeddings for (largest resolution / smallest patch) and interpolate down | 320 to 960 step 64 (11 values) (`rfdetr-train-res`) | same |
| Windows per windowed-attention block | more windows = less global mixing, cheaper attention (but class-token duplication costs) | not stated in the paper (presumably a runtime reshape of the token grid; class token duplicated per window, Appendix L) | 1, 2, 4 (`rfdetr-train-win`) | 1, 2, 4 |
| Number of decoder layers | fewer = faster; 0 layers = single-stage, NMS-free detector | loss at every decoder layer | fixed at 6 (`rfdetr-train-dec`) | 0 to 6 (`rfdetr-inf-dec`) |
| Number of query tokens | fewer = faster, caps max detections | keep top-K queries ranked by max sigmoid class logit at the encoder output | fixed at 300 (`rfdetr-train-q`) | 50, 100, 200, 300 (`rfdetr-inf-q`) |

Note the text of Sec 3 says "At every training iteration, we uniformly sample a random model configuration", while Appendix A says decoder layers and queries are only dropped at inference. So the random sampling during training is over patch size, resolution and windows.

### 3. Weight-sharing training
- Each iteration: uniformly sample one configuration from the training grid, run forward/backward, update the shared weights (Sec 3, Appendix A).
- Resolution sampling is done per batch (whole batch resized to the same size), which also makes all positional-encoding resolutions equally likely.
- Cost: roughly 2 to 4x a non-NAS training run, but yields every size at once (`rfdetr-train-cost`).
- Unlike OFA, no staged training with per-stage LR schedules.

### 4. Picking a configuration at inference
- Grid search all configurations on a validation set only after the base model is fully trained on the target dataset: 6,468 configs (`rfdetr-search-count`), estimated about 10,000 T4 GPU hours (`rfdetr-search-cost`).
- Keep the Pareto-optimal ones; choose one for your latency budget. Configurations with similar parameter counts can have very different latencies (params barely move: DINOv2-S sizes are 30.5M to 33.9M on COCO).
- Optional short fine-tuning of the chosen sub-net: little benefit on COCO (Table 15, e.g. N +0.4, M +0.0 AP), modest gains on RF100-VL (Table 4: N 57.8 -> 58.6). Paper's explanation: the architecture-augmentation regularizer needs more than 100 epochs to converge on small datasets (`rfdetr-ft-100ep`).
- Unseen configurations interpolate well (Appendix F, Figure 5; patch 14 was never trained yet Table 5's base config uses it; Appendix L patch 27 and 18 unseen, `rfdetr-fixed-res-alt`).
- For RF100-VL (100 datasets) there is no single config per size: for each target latency (N..2XL) they search per dataset for models within 10% of the target latency and report the average after fine-tuning (`rfdetr-rf100-10pct`).

### Discovered-architecture trends (Appendix L)
- Patch size is constant within a family: 16 (DINOv2-S det), 20 (DINOv2-B det), 12 (Seg) (`rfdetr-ptch-optimal`). Scaling up within a family raises resolution and decoder depth.
- 2 windows is usually optimal (LW-DETR preferred 4), attributed to class-token duplication cost.
- Accuracy tracks the number of spatial locations (resolution / patch size) more than either alone.
- Detection keeps 300 queries and deepens the decoder ("wide, shallow"); Seg cuts queries and deepens ("thin, deep").
- RF100-VL low-latency optima use fewer queries than COCO optima (fewer objects per image).
- Dataset statistics are only weak predictors of the optimal knobs; the strongest relationship is spatial locations vs windows (R-squared 0.492, `rfdetr-reg-r2-win`; Table 17).

## Equations

The paper has no displayed or numbered equations. Two implicit relations appear in prose:
- Spatial locations (tokens per side) = resolution / patch size. Example: Seg N/S/M at 312/384/432 with patch 12 -> 26/32/36 spatial locations and seg feature resolution 78/96/108 (resolution / 4) (`rfdetr-seg-feat-res`). RF-DETR M (576 / 16) also has 36.
- Search space size = 11 resolutions * 7 patch sizes * 7 decoder layers * 3 windows * 4 query settings = 6,468 (`rfdetr-search-count`; flagged: the inference patch list has 8 values).

## Latency measurement protocol (as the paper describes it)

- Hardware/software: NVIDIA T4, TensorRT 10.4, CUDA 12.4 (`rfdetr-hw`). CUDA graphs enabled in TensorRT (`rfdetr-cudagraphs`); benefits RT-DETR, LW-DETR, RF-DETR but not D-FINE.
- Same artifact for accuracy and latency: accuracy is measured with the very TensorRT engine that is timed, not an FP32 PyTorch model. Motivation: naive FP16 export of D-FINE collapses to 0.5 AP (`rfdetr-dfine-fp16-collapse`); YOLOv8/11 lose AP in TensorRT and more in FP16 (Table 1).
- 200 ms pause between consecutive forward passes to avoid GPU power throttling (`rfdetr-buffer-200ms`); more than 200 ms changes nothing (Table 20, `rfdetr-buffer-beyond`). Explicitly not a throughput benchmark.
- YOLO latency includes NMS (confidence threshold 0.01, `rfdetr-yolo-nms-thr`); YOLO-seg latency includes turning prototypes into masks.
- Models without TensorRT support (GroundingDINO, LLMDet, FastInst, MaskDINO) are timed in PyTorch (starred in tables).
- Latencies are reported to one decimal because TensorRT recompilation gives up to 0.1 ms variance (`rfdetr-latency-variance`).
- Precision: Table 1 reports both FP32 and FP16; the main tables are FP16 by inference only (`rfdetr-fp16-main`, uncertain). The evidence is strong: Table 2's AP and latency equal Table 1's FP-16 column for every model (e.g. YOLOv8 M 47.3 AP / 5.4 ms; its FP32 AP is 49.3), so Table 2 accuracy is also FP16 (except YOLOv11 M latency 5.1 vs 5.2). Batch size is not stated in the paper (`rfdetr-batchsize-latency`, uncertain); the README says batch size 1.
- Buffering effect for RF-DETR M: 4.7 ms with 0 ms buffer vs 4.4 ms with 200 ms (`rfdetr-t20-buf-rfdetrm`). FP32 buffered 20.5 ms vs FP16 buffered 4.4 ms (`rfdetr-t1-latency-rfdetrm`).
- Released tool: github.com/roboflow/single_artifact_benchmarking.

## Results worth remembering

- COCO nano: 48.0 AP at 2.3 ms vs D-FINE N 42.7 at 2.1 ms (+5.3) (`rfdetr-nano-coco-ap`, `rfdetr-nano-beats-dfine`, `rfdetr-t2-cocodet-rfdetroursn`, `rfdetr-t2-cocodet-dfinen`). Nano matches YOLOv8/11 medium (47.3 / 48.3 at 5.4 / 5.1 ms).
- COCO 2XL: 60.1 AP at 17.2 ms, first real-time detector above 60 AP (`rfdetr-first-60ap`); beats D-FINE XL (59.3) by 0.8 (`rfdetr-2xl-vs-dfinexl`). But D-FINE XL beats RF-DETR XL (59.3 vs 58.6, both 11.5 ms; `rfdetr-t12-cocodetL-rfdetroursxl`) and D-FINE M/L beat RF-DETR M/L (55.0 vs 54.7; 57.2 vs 56.5), though RF-DETR is faster at M (4.4 vs 5.4 ms) and L (6.8 vs 7.5 ms).
- Max variant: 61.8 AP at 98.0 ms (`rfdetr-t12-cocodetL-rfdetroursmax`).
- RF100-VL: RF-DETR 2XL 63.3 AP (63.5 fine-tuned) at 15.6 ms vs GroundingDINO T 62.3 at 309.9 ms (PyTorch) (`rfdetr-t4-rf100-rfdetrours2xl`, `rfdetr-2xl-beats-gdino` (uncertain: 1.0 vs 1.2 AP depends on row), `rfdetr-2xl-20x`).
- RF100-VL: RT-DETR M (59.6 AP, 85.7 AP50) beats D-FINE M on AP50 (85.5); the paper reads this as D-FINE hyperparameters overfit to COCO. YOLOv8/11 do not improve with scale on RF100-VL (N 55.0/55.5 -> XL 56.5/56.2).
- NAS ablation (Table 5, COCO, medium): LW-DETR M 52.6 -> gentler HPs 51.6 -> DINOv2 53.6 -> more O365 54.3 -> weight-sharing NAS 54.6, then knob turns to the final config (patch 16, res 576, 2 windows, 4 decoder layers, 300 queries) at 54.6 AP / 4.4 ms (`rfdetr-nas-gain`, `rfdetr-t5-nas-*`). Weight-sharing NAS improves even the fixed base config (+0.3), supporting the regularizer claim.
- Backbone ablation (Table 6): DINOv2-S 54.3 > SAM2 Hiera-S 53.6 (11.2 ms) > CAEv2 52.3 > SigLIPv2-B 50.4 (`rfdetr-t6-bb-*`, `rfdetr-dinov2-gain`).
- Dataset-specific NAS matters: COCO-searched (fixed) architecture transferred to RF100-VL gives N 57.7 vs RF100-VL-searched 57.8 (58.6 fine-tuned); S 60.2 vs 60.9 (61.2); M 61.2 vs 61.7 (62.0) (Table 18, `rfdetr-t18-fixed-*`). At L the fixed architecture actually scores higher (62.2 vs 62.0 at 6.8 vs 6.2 ms).
- Scaling (Table 11): the DINOv2-S family leads D-FINE at S (+2.3) but trails at XL (-1.1); the DINOv2-B family closes the gap as latency grows (-3.1 -> -0.7) (`rfdetr-t11-gap-*`).
- Segmentation: RF-DETR-Seg N 40.3 AP at 3.4 ms beats every YOLOv8/11-seg size (best YOLOv11-XL 40.1 at 13.7 ms) (`rfdetr-seg-nano-vs-yolo11x`); beats FastInst by 5.4 (`rfdetr-seg-nano-vs-fastinst`). 2XL 49.9 at 21.8 ms; Max 50.5 at 95.6 ms (Table 13).
- Objects365 pretraining is not labelled in the result tables. RF-DETR and RF-DETR-Seg are O365-pretrained (Sec 3, Appendix A, Table 5 row "Additional O365 Pre-Training"); LW-DETR is O365-pretrained; D-FINE S and larger are too (Table 11 caption excludes D-FINE N because it was not). YOLO rows are not. RT-DETR baseline APs (R18 49.0, R50 55.0, R101 56.1) match RT-DETR's own O365 results (49.2 / 55.3 / 56.2), not its COCO-only 46.5 / 53.1 / 54.3 (inferred; RF-DETR does not say).
- Decoder/query dropping: removing the last decoder layer costs 2 mAP for 10% latency (`rfdetr-drop-last-dec`); dropping 100 lowest-confidence queries costs almost nothing (`rfdetr-drop-100q`).

## Limitations

- Stated: residual 0.1 ms latency variance from TensorRT recompilation (power throttling during compilation); 200 ms buffering inflates total benchmark time and does not measure sustained throughput.
- Stated: fine-tuning after NAS is inconsistent (helps RF100-VL, not COCO; Seg fine-tuning "Did Not Improve" for S and above, Table 16). NAS on small datasets may need more than 100 epochs (`rfdetr-ft-100ep`).
- Stated: foundation-model families rarely ship tiny ViTs (ViT-S/T), limiting backbone choice; layer norm (needed for consumer GPUs) hurts accuracy a bit.
- Inferred from paper numbers: search cost is about 10,000 T4 GPU hours for the grid search, and 2 to 4x training cost.
- Inferred (not stated by the paper): parameters are large for the latency class (N is 30.5M vs D-FINE N 3.8M); speed comes from TensorRT-friendly ViT + FlashAttention-style kernels, so latency ranking is hardware-specific (only T4 is measured).
- Paper-supported (Appendix E, not in its Limitations paragraph): DINOv2-S family scales worse than D-FINE past M on COCO (Table 11).
- Paper-supported, not in its Limitations paragraph: Sec 4 criticizes D-FINE for tuning on COCO val and reporting on it, but RF-DETR says its configs are grid-searched "on a validation set" (Sec 1) and reports all COCO results on val2017 (Sec 4); it does not say a separate split was used for the COCO search. RF100-VL results are on a separate test split.
- Internal inconsistencies worth knowing (flagged in the JSON): Table 4 vs Table 18 RF-DETR M params/GFLOPs (33.5M/86.7 vs 33.6M/91.0); Table 14 D-FINE XL params/GFLOPs columns look pasted from COCO AP values; Table 5 caption "smaller batch size" vs text "larger" vs Appendix A "same (128)"; Table 6 caption "2.4%" vs rows 2.0; LW-DETR M GFLOPs 83.7 (Table 5) vs 83.9 (Table 2); Table 5 final row does not reproduce Table 2 M (64.8 vs 78.8 GFLOPs, 54.6 vs 54.7 AP); search-count product uses 7 patch sizes but 8 are listed; Sec 4 says nano's >5 AP lead over D-FINE shows "similar trends for small and medium sizes", but Table 2 has RF-DETR S +2.3 and D-FINE M ahead (55.0 vs 54.7) (`rfdetr-similar-trends`); Table 3's 2XL row is labeled "RF-DETR" but is the Seg model (38.6M, Seg config); the RT-DETR baseline labelled "R18" in Tables 1/2 has 36.0M / 100.0 GFLOPs, which are RT-DETR-R50-m's numbers (R18 is 20M / 60 GFLOPs), and Table 4 calls the same 36.0M model "M" (`rfdetr-t2-cocodet-rtdetrr18`, `rfdetr-t4-rf100-rtdetrm`); Appendix H text says classes-vs-decoder-layers and objects-per-image-vs-queries are "strong correlations", but Table 17 gives R-squared 0.026 (p 0.106) and 0.019 (p 0.171).

## Facts useful for visuals (Pareto points)

All latencies: T4, TensorRT 10.4, CUDA 12.4, 200 ms buffer, CUDA graphs, presumably FP16 (see protocol). AP = COCO val2017 AP50:95 or RF100-VL test AP50:95 averaged over 100 datasets.

### COCO detection, Pareto-optimal named configs (Table 7 config + Table 2 / Table 12 metrics)
| Size | Backbone | Res | Patch | Tokens/side | Windows | Dec layers | Queries | Params | GFLOPs | Latency ms | COCO AP | AP50 | claim ids |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | DINOv2-S | 384 | 16 | 24 | 2 | 2 | 300 | 30.5M | 31.9 | 2.3 | 48.0 | 67.0 | rfdetr-t7-detcfg-n, rfdetr-t2-cocodet-rfdetroursn |
| S | DINOv2-S | 512 | 16 | 32 | 2 | 3 | 300 | 32.1M | 59.8 | 3.5 | 52.9 | 71.9 | rfdetr-t7-detcfg-s, rfdetr-t2-cocodet-rfdetrourss |
| M | DINOv2-S | 576 | 16 | 36 | 2 | 4 | 300 | 33.7M | 78.8 | 4.4 | 54.7 | 73.5 | rfdetr-t7-detcfg-m, rfdetr-t2-cocodet-rfdetroursm |
| L | DINOv2-S | 704 | 16 | 44 | 2 | 4 | 300 | 33.9M | 125.6 | 6.8 | 56.5 | 75.1 | rfdetr-t7-detcfg-l, rfdetr-t12-cocodetL-rfdetroursl |
| XL | DINOv2-B | 700 | 20 | 35 | 1 | 5 | 300 | 126.4M | 299.3 | 11.5 | 58.6 | 77.4 | rfdetr-t7-detcfg-xl, rfdetr-t12-cocodetL-rfdetroursxl |
| 2XL | DINOv2-B | 880 | 20 | 44 | 2 | 5 | 300 | 126.9M | 438.4 | 17.2 | 60.1 | 78.5 | rfdetr-t7-detcfg-2xl, rfdetr-t12-cocodetL-rfdetrours2xl |
| Max | DINOv2-B | 828 | 12 | 69 | 1 | 6 | 300 | 132.4M | 1742.5 | 98.0 | 61.8 | 79.7 | rfdetr-t7-detcfg-max, rfdetr-t12-cocodetL-rfdetroursmax |

Tokens/side = Res / Patch, computed here (not a paper column).

COCO baselines to draw on the same axes (Table 2 / Table 12; AP @ ms): LW-DETR T 42.9@1.9, S 48.0@2.6, M 52.6@4.4, L 56.1@6.9, XL 58.3@13.0; D-FINE N 42.7@2.1, S 50.6@3.5, M 55.0@5.4, L 57.2@7.5, XL 59.3@11.5; RT-DETR "R18" 49.0@4.4 (params/GFLOPs are R50-m's, see inconsistencies), R50 55.0@8.5, R101 56.1@12.0 (these RT-DETR APs look like the O365-pretrained checkpoints, not RT-DETR's headline 53.1 / 54.3); YOLOv11 N 37.1@2.2, S 44.1@3.2, M 48.3@5.1, L 49.9@6.5, XL 50.9@10.5; YOLOv8 N 35.2@2.1, S 42.4@2.9, M 47.3@5.4, L 49.5@8.0, XL 50.5@11.3. (ids `rfdetr-t2-cocodet-*`, `rfdetr-t12-cocodetL-*`.)

### COCO segmentation (Table 8 config + Tables 3 / 13)
| Size | Res | Patch | Windows | Dec | Queries | Params | GFLOPs | ms | Mask AP |
|---|---|---|---|---|---|---|---|---|---|
| N | 312 | 12 | 1 | 4 | 100 | 33.6M | 50.0 | 3.4 | 40.3 |
| S | 384 | 12 | 2 | 4 | 100 | 33.7M | 70.6 | 4.4 | 43.1 |
| M | 432 | 12 | 2 | 5 | 200 | 35.7M | 102.0 | 5.9 | 45.3 |
| L | 504 | 12 | 2 | 5 | 300 | 36.2M | 151.1 | 8.8 | 47.1 |
| XL | 624 | 12 | 2 | 6 | 300 | 38.1M | 260.0 | 13.5 | 48.8 |
| 2XL | 768 | 12 | 2 | 6 | 300 | 38.6M | 435.3 | 21.8 | 49.9 |
| Max | 890 | 10 | 1 | 6 | 300 | 40.1M | 1668.2 | 95.6 | 50.5 |
All DINOv2-S. ids `rfdetr-t8-segcfg-*`, `rfdetr-t3-cocoseg-*`, `rfdetr-t13-cocosegL-*`.

### RF100-VL detection (Table 4 / Table 14 / Table 18; averages over 100 datasets, each dataset has its own searched config)
| Size | Params | GFLOPs | ms | NAS AP | NAS + fine-tune AP | COCO-fixed-arch AP (ms) | claim ids |
|---|---|---|---|---|---|---|---|
| N | 31.2M | 34.5 | 2.5 | 57.8 | 58.6 | 57.7 (2.3) | rfdetr-t4-rf100-rfdetroursn, rfdetr-t18-fixed-rfdetroursfixedarchitecturen |
| S | 33.5M | 62.4 | 3.7 | 60.9 | 61.2 | 60.2 (3.5) | rfdetr-t4-rf100-rfdetrourss |
| M | 33.5M (T18: 33.6M) | 86.7 (T18: 91.0) | 4.6 | 61.7 | 62.0 | 61.2 (4.4) | rfdetr-t4-rf100-rfdetroursm (uncertain) |
| L | 34.1M | 119.1 | 6.2 | 62.0 | 62.3 | 62.2 (6.8) | rfdetr-t14-rf100L-rfdetroursl |
| XL | 35.0M | 199.0 | 9.7 | 62.6 | 63.0 | 62.9 (11.5, DINOv2-B) | rfdetr-t14-rf100L-rfdetroursxl |
| 2XL | 123.5M | 410.2 | 15.6 | 63.3 | 63.5 | 63.2 (17.1, DINOv2-B) | rfdetr-t4-rf100-rfdetrours2xl |
Params/GFLOPs/latency here are averages over the per-dataset models, so they are not one network. RF100-VL XL is DINOv2-S sized (35.0M) while COCO XL is DINOv2-B (126.4M).
RF100-VL baselines (AP @ ms): LW-DETR N 57.1@1.9, S 57.4@2.6, M 59.8@4.3, L 61.5@6.8, XL 62.1@13.0; D-FINE N 58.2@2.0, S 60.3@3.5, M 60.6@5.6, L 61.6@7.5, XL 62.2@11.4; RT-DETR M 59.6@4.3, R50 61.7@8.4, R101 61.0@11.9; GroundingDINO T 62.3@309.9 (PyTorch); LLMDet T 62.3@308.4 (PyTorch); YOLOv11 N 55.5@3.0 ... XL 56.2@10.3.

### Guidance for the latency-budget slider
- The discrete, table-backed points are the COCO rows above (N..Max). The continuous curves in Figure 1 (two lines: RF-DETR DINOv2-S and DINOv2-B) have no data table; any intermediate value read off the plot must be marked "read from Figure 1" (uncertain). From the plot the DINOv2-S curve starts near ~1.5 ms / ~36 AP and plateaus near ~58.8 AP by ~16-18 ms; the DINOv2-B curve reaches ~60.4 AP near ~19.8 ms (visual estimates only, `rfdetr-fig1-visual-read`, uncertain).
- Two separate base networks (DINOv2-S and DINOv2-B) exist; "one training run" is per backbone family.
- Knob moves the NAS makes as budget grows (COCO det, DINOv2-S): resolution 384 -> 512 -> 576 -> 704, decoder layers 2 -> 3 -> 4 -> 4, patch fixed 16, windows fixed 2, queries fixed 300. Then a jump to DINOv2-B at patch 20.
- For RF100-VL there is no per-size knob setting to display (per-dataset configs averaged).

### README vs paper (github.com/roboflow/rf-detr, develop @ 990f5d9)
- README protocol: T4, TensorRT, FP16, batch size 1; COCO AP measured in-house with SAB on val2017; params are fused-module counts (`rfdetr-readme-scope`).
- README detection table (`rfdetr-readme-N` ... `rfdetr-readme-2XL`): COCO AP50:95 48.4 / 53.0 / 54.7 / 56.5 / 58.6 / 60.1; RF100VL AP50:95 57.7 / 60.2 / 61.2 / 62.2 / 62.9 / 63.2; latency 2.3 / 3.5 / 4.4 / 6.8 / 11.5 / 17.2 ms; params 30.5 / 32.1 / 33.7 / 33.9 / 126.4 / 126.9M; resolutions 384 / 512 / 576 / 704 / 700 / 880.
- Differences: README COCO N, S and M (48.4 / 67.6, 53.0 / 72.1, 54.7 / 73.6) equal the paper's fine-tuned values (Table 2 + Table 15 deltas), not Table 2 (48.0 / 67.0, 52.9 / 71.9, 54.7 / 73.5); L, XL and 2XL equal the non-fine-tuned rows. README RF100-VL numbers for every size equal Table 18 "Fixed Architecture" rows (COCO-searched config transferred to RF100-VL), NOT the paper's per-dataset NAS results of Table 4/14 (57.8 / 60.9 / 61.7 / 62.0 / 62.6 / 63.3). So the shipped checkpoints do not embody the paper's headline RF100-VL NAS numbers; the README says per-dataset NAS is offered on the Roboflow platform and claims it now beats the paper's NAS (`rfdetr-readme-platform-nas`, uncertain, chart only).
- XL and 2XL are licensed PML 1.0 (not Apache 2.0) in the README.
- README Seg table matches paper Tables 3 / 13 exactly (`rfdetr-readme-seg`).
- README baseline numbers differ in places (YOLO11-N 37.4 AP at 2.5 ms vs paper 37.1 at 2.2 ms, `rfdetr-readme-yolo11n`) and add YOLO26, SAM 3 and a keypoint preview not in the paper (`rfdetr-readme-extras`).
