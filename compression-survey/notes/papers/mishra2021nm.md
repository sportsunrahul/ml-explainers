# mishra2021nm -- Accelerating Sparse Deep Neural Networks (BRANCH, full vision rows)

## Identity
- Title: Accelerating Sparse Deep Neural Networks
- arXiv 2104.08378, read v1 (2021-04-16, only version) (abs.html).
- Authors: Asit Mishra ... Paulius Micikevicius (NVIDIA; 8 authors). No venue stated in abs.html.
- Code named in the paper: Automatic SParsity (ASP) library for PyTorch (Sec. 4.1); cuSPARSELt; TensorRT >= 8.0.
- Source macro: `\SPARSITY` expands to "2:4 " (src/AcceleratingSparsity.tex).

## Key idea (two sentences)
Require that every group of 4 consecutive weights has at least 2 zeros (2:4, 50% sparsity); this fixed,
fine-grained pattern lets Ampere Sparse Tensor Cores skip half the multiplies with a tiny index (2 bits per kept
value). A simple workflow (train dense, prune 2:4 by magnitude once, retrain with the same schedule and a fixed
mask) keeps accuracy across classification, detection, segmentation, GANs and NLP.

## Pattern and storage (Sec. 3.1, Fig. 1)
- Pattern: "for each group of 4 values, at least 2 must be zero" -> 50% sparsity. Groups run along the GEMM
  reduction dimension K (rows of A are compressed from K to K/2, Fig. 2). For TF32 the pattern becomes 1:2.
- Compressed format: an R x C matrix becomes R x C/2 values plus R x C/2 2-bit indices. Fig. 1 first row
  metadata: [[0, 3], [1, 2]]. A group with more than 2 zeros still stores 2 values.
- Overhead: 2 bits of metadata per stored value = 12.5% (16-bit values) or 25% (8-bit values). CSR for
  unstructured sparsity can cost up to 200% metadata overhead (8b values with 16b column indices).
- Savings: FP16: 4*16 = 64 bits dense vs 2*16 + 2*2 = 36 bits -> ~44% less storage. INT8: ~38% less memory
  capacity and bandwidth.
- Why it is fast: same sparsity in every sub-block, so full-width memory reads and no data-dependent indirection
  (vs CSR/CSC/COO) (Sec. 3.1).

## Math throughput (Sec. 3.2, Table 1, Fig. 3)
- Sparse Tensor Core: sparse A (compressed) x dense B = dense C; hardware selects the matching elements of B
  using the metadata; "up to" 2x over dense GEMM. Supported inputs: FP16, BF16, INT8; TF32 at 1:2.
- Table 1, A100 peak TOPS (dense / sparse): FP32-FP32 acc 19.5 / -; TF32 156 / 312; FP16 (FP32 acc) 312 / 624;
  BF16 312 / 624; FP16 (FP16 acc) 312 / 624; INT8 (INT32 acc) 624 / 1248.
- Constraint: input dimensions must be multiples of 16 (FP16/BF16) or 32 (INT8) (Sec. 3.2, 4.2).
- Software: cuSPARSELt APIs; TensorRT from version 8.0 compresses and runs 2:4 weights automatically (Sec. 3.2).
- Fig. 3 (only measured speed in the paper): INT8 GEMM, cuSPARSELt (sparse) vs cuBLAS (dense), NVIDIA A100,
  GEMM-M = GEMM-N = 10240, GEMM-K swept 1280 to 20480. Caption: "Larger GEMMs achieve nearly a 2x speedup".
  Text: larger GEMMs have higher arithmetic intensity and get closer to 2x. Bars are not stated in text; the
  smallest K (1280) reads about 1.2x from the plot (approximate, our reading).
- NO end-to-end network latency or throughput is measured anywhere in the paper. Every vision table is accuracy
  only. Any network-level speedup on the page must come from another source or be labelled "up to 2x math".

## Workflow (Sec. 4.1, Fig. 4)
1. Train the model dense (Step 1).
2. Prune to 2:4 by magnitude: remove the two smallest of each group of four (Step 2, Fig. 5). Optional channel
   permutation first (Fig. 6) for parameter-efficient nets.
3. Retrain (Step 3): initialize from the Step-2 weights; same optimizer, lr schedule and number of epochs as
   Step 1; keep the mask fixed; reset optimizer state (momenta) and hyper-parameters.
- Training cost doubles; no hyper-parameter search (Sec. 4, Sec. 6).
- Layers pruned: conv, FC, recurrent GEMM layers; not embeddings, not training-only heads; not layers whose
  GEMM-K (C x R x S for conv) is not a multiple of 16/32 (Sec. 4.2). In vision, the first 7x7 conv on 3 input
  channels is not pruned (Sec. 5.1).
- Detection / segmentation recipe (Sec. 4.3, Fig. 8): train dense backbone on ImageNet, add heads, train on
  COCO; THEN prune everything (backbone + heads) and repeat ONLY the detection/segmentation fine-tuning phase.
  "the task-specific heads have to be fine-tuned before they can be pruned". (BERT is the opposite case: repeat
  pre-training, Fig. 7.)
- Quantization: apply INT8 calibration (and any fine-tuning) AFTER sparse retraining (Sec. 4.4).
- Fig. 6 toy: 4x8 random matrix, 2:4 pruning keeps total magnitude 83.7; after column permutation 102.9.

## INT8 setup for the vision rows (Sec. 5.1, 5.2)
- Classification: weights and input activations of conv (incl. first layer) and FC quantized to INT8. Activations:
  entropy calibration, per-tensor scale. Weights: max calibration, per-channel (conv) / per-row (FC) scale.
- Detection / segmentation: all layers (incl. first conv) INT8; same calibrators and granularity.
- The text describes calibration only (PTQ); no INT8 fine-tuning is mentioned for vision rows (our reading).
- IMPORTANT for reading the tables: there is NO dense-INT8 column in Tables 2 and 4. "Sparse INT8" is compared
  with "Dense FP16". The FP16 columns are FP16 inference of FP-trained nets.
- Paper's own reading: differences "are within bounds of run-to-run variation caused by random seeds or
  fine-tuning non-determinism" (Sec. 5.1).

## Table 4 (COCO 2017; bbox AP top block, mask AP bottom block; Dense FP16 / Sparse FP16 / Sparse INT8)
Detection (bbox AP):
| network | dense FP16 | sparse FP16 | sparse INT8 |
|---|---|---|---|
| MaskRCNN-RN50 | 37.9 | 37.9 | 37.8 |
| RetinaNet-RN50 | 34.8 | 35.2 | 35.1 |
| SSD-RN50 | 24.8 | 24.8 | 24.9 |
| FasterRCNN-RN50-FPN-1x | 37.6 | 38.6 | 38.4 |
| FasterRCNN-RN50-FPN-3x | 39.8 | 39.9 | 39.4 |
| FasterRCNN-RN101-FPN-3x | 41.9 | 42.0 | 41.8 |
| MaskRCNN-RN50-FPN-1x | 39.9 | 40.3 | 40.0 |
| MaskRCNN-RN50-FPN-3x | 40.6 | 40.7 | 40.4 |
| MaskRCNN-RN101-FPN-3x | 42.9 | 43.2 | 42.8 |
| RetinaNet-RN50-FPN-1x | 36.4 | 37.4 | 37.2 |
| RPN-RN50-FPN-1x | 45.8 | 45.6 | 45.5 |
Segmentation:
| network | dense FP16 | sparse FP16 | sparse INT8 |
|---|---|---|---|
| MaskRCNN-RN50 (mask AP) | 34.5 | 34.6 | 34.5 |
| MaskRCNN-RN50-FPN-3x (mask AP) | 36.9 | 37.0 | - |
| MaskRCNN-RN101-FPN-3x (mask AP) | 38.7 | 38.9 | - |
| ResNeXt-101* (IoU) | 81.4 | 81.7 | 81.2 |
- `*` "IoU of network pre-trained on Mapillary and fine-tuned on Cityscapes dataset" (not COCO; semantic seg).
- 1x / 3x = Detectron2 fine-tuning schedule length. RN = ResNet; FPN = Feature Pyramid Network; RPN = Region
  Proposal Network (RPN row metric is the paper's "bbox AP" column; as printed).
- Sources of models: Torchvision, Detectron2, NVIDIA DL Examples, NVIDIA ADLR (Sec. 5.2).
- Shape fact: sparse FP16 minus dense FP16 ranges from -0.2 (RPN) to +1.0 (FasterRCNN-RN50-FPN-1x and
  RetinaNet-RN50-FPN-1x) absolute AP points (our arithmetic on Table 4; the page should recompute).

## Table 2 (ImageNet top-1 %, Dense FP16 / Sparse FP16 / Sparse INT8)
| network | dense FP16 | sparse FP16 | sparse INT8 |
|---|---|---|---|
| ResNet-34 | 73.7 | 73.9 | 73.7 |
| ResNet-50 | 76.1 | 76.2 | 76.2 |
| ResNet-50 (SWSL) | 81.1 | 80.9 | 80.9 |
| ResNet-101 | 77.7 | 78.0 | 77.9 |
| ResNeXt-50-32x4 | 77.6 | 77.7 | 77.7 |
| ResNeXt-101-32x16 | 79.7 | 79.9 | 79.9 |
| ResNeXt-101-32x16 (WSL) | 84.2 | 84.0 | 84.2 |
| DenseNet-121 | 75.5 | 75.3 | 75.3 |
| DenseNet-161 | 78.8 | 78.8 | 78.9 |
| Wide ResNet-50 | 78.5 | 78.6 | 78.5 |
| Wide ResNet-101 | 78.9 | 79.2 | 79.1 |
| Inception v3 | 77.1 | 77.1 | 77.1 |
| Xception | 79.2 | 79.2 | 79.2 |
| VGG-11 | 70.9 | 70.9 | 70.8 |
| VGG-16 | 74.0 | 74.1 | 74.1 |
| VGG-19 | 75.0 | 75.0 | 75.0 |
| SUNet-128 | 75.6 | 76.0 | 75.4 |
| SUNet-7-128 | 76.4 | 76.5 | 76.3 |
| DRN26 | 75.2 | 75.3 | 75.3 |
| DRN-105 | 79.4 | 79.5 | 79.4 |

## Table 3 (ImageNet top-1 %, FP16; parameter-efficient nets; Dense / 2:4 Default / 2:4 Permuted)
| network | dense | 2:4 default | 2:4 permuted |
|---|---|---|---|
| MobileNet v2 | 71.55 | 69.56 | 71.56 |
| SqueezeNet v1.0 | 58.09 | 54.08 | 58.38 |
| SqueezeNet v1.1 | 58.21 | 56.96 | 58.23 |
| MNASNet 1.0 | 73.24 | 71.99 | 73.27 |
| ShuffleNet v2 | 68.32 | 66.97 | 68.42 |
| EfficientNet B0 | 77.25 | 75.98 | 77.29 |
| EfficientNet-WideSE B0 | 77.63 | 76.64 | 77.63 |
- "Default" = the basic workflow; "Permuted" = channel permutation before pruning (Sec. 4.1, Fig. 6). How
  permutations are found is "presented separately" (not in this paper).

## Other context numbers
- Related work (Sec. 2): channel pruning gives ResNet-50 a 2x speedup "but close to 1.5% accuracy is lost"
  (citing He 2017); unstructured pruning benefit "is negligible and at times negative, even when pruning rate is
  high (e.g. 95%)" (citing Wen 2016).
- Conclusion (Sec. 6): unstructured sparsity "requires very high levels of sparsity in order to achieve speedups
  over dense math on modern processors with matrix-math pipelines".
- Non-vision tables (not extracted in full): GANs Table 5 (FID), translation Table 6 (BLEU), LM Table 7.

## Outline feedback
- ch-lottery "2:4 (Mishra 2021: train dense, prune 2:4, retrain)": correct for classification. For detectors the
  paper's recipe is "train dense detector, prune backbone + heads, repeat only the COCO fine-tuning" (Sec. 4.3,
  Fig. 8). The guide should state this; it halves the retraining cost vs repeating ImageNet pre-training.
- Detection/segmentation rows ARE present (Table 4): 11 detection + 4 segmentation rows above.
- The guide's "depthwise-heavy detector" row: Table 3 shows the basic workflow loses accuracy on MobileNet v2
  (-1.99) and EfficientNet B0 (-1.27) unless channels are permuted (our arithmetic). EfficientDet uses an
  EfficientNet backbone, so plan on permutation (ASP) (our reading).
- No end-to-end speedup is measured; the only measured number is the INT8 GEMM microbenchmark (Fig. 3). The
  roofline chapter's toy can use "up to 2x math" and Table 1 TOPS; a real detector speedup must be "measure it".
- viz-sparse "half the math with sparse tensor cores": correct wording is "up to 2x math throughput"; memory for
  weights falls by ~44% (FP16) / ~38% (INT8), not 50%, because of the 2-bit metadata.
- Requires Ampere or newer (A100 is the only GPU in the paper); the guide's GPU list should mark Turing as no
  sparse tensor cores (our reading; the paper only says "NVIDIA Ampere GPU architecture introduces").
