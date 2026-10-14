# han2016dc -- Deep Compression (SPINE)

## 1. Identity
- Title: Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding
- arXiv 1510.00149, read v5 (2016-02-15). v1 submitted 2015-10-01 (abs.html).
- Authors: Song Han (Stanford) ... William J. Dally (Stanford, NVIDIA); middle author Huizi Mao (Tsinghua).
- Venue: ICLR 2016 (oral), per abs.html comments. No repo cited in the paper.
- Source read: src/iclr2016_conference.tex (exact tables/equations); paper.pdf for figure, table and equation numbers.

## 2. Core problem
AlexNet (240 MB) and VGG-16 (552 MB) are too large for mobile apps and their weights live in DRAM, which costs
energy. The goal is to cut storage (and DRAM traffic) enough that weights fit in on-chip SRAM, with no accuracy loss.

## 3. What prior methods did and why that hurt (per the paper)
- Pruning alone (Han 2015): 9x on AlexNet, not counting index overhead (Table 7 caption).
- HashedNets: weight sharing fixed by a hash before training, "doesn't capture the nature of images"; vector
  quantization (Gong 2014): 1% accuracy loss; both only on FC layers (Sec. 7).
- SVD (Denton 2014): 5x on AlexNet with 44.02% top-1 error vs 42.78% (Table 7).

## 4. The one key idea
Pruning and trained quantization compress "without interfering each other" (Sec. 1), so their ratios
multiply; Huffman coding then exploits the skewed distribution of the remaining codes.

## 5. Mechanism, step by step
1. Train the dense network normally ("learn the connectivity").
2. Prune: remove all connections with |weight| below a threshold (Sec. 2). Retrain the survivors; pruned
   connections are masked in Caffe so they get no update (Sec. 5). AlexNet keeps 11% of weights (9x),
   VGG-16 keeps 7.5% (13x) (Tables 4, 5).
3. Store the sparse layer in CSR or CSC: 2a+n+1 numbers (a = non-zeros, n = rows or columns) (Sec. 2).
4. Store the index *difference* to the previous non-zero (relative index) in few bits. If the gap exceeds
   the largest code, insert a filler zero (Fig. 2: a 3-bit diff cannot span more than 8, so a zero is
   stored at position 12).
5. Weight sharing: per layer, 1-D k-means on the surviving weights into k clusters (Eq. 2). Weights are not
   shared across layers. Each connection then stores a log2(k)-bit index into a codebook of k FP32 centroids.
6. Centroid initialization: linear (evenly spaced over [min, max]) is used; it beats density-based (CDF
   spaced) and Forgy (random sample) because it keeps a few large centroids for the rare large weights
   (Sec. 3.2, 6.2).
7. Fine-tune the centroids: gradients of all weights in a cluster are summed (Eq. 3), multiplied by the
   learning rate, and subtracted from that centroid (Fig. 3). The index assignment stays fixed.
8. Bits chosen: 8 bits (256 shared weights) per CONV layer, 5 bits (32 shared weights) per FC layer for
   AlexNet and VGG-16 (Sec. 3, 5.2, 5.3).
9. Huffman-code both the quantized weights and the relative indices offline (no training). Saves 20%-30%
   of storage (Sec. 4); 22% extra on AlexNet (Sec. 5.2).

## 6. Equations
Custom macros: none for these; Eq. 3 uses `\mathds{1}` from package dsfont and `\cal{L}`.

- Eq. 1 (label `eq_save`), compression rate of weight sharing:
  `r = \frac{nb}{nlog_{2}(k)+kb}`
  Paper: "for a network with n connections and each connection is represented with b bits, constraining the
  connections to have only k shared weights will result in a compression rate of" r.
  Gloss: n = number of connections; b = bits per original weight (32); k = number of clusters; n log2(k) =
  index bits; k b = codebook bits. Worked example in text: 16*32/(4*32+2*16) = 3.2 for the 4x4 layer of Fig. 3.
- Eq. 2 (label `eq_save2`), k-means objective (WCSS):
  `\underset{C} {\operatorname{arg\min}}  \sum_{i=1}^{k} \sum_{ w \in c_i} \left|  w - c_i \right|^2`
  Paper: partition n weights W into k clusters C, n >> k, "so as to minimize the within-cluster sum of squares".
  Gloss: c_i is cluster i and also its centroid value; w ranges over the weights assigned to it.
- Eq. 3 (unlabelled; numbered (3) in the PDF), shared-weight gradient:
  `\frac{\partial \cal{L}}{\partial C_k} = \sum_{i,j}\frac{\partial  \cal{L}}{\partial  W_{ij}}\frac{\partial W_{ij}}{\partial C_k} = \sum_{i,j}\frac{\partial  \cal{L}}{\partial  W_{ij}}\mathds{1}({ I_{ij}=k })`
  Paper: "the gradient of the centroids is calculated as" above.
  Gloss: L = loss; W_ij = weight at column i, row j; I_ij = its centroid index; C_k = centroid k;
  the indicator is 1 when weight (i,j) belongs to cluster k. So a centroid's gradient is the sum of the
  ordinary weight gradients over its members.
- Magnitude criterion: no equation in this paper. Sec. 2: "all connections with weights below a threshold
  are removed". How the threshold is set is in Han 2015 (quality parameter x layer std; see han2015 notes).
- Sparse storage size: "2a+n+1 numbers" (Sec. 2, inline).

## 7. Config / setup
| item | value | loc |
|---|---|---|
| framework | Caffe; pruning = mask on blobs; quantization = codebook + group-by-index of gradients | Sec. 5 |
| networks | LeNet-300-100, LeNet-5 (MNIST); AlexNet, VGG-16 (ImageNet ILSVRC-2012) | Sec. 5 |
| reference models | Caffe model zoo; accuracy measured without data augmentation | Table 1 footnote |
| CONV bits | 8 (256 shared weights) | Sec. 3, 5.2, 5.3 |
| FC bits | 5 (32 shared weights) | Sec. 3, 5.2, 5.3 |
| relative index bits | Sec. 2 text: 8 (conv), 5 (fc). Sec. 5.2 and Table 4: 4 for AlexNet. Table 5: 5 for every VGG-16 layer. Tables 2-3: 5 for LeNets. Inconsistent in the source; do not reconcile. | Sec. 2; Sec. 5.2; Tables 2-5 |
| centroid init | linear | Sec. 3.2, 6.2 |
| Huffman | offline, after all fine-tuning | Sec. 5 |
| speed benchmark layers | FC6, FC7, FC8 of AlexNet and VGG-16 only; quantized model NOT benchmarked (BLAS lacks indirect lookup and relative indexing) | Sec. 6.3 |
| speed hardware | NVIDIA GeForce GTX Titan X; Intel Core i7-5930K; NVIDIA Tegra K1 (Jetson TK1) | Sec. 6.3 |
| kernels | GPU: cuBLAS GEMV (dense) vs cuSPARSE CSRMV (pruned). CPU: MKL CBLAS GEMV vs MKL SPBLAS CSRMV. Batch 64: gemm vs csrmm | Sec. 6.3; Table 8 caption |
| timing method | time per layer over 4096 input samples, averaged per sample; cudaMalloc/cudaMemcpy not counted | Table 8 caption |
| power method | Titan X via nvidia-smi; CPU socket+DRAM via pcm-power; TK1 power meter scaled to AP+DRAM (15% AC-DC loss, 85% regulator efficiency, 15% peripherals) | Sec. 6.3 |

## 8. Results worth remembering
Table 1 (whole pipeline; errors are top-1 / top-5 %):
| network | ref err | compressed err | size ref -> compressed | ratio |
|---|---|---|---|---|
| LeNet-300-100 | 1.64 / - | 1.58 / - | 1070 KB -> 27 KB | 40x |
| LeNet-5 | 0.80 / - | 0.74 / - | 1720 KB -> 44 KB | 39x |
| AlexNet | 42.78 / 19.73 | 42.78 / 19.70 | 240 MB -> 6.9 MB | 35x |
| VGG-16 | 31.50 / 11.32 | 31.17 / 10.91 | 552 MB -> 11.3 MB | 49x |

Stage-by-stage ratios (Total rows of Tables 2-5): P only / P+Q / P+Q+H
- LeNet-300-100: 12x / 32x / 40x. LeNet-5: 12x / 33x / 39x.
- AlexNet: 9x (11% weights kept) / 27x (3.7%) / 35x (2.88%). Avg weight bits 5.4 -> 4 after H; index bits 4 -> 3.2.
- VGG-16: 13x (7.5% kept) / 31x (3.2%) / 49x (2.05%). Avg weight bits 6.4 -> 4.1; index bits 5 -> 3.1.
- Fig. 1 caption rounds pruning to "10x", P+Q "27x and 31x", P+Q+H "35x and 49x".

Per-layer highlights (Table 4 AlexNet, Table 5 VGG-16; Weights% kept after pruning):
- AlexNet conv1 84%, conv2 38%, conv3 35%, conv4 37%, conv5 37%; fc6 9%, fc7 9%, fc8 25%. fc6 has 38M of 61M weights.
- VGG-16 conv1_1 58% ... conv5_3 36%; fc6 4% (103M weights), fc7 4%, fc8 23%. fc6/fc7 compressed to 1.10% / 1.25% after P+Q+H.
- Shape fact: fc6/fc7 keep 4%-9%; fc8 keeps 23%-25%; CONV layers keep 22%-84% (Tables 4, 5).

Bit-width ablation, AlexNet (Table 6; CONV bits / FC bits -> top-1 err, top-5 err):
32/32 -> 42.78, 19.73; 8/5 -> 42.78, 19.70; 8/4 -> 42.79, 19.73; 4/2 -> 44.77, 22.33 (+1.99, +2.60 absolute points).
Fig. 7 text (Sec. 6.1): CONV accuracy drops significantly below 4 bits; FC not until 2 bits.

Pruning + quantization together (Fig. 6, Sec. 6.1): alone, each drops significantly below 8% of original
size; combined, 3% of original size with no loss. Pruned AlexNet has 6.7M weights to quantize vs 60M unpruned.

Centroid initialization (Fig. 8, Sec. 6.2): linear beats density and random "in all cases except at 3 bits";
x-axis 2-8 bits. (Fig. 8 legend prints "uniform init" for the linear method.)

Storage split (Fig. 11, printed percentages; weight / index / codebook):
AlexNet 48.6 / 51.4 / 0.1; VGGNet 51.8 / 48.1 / 0.1; LeNet-300-100 51.0 / 47.5 / 1.5; LeNet-5 41.0 / 53.8 / 5.2.

Speed and energy (Fig. 9, Fig. 10, batch 1, FC layers only, pruned vs dense, average):
- Speedup 3x CPU, 3.5x GPU, 4.2x mobile GPU (Fig. 9 caption). Energy 7x, 3.3x, 4.2x less (Fig. 10 caption).
- GPU utilization reported by nvidia-smi is 99% for both dense and sparse (Sec. 6.3).
- Why batch 1 only: MV has O(n^2) memory and O(n^2) compute (memory-bound); MM has O(n^2) memory vs O(n^3)
  compute, ratio 1/n, and blocking gives reuse (Sec. 6.3). VGG16 FC6 = 25088 x 4096 x 4 Bytes ~ 400MB, far
  beyond L3 (Sec. 6.3).
- Batched case: "pruned network no longer shows its advantage" (Sec. 6.3). In Table 8 at batch 64 the sparse
  layer is slower than the dense one in 16 of 18 layer x device cells; the exceptions are Tegra K1 FC7 of
  AlexNet (2056.8 dense vs 1372.8 sparse) and of VGG16 (2050.7 vs 660.2). All 6 Titan X cells are slower.

Table 8, time per sample (us). Titan X:
| layer | dense b1 | sparse b1 | dense b64 | sparse b64 |
|---|---|---|---|---|
| AlexNet FC6 | 541.5 | 134.8 | 19.8 | 94.6 |
| AlexNet FC7 | 243.0 | 65.8 | 8.9 | 51.5 |
| AlexNet FC8 | 80.5 | 54.6 | 5.9 | 23.2 |
| VGG16 FC6 | 1467.8 | 167.0 | 53.6 | 121.5 |
| VGG16 FC7 | 243.0 | 39.8 | 8.9 | 24.4 |
| VGG16 FC8 | 80.5 | 48.0 | 5.9 | 22.0 |
Core i7-5930k VGG16 FC6: dense b1 35022.8, sparse b1 3774.3, dense b64 1056.0, sparse b64 1780.3.
Tegra K1 VGG16 FC6: dense b1 35427.0, sparse b1 4377.2, dense b64 2001.4, sparse b64 8024.8.
Batch 1 exception: Core i7 VGG16 FC8 dense 774.2 vs sparse 777.3 (no gain). The page should compute any
ratio itself from these raw values (our reading).

Context numbers: FC layers are more than 90% of model size; up to 38% of Fast R-CNN computation time is in FC
layers (Sec. 6.3). 45nm energy: 32-bit float add 0.9 pJ, 32-bit SRAM access 5 pJ, 32-bit DRAM access 640 pJ;
1G connections at 20 fps -> 12.8 W of DRAM access (Sec. 1).

Toy shape facts:
- Fig. 3 example: 4x4 FP32 weights -> 4 clusters, 2-bit indices; centroids 2.00, 1.50, 0.00, -1.00 become
  1.96, 1.48, -0.04, -0.97 after one fine-tuning step. Ratio 3.2 (Sec. 3).
- Fig. 2 example: 3-bit relative index; a gap larger than 8 forces a filler zero.
- Fig. 4: conv3 of pruned AlexNet has a bimodal weight distribution (pruning removed the middle); example uses
  13 clusters. Fig. 5: quantized-weight and index-difference distributions are skewed; index differences are
  "rarely above 20" (Sec. 4).

## 9. Limitations stated
- The quantized, weight-shared model was never timed: cuSPARSE and MKL SPBLAS lack indirect lookup and
  relative-index CSR/CSC (Sec. 6.3, Sec. 8). Suggested fixes: custom GPU kernels or a custom ASIC (EIE, cited
  in Sec. 1 as later work).
- Speedup claims are for FC layers at batch 1 only. With batching the pruned layer loses (Sec. 6.3, Table 8).
- Unstructured sparsity speed on GPU tensor hardware is addressed later by 2:4 structured sparsity (Mishra 2021,
  see mishra2021nm notes).

## 10. Outline feedback
- ch-prune "Sourced: Han Tables 1-4": the VGG-16 per-layer table is Table 5; the bit-width ablation is
  Table 6; the method comparison is Table 7; timing is Table 8 (Appendix A); power is Table 9.
- ch-prune "why unstructured is slow on GPUs (Han's own reasoning about EIE hardware + Blalock)": the paper's
  own evidence is not EIE. Use Sec. 6.3 (MV vs MM memory/compute argument; speedup only at batch 1), Table 8
  (sparse slower than dense at batch 64 on Titan X, i7 and TK1), and Sec. 8 (quantized model not benchmarkable
  with cuSPARSE/MKL). Stronger secondary sources: Li 2017 Sec. 3.1 ("sparse convolutional kernels ... hard to
  accelerate"), Mishra 2021 Sec. 3.1 (CSR metadata up to 200%, poor cache-line use). Blalock measures no
  hardware; it only says FLOPs/params are a loose proxy for latency.
- "Han Eq. 3" for the shared-weight gradient is correct (PDF numbering). There is no magnitude-criterion
  equation in this paper; cite Han 2015 Sec. 4 for the threshold rule.
- viz-dc: the "bits per weight" readout should note the paper's own index-bit inconsistency (Sec. 2 says 8/5,
  AlexNet uses 4) or quote only table values.
- On a detector: the paper's speed case is FC-heavy, batch-1 (Fast R-CNN FC layers). Modern detectors are
  conv-dominated, so these speedups do not transfer directly (our reading).
