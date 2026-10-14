# Video memory background: STM -> XMem (+ Cutie)

Scope: short background for ch-sam2 (memory bank, memory attention). Architecture lens, light.

---------------------------------------------------------------------------------------------------
## STM -- Video Object Segmentation using Space-Time Memory Networks

### 1. Identity
- Oh (Seoung Wug, first), Lee, Xu, Kim (Seon Joo, last).
- arXiv 1904.00607v2 (v1 2019-04-01). Venue: ICCV 2019 (abs.html comment).

### 2. Core problem
Semi-supervised VOS: the first frame's mask is given; segment the object in every later frame. Cues get richer as more frames are segmented, but prior methods used only the first and/or previous frame.

### 3. Prior methods and why that hurt (Sec. 1, Fig. 1)
- Previous-frame propagation: handles appearance change, drifts and fails on occlusion.
- First-frame detection: robust to drift, fails when appearance changes.
- First + previous (RGMP): better, but still only two frames.
- Online fine-tuning methods are slow.

### 4. Key idea
Treat past frames with masks as an external memory. The query frame reads it with dense space-time attention.

### 5. Mechanism
1. Memory encoder: RGB + mask (single-channel probability map) -> 4-channel ResNet-50, res4 features -> key (C/8) and value (C/2) maps per memory frame. Stack T frames: k^M in R^{T x H x W x C/8}, v^M in R^{T x H x W x C/2} (Sec. 3.1).
2. Query encoder: RGB only -> k^Q in R^{H x W x C/8}, v^Q in R^{H x W x C/2}.
3. Space-time memory read (Eq. 1-2): every query pixel attends to every memory pixel at every stored time step; the weighted memory value is concatenated with the query value.
4. Decoder with refinement modules and skip connections -> mask (Sec. 3.3).
5. Which frames are stored (Sec. 3.6, "Inference"): first frame and previous frame always; plus "we simply save a new memory frame every N frames", N = 5 by default.
6. Training (Sec. 3.5): pretrain on 3-frame synthetic clips from static images (random affine); then 3 sampled frames per video, max skip grown from 0 to 25 (curriculum).

### 6. Equations (final.tex, unlabelled; printed as (1) and (2))
- Eq. 1: `\mathbf{y}_i = \big[\mathbf{v}^Q_i, \;  \frac{1}{Z}\sum_{\forall j}{f(\mathbf{k}^Q_i,\mathbf{k}^M_j)\mathbf{v}^M_j}\big],`
  with `Z = \sum_{\forall j}f(\mathbf{k}^Q_i,\mathbf{k}^M_j)`.
  Paper: "i and j are the index of the query and the memory location, ... [.,.] denotes the concatenation."
  Gloss: y_i = output at query pixel i; v^Q_i = the query's own value; the sum = softmax-weighted average of memory values over all memory pixels j (all frames, all positions).
- Eq. 2 `\label{Eq:similarity}`: `f(\mathbf{k}^Q_i,\mathbf{k}^M_j) = \text{exp}(\mathbf{k}^Q_i\circ\mathbf{k}^M_j)`, "where o denotes the dot-product." So Eq. 1 is dot-product attention without the 1/sqrt(d) scale (our reading).

### 7. Config
| item | value | loc |
|---|---|---|
| backbone | ResNet-50 res4, both encoders | Sec. 3.1 |
| key / value channels | C/8 / C/2 | Sec. 3.1 |
| memory rule | first + previous + every N = 5 frames | Sec. 3.6 |
| speed | 0.16 s/frame, DAVIS 2016 val | abstract, Table 2 |

### 8. Results
- DAVIS 2017 val (+YouTube-VOS training): J 79.2, F 84.3 (Table 3). STM prints no J&F mean. XMem Table 2 lists STM DAVIS 2017 J&F 81.8.
- DAVIS 2016 val (+YV): J 88.7, F 89.9 (Table 2). YouTube-VOS 2018 val overall 79.4 (Table 1, abstract).

### 9. Limitations
Memory grows with every stored frame (one frame per N). XMem Sec. 1: such methods "usually struggle to handle videos longer than a minute on consumer-grade hardware." -> XMem.

---------------------------------------------------------------------------------------------------
## XMem -- Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

### 1. Identity
- Cheng (Ho Kei, first), Schwing (last). arXiv 2207.07115v2 (v1 2022-07-14). ECCV 2022 (abs.html). Repo: hkchengrex.github.io/XMem.

### 2. Core problem
With one feature memory type, memory use and accuracy are tied: store more frames -> better but GPU memory explodes on long videos; store fewer -> worse (Sec. 1, Fig. 1).

### 3. Prior methods
STM-style memory banks grow linearly with stored frames; they "struggle to handle videos longer than a minute on consumer-grade hardware" (Sec. 1). Eager compression (AFB-URR) loses accuracy.

### 4. Key idea
Three memory stores, after the Atkinson-Shiffrin model: sensory (every frame), working (high-res, recent), long-term (compact prototypes), plus consolidation from working to long-term.

### 5. Mechanism (Sec. 3)
1. Sensory memory: hidden state h_t (C^h = 64 channels), a GRU updated every frame from decoder features; "deep update" from the value encoder every r-th frame (Sec. 3.5).
2. Working memory: STCN-style key/value per stored frame; a new frame is added every r-th frame (r = 5 by default) (Sec. 3.4).
3. Long-term memory: when working memory reaches T_max frames, keep the first frame and the most recent T_min - 1; the rest are candidates. Pick top-P keys by usage (cumulative attention), "potentiate" their values by attention over all candidates, append to long-term (Sec. 3.3). Defaults T_min = 5, T_max = 10, P = 128, compression ratio 6328% (Sec. 4).
4. When long-term memory is full: evict least-frequently-used elements. Default max long-term size 10,000 elements, about 3,900 frames at r = 10 (App.). With 6GB GPU memory: up to 34,000 frames before memory issues (Sec. 3.3).
5. Read (Eq. 1-2): query attends over working + long-term keys with an "anisotropic L2" similarity; sensory memory goes to the decoder directly.

### 6. Equations (methods.tex; printed (1)-(3))
- Eq. 1 `\label{eq:readout}`: `\mathbf{F} = \mathbf{v}\mathbf{W(\mathbf{k}, \mathbf{q})}.` W = softmax over the memory dimension of S; k, v = keys/values of N = THW + L memory elements; q = query (C^k x HW).
- Eq. 2 `\label{eq:similarity}`: `\mathbf{S}(\mathbf{k}, \mathbf{q})_{ij} = -\mathbf{s}_i \sum_{c}^{\C{k}}{ \mathbf{e}_{cj} \left( \mathbf{k}_{ci} - \mathbf{q}_{cj} \right)^2 },` s_i = shrinkage (confidence of memory element i, in [1, inf)); e_cj = selection weight of channel c for query j (in [0,1]). Macro `\C{k}` = C^k.
- Eq. 3: `\mathbf{v}^{\text{p}} = \mathbf{v}^{\text{c}}\mathbf{W}(\mathbf{k}^\text{c}, \mathbf{k}^\text{p}).` Prototype values = attention-weighted average of candidate values.

### 7. Config (Sec. 3.6)
Query encoder ResNet-50, value encoder ResNet-18, stride 16; C^k = 64, C^v = 512, C^h = 64. Training ~35 hours on two RTX A6000; 8-frame sequences.

### 8. Results
DAVIS 2017 val J&F: XMem 86.2; XMem* (BL30K pretraining) 87.7 (Table 2).

### 9. Limitations
Pixel-level matching is noisy with distractors -> Cutie (object-level memory).

---------------------------------------------------------------------------------------------------
## Cutie (one line)
arXiv 2310.12982v2 (v1 2023-10-19), CVPR 2024 Highlight; Cheng ... Schwing. Adds object-level memory: a small set of object queries (N; "8 queries are sufficient", Sec. 4) and an object memory read top-down through a query-based object transformer, on top of XMem-style pixel memory; MOSE +8.7 J&F over XMem (abstract). DAVIS 2017 val J&F: Cutie-base 88.8 (Table 1, trained without MOSE).

## 10. Outline feedback (ch-sam2)
1. SAM 2's memory bank is STM-lineage: FIFO of recent frames + prompted frames, read by cross-attention. One line: "STM stored first + previous + every 5th frame; XMem added a compact long-term store; SAM 2 keeps a fixed FIFO and no long-term store" (the last clause from sam2.md, not checked here).
2. STM's read is attention with exp(dot product) and no scale; SAM 2 uses standard transformer cross-attention (check in sam2.md).
3. If a DAVIS 2017 number is shown for STM, use J 79.2 / F 84.3 from STM Table 3, or label 81.8 as "J&F as reported by XMem Table 2".
