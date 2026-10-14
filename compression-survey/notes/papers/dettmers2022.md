# dettmers2022 -- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

## 1. Identity
- Title: "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale".
- arXiv 2208.07339, version read: v2 (2022-11-10). v1 submitted 2022-08-15 (abs.html).
- Authors: Tim Dettmers ... Luke Zettlemoyer (4 authors: Dettmers, Lewis, Belkada, Zettlemoyer). UW, FAIR, Hugging Face, ENS Paris-Saclay.
- Venue: NeurIPS 2022, camera-ready (abs.html comments).
- Repo: https://github.com/TimDettmers/bitsandbytes (footnote, Sec 1). No README in sources/, so no code-only facts.
- Sources read: src/neurips_2022.tex (all), paper.pdf pages 2, 7, 8 as images for Figs 1, 3, 4.

## 2. Core problem
Feed-forward and attention projection matmuls hold about 95% of parameters and 65-85% of compute in transformers at and beyond 6.7B (Sec 1). Prior 8-bit methods degrade, need tuning, and were studied only below 350M parameters. Goal: load a 16/32-bit checkpoint of up to 175B, convert linear layers to Int8, run with no degradation, at half the memory.

## 3. What prior methods did and why that hurt (per the paper)
- Per-tensor absmax and zeropoint quantization: one scale per tensor, so "a single outlier can reduce the quantization precision of all other values" (Sec 3 para 1).
- Row-wise (fbgemm) and vector-wise: scale per row of X (per token). Outliers live in columns (feature dims h), so row scaling cannot isolate them (Sec 4.3).
- Zeropoint helps because outliers are mostly one-sided, but fails at 13B and needs a multiply_i16 instruction GPUs lack, so it is slow when unrolled (Sec 2.1, Eq 5).
- Table 1: every Int8 variant without decomposition has worse perplexity at 13B than at 6.7B.

## 4. The one key idea
The outliers sit in a handful of fixed hidden dimensions (at most 7 up to 13B, about 0.1% of dims). Pull those columns of X (and matching rows of W) out and multiply them in FP16; multiply the other 99.9% in Int8 with one scale per row of X and one per column of W.

## 5. Mechanism, step by step (Fig 2, p3)
1. Input X_f16 in R^{s x h} (s tokens, h features), W_f16 in R^{h x o}.
2. Find outlier set O = feature dims h with at least one |X| value > alpha = 6.0 (Sec 3.2). Done at inference time per input (our reading of "decomposed into sub-matrices of large magnitude features", Fig 2 caption).
3. Outlier part: X_f16[:, O] times W_f16[O, :] in FP16.
4. Regular part: row-wise absmax constants c_x (one per row of X, length s) and column-wise c_w (one per column of W, length o); quantize to Int8 in [-127, 127].
5. Int8 x Int8 -> Int32 matmul (cuBLASLt Int8 per App D).
6. Dequantize by the outer product c_x (x) c_w (Eq 7), i.e. divide entry (i, j) by c_x[i] * c_w[j].
7. Add the FP16 outlier product; output FP16.
Memory: weights stored in Int8, giving about 50% reduction vs 16-bit; 1.96x for BLOOM-176B (Sec 3 para 2).

## 6. Equations (TeX verbatim from src/neurips_2022.tex)
Custom macro used: `\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}` (mathtools). Equation numbers checked against paper.pdf.

**Absmax quantization (unnumbered display, Sec 2.1, p3)**
```
\mathbf{X}_{i8} = \Biggl{\lfloor}{\dfrac{127\cdot\mathbf{X}_{f16}}{\max\limits_{ij}(|\mathbf{X}_{f16_{ij}}|)}}\Biggr{\rceil}= \Biggl{\lfloor}{\dfrac{127}{\norm{\mathbf{X}_{f16}}_\infty}\mathbf{X}_{f16}}\Biggr{\rceil} = \Bigl{\lfloor}{s_{x_{f16}} \mathbf{X}_{f16}}\Bigr{\rceil},
```
Paper: scales inputs into [-127, 127] by multiplying with s_x = 127 / absolute max of the whole tensor ("equivalent to dividing by the infinity norm and multiplying by 127"). Gloss: X_i8 = Int8 tensor; X_f16 = FP16 input; max_ij |X_ij| = largest magnitude anywhere in the tensor; s_x = scale; the floor/ceil bracket = round to nearest.

**Zeropoint quantization (Eq 1-3, Sec 2.1, p3)**
```
nd_{x_{f16}} = \dfrac{2\cdot 127}{\max\limits_{ij}(\mathbf{X}_{f16}^{ij}) - \min\limits_{ij}(\mathbf{X}_{f16}^{ij})}      (1)
zp_{x_{i16}} = \bigl\lfloor{\mathbf{X}_{f16} \cdot \min\limits_{ij}(\mathbf{X}_{f16}^{ij})\bigr\rceil}                  (2)
\mathbf{X}_{i8} = \bigl\lfloor{{nd_{x_{f16}} \mathbf{X}_{f16}}\bigr\rceil}                                             (3)
```
Paper: "shifts the input distribution into the full range [-127, 127] by scaling with the normalized dynamic range nd_x and then shifting by the zeropoint zp_x". Gloss: nd_x = 254 / (max - min), the scale; zp_x = Int16 offset; X_i8 = scaled, rounded values. As written, Eq 2 multiplies X by min(X) and Eq 3 omits the shift. Our reading: transcription slips; the intended form is zp = round(nd_x * min(X)) and the shift is applied by the special instruction (Eq 4). Quote as written on the page if shown.

**Zeropoint multiply (Eq 4-5, p3)**
```
C_{i32}  = \text{multiply}_{i16}({A}_{zp_{a_{i16}}},{B}_{zp_{b_{i16}}}) = ({A}_{i8}+zp_{a_{i16}})({B}_{i8}+zp_{b_{i16}})      (4)
C_{i32}= {A}_{i8}{B}_{i8} + {A}_{i8}zp_{b_{i16}} + {B}_{i8}zp_{a_{i16}} + zp_{a_{i16}}zp_{b_{i16}},                     (5)
```
Paper: Eq 5 is the unrolled form needed "if the instruction multiply_i16 is not available such as on GPUs or TPUs"; zeropoint "can be slow". Gloss: A_i8, B_i8 = Int8 operands; zp = their zeropoints; C_i32 = Int32 accumulator; the three cross terms are the extra work.

**Int8 matmul with FP16 in/out, tensor-wise (Eq 6, p4)**
```
\mathbf{X}_{f16} \mathbf{W}_{f16} =\mathbf{C}_{f16}  & \approx \frac{1}{c_{x_{f16}}c_{w_{f16}}} \mathbf{C}_{i32} = S_{f16}\cdot\mathbf{C}_{i32}\\
& \approx S_{f16}\cdot\mathbf{A}_{i8}\mathbf{B}_{i8} = S_{f16}\cdot Q(\mathbf{A}_{f16})\phantom{.} Q(\mathbf{B}_{f16}),
```
Gloss: X in R^{s x h}, W in R^{h x o}; c_x, c_w = one scaling constant per tensor (s_x, s_w for absmax; nd_x, nd_w for zeropoint); S = 1/(c_x c_w) dequantizes; Q = absmax or zeropoint quantizer.

**Vector-wise quantization (Eq 7, Sec 3.1, p4)**
```
\mathbf{C}_{f_{16}} \approx \frac{1}{\mathbf{c}_{x_{f16}}\otimes \mathbf{c}_{w_{f16}}} \mathbf{C}_{i32} = S\cdot\mathbf{C}_{i32}= \mathbf{S}\cdot\mathbf{A}_{i8}\mathbf{B}_{i8} = \mathbf{S}\cdot Q(\mathbf{A}_{f16})\phantom{.} Q(\mathbf{B}_{f16}),
```
Paper: "matrix multiplication can be seen as a sequence of independent inner products ... assign a different scaling constant c_x to each row of X and c_w to each column of W"; denormalize by the outer product. Gloss: c_x in R^s (one per token row), c_w in R^o (one per output column); (x) = outer product; S = elementwise reciprocal of that s x o matrix. App B: equivalent to per-channel conv quantization plus row quantization.

**Mixed-precision decomposition (Eq 8, Sec 3.2, p5)**
```
\mathbf{C}_{f16} \approx \sum\limits_{h\in O} \mathbf{X}_{f16}^h \mathbf{W}_{f16}^h + \mathbf{S}_{f16} \cdot  \sum\limits_{h\not\in O} \mathbf{X}_{i8}^h \mathbf{W}_{i8}^h
```
Paper: O = {i | i in Z, 0 <= i <= h} "contains all dimensions of h which have at least one outlier with a magnitude larger than the threshold alpha"; "alpha=6.0 is sufficient to reduce transformer performance degradation close to zero"; Einstein notation, indices as superscripts. Gloss: h in O = outlier feature dims, computed in FP16; h not in O = the rest, in Int8 with vector-wise scales S_f16. |O| <= 7 up to 13B, so the FP16 part costs about 0.1% extra memory.

## 7. Setup
Method (inference):
| Item | Value | Loc |
|---|---|---|
| Layers quantized | FFN and attention projection linear layers; attention function (QK^T, softmax, PV) not Int8 | Sec 1, Sec 6 |
| Weight granularity | per output column (c_w), absmax | Sec 3.1 |
| Activation granularity | per token row (c_x), dynamic absmax | Sec 3.1; SmoothQuant Table 2 calls it "per-token dynamic+FP16" |
| Outlier threshold | alpha = 6.0, applied to |X| | Sec 3.2 |
| Calibration | none; "used immediately" | Abstract |
| Perplexity eval | fairseq dense LMs 125M-13B, C4 validation, NVIDIA A40 GPUs | Sec 3.3 |
| Zero-shot eval | OPT 125M-175B, EleutherAI harness: WinoGrande, HellaSwag, PIQA, LAMBADA (mean) | Sec 3.3, Fig 1 |

Emergence analysis (measurement setup, Sec 4.1, p6-7):
- Hidden state X_l in R^{s x h} at the input of attention projections (q/k/v/o) and FFN expansion layer; attention function and FFN contraction ignored ("feature outliers only occur in" those).
- Outlier feature: dimension h_i with at least one value of magnitude >= 6 (prose: "at least 6.0"), in the same h_i in >= 25% of layers, and in >= 6% of sequence dimensions.
- Threshold choice: 6 = where decomposition stops perplexity loss; 25% and 6% picked so the 125M model yields a single outlier; the second most common 125M outlier is in one layer (~2% of layers).
- Models: four GPT-2 (OpenAI), five fairseq (Meta), GPT-J 6B (EleutherAI, Mesh-TF); up to 13B. Inference frameworks: fairseq and Hugging Face. Sequence length 2048 for the 150k count.
- Effect test: zero outlier dims before attention projections, per layer independently (no cascade), compare top-1 softmax prob; also propagate to get perplexity; control = same number of random non-outlier dims.

## 8. Results worth remembering
- Table 1 (p6), C4 val perplexity, columns 125M / 1.3B / 2.7B / 6.7B / 13B:
  - 32-bit Float: 25.65 / 15.91 / 14.43 / 13.30 / 12.45
  - Int8 absmax: 87.76 / 16.55 / 15.11 / 14.59 / 19.08
  - Int8 zeropoint: 56.66 / 16.24 / 14.76 / 13.49 / 13.94
  - Int8 absmax row-wise: 30.93 / 17.08 / 15.24 / 14.13 / 16.49
  - Int8 absmax vector-wise: 35.84 / 16.82 / 14.98 / 14.13 / 16.48
  - Int8 zeropoint vector-wise: 25.72 / 15.94 / 14.36 / 13.38 / 13.47
  - Int8 absmax row-wise + decomposition: 30.76 / 16.19 / 14.65 / 13.25 / 12.46
  - Absmax LLM.int8() (vector-wise + decomp): 25.83 / 15.93 / 14.44 / 13.24 / 12.45
  - Zeropoint LLM.int8(): 25.69 / 15.92 / 14.43 / 13.24 / 12.45
- Fig 1 (p2), OPT mean zero-shot accuracy (plot, uncertain): 16-bit and LLM.int8() overlap from ~0.45 (125M) to ~0.76 (175B); 8-bit absmax vector-wise baseline tracks to 2.7B (~0.65), then ~0.45 at 6.7B, ~0.33 at 13B/30B, ~0.31 at 175B. This figure is the only in-paper evidence for "no degradation up to 175B"; Table 1 stops at 13B.
- Independent check: SmoothQuant Table 3 (OPT-175B) has LLM.int8() avg 66.7 vs FP16 66.9, WikiText 11.10 vs 10.99. So "zero degradation" is near-zero.
- Emergence (Sec 4.2, Table 4 p17), see claim card below.
- Memory, Table 2 (p10; duplicated as Table 3 p16): largest model runnable 8-bit vs 16-bit: 8x A100 80GB: OPT-175B/BLOOM both; 8x A100 40GB: OPT-175B/BLOOM vs OPT-66B; 8x RTX 3090 24GB: OPT-175B/BLOOM vs OPT-66B; 4x RTX 3090: OPT-66B vs OPT-30B; Colab Pro 15GB: OPT-13B vs GPT-J-6B; Colab 12GB: T0/T5-11B vs GPT-2 1.3B.
- Speed, Table 5 (p18): speedup vs FP16 matmul for the first FFN layer of GPT-3-sized models (hidden = 4x model dim). GPU not stated. Columns Small(768) / Medium(1024) / Large(1536) / XL(2048) / 2.7B(2560) / 6.7B(4096) / 13B(5140) / 175B(12288):
  - Int8 without overhead: 0.99 / 1.08 / 1.43 / 1.61 / 1.63 / 1.67 / 2.13 / 2.29
  - Absmax PyTorch+NVIDIA: 0.25 / 0.24 / 0.36 / 0.45 / 0.53 / 0.70 / 0.96 / 1.50
  - Vector-wise PyTorch+NVIDIA: 0.21 / 0.22 / 0.33 / 0.41 / 0.50 / 0.65 / 0.91 / 1.50
  - Vector-wise (custom kernels): 0.43 / 0.49 / 0.74 / 0.91 / 0.94 / 1.18 / 1.59 / 2.00
  - LLM.int8() (vector-wise+decomp): 0.14 / 0.20 / 0.36 / 0.51 / 0.64 / 0.86 / 1.22 / 1.81
  - Text (App D.1): raw Int8 in cuBLASLt "begins to be two times faster than cuBLAS at a model size of 5140"; with quantize/dequantize, 1.6x at 5140; decomposition leaves speedups only at 13B and 175B. Sec 3.4 says "about two times faster for large matrix multiplications equivalent to those in 175B models" (Table 5 LLM.int8() row says 1.81x; the 2.00x is vector-wise without decomposition).
- End-to-end, Table 6 (p18): BLOOM-176B per-token generation latency (ms), Hugging Face + accelerate, A100 80GB. Batch 1 / 8 / 32: bf16 8 GPUs 239 / 32 / 9.94; LLM.int8() 8 GPUs 253 / 34 / 10.44; 4 GPUs 246 / 33 / 9.40; 3 GPUs 247 / 33 / 9.11. "slightly slower but close".

## Claim card: emergent large-magnitude features (Sec 4, p6-9; Table 4 p17; Figs 3-4 p7-8)
- Claim: Transformers grow outlier features (|x| >= 6) concentrated in a few fixed hidden dims. By parameter count, they jump to all layers "suddenly between 6B and 6.7B" (layers affected 65% -> 100%, sequence dims 35% -> 75%; Sec 4.2 point 1). By C4 perplexity the rise is smooth and exponential (Fig 3b), and the outlier-dim count is strictly monotonic in perplexity while non-monotonic in size (Fig 4b). The paper concludes "model perplexity rather than mere model size determines the phase shift" (point 4). Median outlier magnitude jumps after emergence (Fig 4a), and that is "the core reason why quantization methods fail starting at the 6.7B scale" (point 3).
- Shape at 6.7B: ~150k outliers per 2048-token sequence in only 6 hidden dims (Sec 4.2); all 6 one-sided (Table 4). 13B: 7 dims, 6 one-sided, 100% layers, 73% seq dims. Sec 4 opener also attributes "up to 150k outliers per 2048 token sequence" to a 13B model with at most 7 dims (two attributions in the paper; do not merge).
- Magnitude: other dims "usually have a range of [-3.5, 3.5]"; outliers "3-20x larger" (Table 4 caption); intro says "up to 20x". Median quartiles of the most common outlier: GPT-J 6.0B (-21, -17, -14); 6.7B (-44, -40, -35); 13B (-63, -58, -45).
- Measurement: 10 models (Table 4 rows): GPT-2 117M, 345M, 762M, 1.5B; fairseq 125M, 1.3B, 2.7B, 6.7B, 13B; GPT-J 6.0B. Thresholds as in Sec 7 above. Effect: zero outlier dims -> mean top-1 softmax ~40% -> ~20%, perplexity +600-1000% (relative); control: zero 7 random dims -> top-1 drops 0.02-0.3%, perplexity +0.1%.
- Falsifiers: (a) zeroing random dims hurts as much as zeroing outlier dims (control says no); (b) a model past the shift whose outliers do not occupy all layers or spread over many dims; (c) a lower-perplexity model with fewer outlier dims (breaks the monotonic claim); (d) per-tensor or vector-wise Int8 holding full perplexity at 6.7B-13B (Table 1 says no).
- Caveats to show on the card: thresholds were tuned on the 125M model; analysis stops at 13B; "sudden" depends on the x-axis; text (65%, 35%) and Table 4 GPT-J row (62%, 28%) differ. Our reading: the 6B -> 6.7B step compares GPT-J (EleutherAI, Mesh-TF) with fairseq 6.7B, so a model-family change coincides with the jump.

## 9. Limitations the paper states (Sec 6)
- Int8 only; FP8 not studied because "current GPUs and TPUs do not support this data type" (2022). FP8 chapter: Micikevicius 2209.05433.
- Only up to 175B; larger scales may bring new emergent properties.
- Attention function not in Int8 (it has no parameters; needs other methods).
- Inference only; Int8 training/fine-tuning is initial (App E-F).
- Speed: overhead slows models below 6.7B (Sec 3.4, Table 5).
- Fixed later (from given sources): SmoothQuant (2211.10438) removes the FP16 decomposition and runs all GEMMs and BMMs in INT8 with off-the-shelf kernels; its Table 11 shows LLM.int8() slower than FP16 (237.1 vs 152.6 ms, OPT-13B seq 256) where SmoothQuant-O3 is 112.1 ms.
- Our reading: Sec 4.2 point 3 writes the FP16 max as 65535; the IEEE FP16 max is 65504. If quoted, quote as written.

## 10. Outline feedback (ch-outliers)
- "20x+ larger" in the intuition: Dettmers gives "up to 20x" / "3-20x" vs a [-3.5, 3.5] background; SmoothQuant gives "~100x" and ">70" (OPT-13B). Attribute each number to its paper; do not write a single "20x+".
- "Dettmers Eq. mixed-precision" is Eq 8; vector-wise is Eq 7; absmax is unnumbered (Sec 2.1).
- Claim card says "when outliers appear, by model size". The paper itself argues perplexity, not size, is the driver (Sec 4.2 points 2 and 4). Show both axes (Fig 3a vs 3b) on the card.
- "degradation-free up to 175B": source it to Fig 1 (OPT zero-shot, plot) plus SmoothQuant Table 3; Table 1 only reaches 13B. Say "near-zero" or quote the paper's wording.
- Add one line on speed: LLM.int8() is a memory method; it is slower than FP16 below 13B-sized layers (Table 5, GPU not stated) and only 1.22x/1.81x at 13B/175B sizes. This is the hinge to SmoothQuant.
- Toy shape facts from this paper: few dims (6-7 of thousands, ~0.1%), one-sided sign, present in ~75% of tokens (not 100%) at 6.7B. SmoothQuant Fig 4 shows ridges across all tokens; pick one and say which.
