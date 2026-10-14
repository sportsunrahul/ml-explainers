# frantar2023 -- GPTQ (SPINE)

## 1. Identity
- Title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- arXiv 2210.17323, read v2. v1 submitted 2022-10-31; v2 (read) 2023-03-22 (abs.html dateline).
- Authors: Elias Frantar (first) ... Dan Alistarh (last); also Saleh Ashkboos, Torsten Hoefler.
- Venue: ICLR 2023 (abs.html Comments; PDF header "Published as a conference paper at ICLR 2023").
- Repo: https://github.com/IST-DASLab/gptq (abstract). No README in `sources/frantar2023/`, so nothing here is code-only.
- Source used: `src/gptq.tex` (matches the PDF; `src/arxiv.tex` is a different layout of the same paper, not used).

## 2. Core problem
GPT-scale models (OPT-175B, BLOOM-176B) need 326GB in FP16 (Sec. 1), so inference needs several GPUs. Accurate
post-training quantizers (AdaRound, BRECQ, OBQ) take about 1 hour for a 25M-parameter ResNet-50 and do not
scale to billions of parameters. Round-to-nearest (RTN) scales but fails below 8 bits.

## 3. What prior methods did and why that hurt
- RTN (ZeroQuant, LLM.int8(), nuQmm): picks a granularity, then rounds each weight to the nearest level. Works
  at 8 bit; at 4 bit OPT-175B goes 8.34 -> 10.54 WikiText2 PPL, at 3 bit it collapses (7.3e3) (Table 3).
- ZeroQuant-LKD (layer-wise distillation): 3 hours for a 1.3B model on an A100; extrapolates to "several hundred
  hours (a few weeks)" for 175B (Sec. 5, Runtime).
- OBQ (frantar2022obc): greedy weight-by-weight, exact OBS update per row. Cost O(d_row * d_col^3): cubic in the
  input dimension, so too expensive at billions of parameters (Sec. 3).

## 4. The one key idea
Quantize the weight matrix one column at a time, in the same fixed order for every row, and after each column
push its rounding error onto the columns not yet quantized using the inverse Hessian H^-1 of the layer inputs
(H = 2 X X^T). Because the order is shared, H^-1 is the same for all rows, so the whole sequence of inverse
information comes from one Cholesky factorization.

## 5. Mechanism, step by step
1. Run calibration data (128 random 2048-token C4 segments) through the model one Transformer block (6 layers)
   at a time (Sec. 5, Setup).
2. For each linear layer, accumulate H = 2 X X^T from its inputs X (d_col x m). Add dampening lambda = 1% of the
   mean diagonal of H (Step 3).
3. Fix the quantization grid before the process: standard uniform per-row asymmetric min-max grid (Setup;
   Sec. 3 "the quantization grid ... is fixed before the process"). With grouping, each group's grid is set
   when the column loop reaches it, from the current (already updated) weights (Sec. 5, paragraph under "175 Billion Parameter Models").
4. Compute H^-1, then replace it with its upper Cholesky factor: `H^-1 <- Cholesky(H^-1)^T` (Algorithm 1).
5. Loop over columns left to right (Step 1: arbitrary fixed order; all rows share it). For column j:
   quantize `Q[:, j] = quant(W[:, j])`; error `E[:, j-i] = (W[:, j] - Q[:, j]) / [H^-1]_jj` where `[H^-1]` now
   means the Cholesky factor; update the rest of the block `W[:, j:(i+B)] -= E[:, j-i] * H^-1[j, j:(i+B)]`.
6. After each block of B = 128 columns, apply the batched update to all later columns
   `W[:, (i+B):] -= E * H^-1[i:(i+B), (i+B):]` (Step 2, lazy batch updates). This changes memory traffic, not
   the result: "The final rounding decisions for column i are only affected by updates performed on this very
   column" (Step 2).
7. Send the block inputs through the now-quantized block to get inputs for the next block (Setup). Forward
   passes only; no gradient anywhere.

Traps for the `viz-gptq` toy (our reading of Algorithm 1 and Step 3):
- After line 3 of Algorithm 1 every `[H^-1]_jj` and `H^-1_{j, ...}` is the Cholesky factor U (H^-1 = U^T U, U
  upper), not the inverse. Step 3 says the factor's row q equals the Eq. 3 row divided by
  `([H^-1_{F_q}]_qq)^{1/2}`. The error line divides by U_jj, so `E * U[j, j:]` = `(w - quant(w)) / [H_F^-1]_qq *
  (H_F^-1)_{q, :}`, which is exactly the Eq. 2 update. Using the plain inverse in Algorithm 1 is wrong.
- For an 8x16 toy, B = 128 means one block. A smaller B for display gives the same numbers (Step 2); label it.
- Per-row asymmetric min-max grid: the paper does not print the formula. Our reading:
  scale = (max - min) / (2^b - 1), zero = round(-min / scale), quant(w) = scale * (clamp(round(w / scale) + zero,
  0, 2^b - 1) - zero). Compare GPTQ and RTN on the SAME grid (Baselines paragraph).
- Readout: ||W X - Q X||_2^2 (Eq. 1) for GPTQ vs RTN.
- The "act-order"/reorder heuristic is not in this paper (AWQ calls it "GPTQ-R"; see lin2024). Toy uses natural
  column order.

## 6. Equations (from `src/gptq.tex`; no custom macros needed beyond `\mathbf`, `\boldsymbol`)
Eq. 1 (`eq:layerwise-quantization`, PDF Eq. 1):
```
\text{argmin}_{\mathbf{\widehat{W}}} \, ||\mathbf{W} \mathbf{X} - \mathbf{\widehat{W}} \mathbf{X}||_2^2.
```
Paper: find quantized weights "which minimizes the squared error, relative to the full precision layer output".
Gloss: W = layer weights (d_row x d_col); X = layer inputs for m calibration points (d_col x m); W-hat = quantized
weights on a fixed grid; the norm is the squared Frobenius norm of the output difference.

Eq. 2 (`eq:obs-quant`, PDF Eq. 2):
```
w_q = \text{argmin}_{w_q} \, \frac{(\text{quant}(w_q) - w_q)^2}{[\mathbf{H}_F^{-1}]_{qq}}, \quad \boldsymbol{\delta}_F = - \frac{w_q - \text{quant}(w_q)}{[\mathbf{H}_F^{-1}]_{qq}} \cdot (\mathbf{H}_F^{-1})_{:, q}.
```
Paper: "the greedy-optimal weight to quantize next ... and the corresponding optimal update of all weights in F".
Gloss: F = set of weights still in full precision; H_F = 2 X_F X_F^T (Hessian of the row objective restricted to
F); quant(w) = nearest grid value; [H_F^-1]_qq = diagonal entry q of the inverse; delta_F = change applied to
all remaining weights to compensate. GPTQ keeps the update and drops the argmin (fixed order instead).

Eq. 3 (`eq:inv-update`, PDF Eq. 3), verbatim, including a source typo (`_{-p}` where `_{-q}` is meant):
```
\mathbf{H}_{-q}^{-1} = \Big(\mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}_{:, q} \mathbf{H}^{-1}_{q, :} \Big)_{-p}.
```
Paper: "removing the qth row and column of H ... directly in the inverse via one step of Gaussian elimination".
Gloss: new inverse after weight q leaves F, without re-inverting.

Eq. 4-5 (unlabelled / `eq:group-inv-update`, PDF Eq. 4 and 5), block versions:
```
\boldsymbol{\delta}_F &= -(\mathbf{w}_Q - \text{quant}(\mathbf{w}_Q))([\mathbf{H}_F^{-1}]_{QQ})^{-1} (\mathbf{H}_F^{-1})_{:, Q}, \\
\mathbf{H}_{-Q}^{-1} &= \Big(\mathbf{H}^{-1} - \mathbf{H}^{-1}_{:, Q} ([\mathbf{H}^{-1}]_{QQ})^{-1} \mathbf{H}^{-1}_{Q, :} \Big)_{-Q}.
```
Gloss: Q = a set of indices (a block of columns); same as Eq. 2-3 for many weights at once.

Cholesky reformulation (Step 3, no equation number): only row q of H_{F_q}^-1 from the diagonal onward is needed;
the sequence of Eq. 3 row removals "essentially corresponds to taking a Cholesky decomposition, except for the
minor difference that the latter divides row q by ([H^-1_{F_q}]_qq)^{1/2}".

Algorithm 1 (`alg:gptq`), verbatim:
```
\caption{Quantize $\mathbf{W}$ given inverse Hessian $\mathbf{H}^{-1} = (2 \mathbf{X} \mathbf{X}^\top + \lambda \mathbf{I})^{-1}$ and blocksize $B$.}
\STATE $\mathbf{Q} \gets \mathbf{0}_{d_\text{row} \times d_\text{col}}$ \quad \COMMENT{quantized output}
\STATE $\mathbf{E} \gets \mathbf{0}_{d_\text{row} \times B}$ \quad \COMMENT{block quantization errors}
\STATE $\mathbf{H}^{-1} \gets \text{Cholesky}(\mathbf{H}^{-1})^\top$ \COMMENT{Hessian inverse information}
\FOR {$i = 0, B, 2B, \dots$}
    \FOR {$j = i, \dots, i + B - 1$}
        \STATE $\mathbf{Q}_{:, j} \gets \text{quant}(\mathbf{W}_{:, j})$ \quad \COMMENT{quantize column}
        \STATE $\mathbf{E}_{:, j - i} \gets (\mathbf{W}_{:, j} - \mathbf{Q}_{:, j}) \, / \, [\mathbf{H}^{-1}]_{jj}$ \COMMENT{quantization error}
        \STATE $\mathbf{W}_{:, j:(i + B)} \gets \mathbf{W}_{:, j:(i + B)} - \mathbf{E}_{:, j - i} \cdot \mathbf{H}^{-1}_{j, j:(i + B)}$ \COMMENT{update weights in block}
    \ENDFOR
    \STATE $\mathbf{W}_{:, (i + B):} \gets \mathbf{W}_{:, (i + B):} - \mathbf{E} \cdot \mathbf{H}^{-1}_{i:(i + B), (i + B):}$ \COMMENT{update all remaining weights}
\ENDFOR
```

Complexity (Step 1): OBQ O(d_row * d_col^3) -> GPTQ O(max{d_row * d_col^2, d_col^3}), "by a factor of
min{d_row, d_col}".

## 7. Config / setup
| item | value | loc |
|---|---|---|
| Objective | layer-wise squared output error, Eq. 1 | Sec. 3 |
| Hessian | H = 2 X X^T (+ lambda I) | Sec. 3; Alg. 1 caption |
| Dampening | lambda = 1% of the average diagonal value of H, "always" | Sec. 4 Step 3 |
| Block size | B = 128 columns | Sec. 4 Step 2 |
| Order | fixed, same for all rows (arbitrary order insight) | Sec. 4 Step 1 |
| Grid | uniform per-row asymmetric min-max, fixed before the process | Sec. 5 Setup; Sec. 3 |
| Bits | 4 and 3 (main); 2-bit and ternary with groups | Tables 3-5, 7 |
| Grouping | g1024 (~0.02 extra bits), g128 (~0.15 extra bits); 2-bit g128 ~2.2 bit, g32 ~2.6 bit; ternary with g8 | Sec. 5 |
| Calibration | 128 random 2048-token segments from C4 (generic web text; no task data) | Sec. 5 Setup |
| Processing unit | one Transformer block (6 layers) at a time; next block sees outputs of the quantized block | Sec. 5 Setup |
| Hardware for quantizing | single NVIDIA A100 80GB, for all models incl. 175B | Sec. 5 Setup |
| Kept in FP16 | embeddings and output layer (for the 3-bit OPT-175B memory figure) | Sec. 5 Practical Speedups |
| Activations | not quantized (W-only; FP16 activations) | Sec. 1, Sec. 6 |
| Gradients | none; forward passes and linear algebra only | our reading |

## 8. Results worth remembering
- Runtime, single A100 (Table 2): OPT 13B 20.9m, 30B 44.9m, 66B 1.6h, 175B 4.2h; BLOOM 1.7B 2.9m, 3B 5.2m,
  7.1B 10.0m, 176B 3.8h. Abstract: "approximately four GPU hours".
- OPT-175B / BLOOM-176B (Table 5; columns Wiki2, PTB, C4 PPL, LAMBADA acc):
  - Baseline 16 bit: OPT 8.34 / 12.01 / 10.13 / 75.59; BLOOM 8.11 / 14.59 / 11.71 / 67.40.
  - RTN 4: OPT 10.54 / 14.22 / 11.61 / 71.34; BLOOM 8.37 / 15.00 / 12.04 / 66.70.
  - GPTQ 4: OPT 8.37 / 12.26 / 10.28 / 76.80; BLOOM 8.21 / 14.75 / 11.81 / 67.71.
  - RTN 3: OPT 7.3e3 / 8.0e3 / 4.6e3 / 0; BLOOM 571. / 107. / 598. / 0.17.
  - GPTQ 3: OPT 8.68 / 12.68 / 10.67 / 76.19; BLOOM 8.64 / 15.57 / 12.27 / 65.10.
  - GPTQ 3/g1024: OPT 8.45 / 12.48 / 10.47 / 77.39; BLOOM 8.35 / 15.01 / 11.98 / 67.47.
  - GPTQ 3/g128: OPT 8.45 / 12.37 / 10.36 / 76.42; BLOOM 8.26 / 14.89 / 11.85 / 67.86.
- Text (Sec. 5): GPTQ "loses only 0.03 perplexity at 4-bit on the 175B model, while RTN drops 2.2 points".
- OPT WikiText2 full family (Table 3) e.g. 13B: full 10.13, RTN4 11.32, GPTQ4 10.31, RTN3 3.4e3, GPTQ3 11.61.
  OPT-66B outlier: RTN4 110, GPTQ3 14.16 (footnote: dead units in early layers).
- 2-bit (Table 7, WikiText2): OPT-175B FP16 8.34, g128 9.58, g64 9.18, g32 8.94, 3-bit 8.68; BLOOM 8.11, 9.55,
  9.17, 8.83, 8.64. Ternary (g8) OPT-175B 9.20 (Sec. 5 text).
- Vision row (Table 1, ImageNet top-1, same setup as OBC; FP: RN18 69.76, RN50 76.13; columns 4bit, 3bit):
  AdaRound RN18 69.34 / 68.37, RN50 75.84 / 75.14; AdaQuant 68.12 / 59.21, 74.68 / 64.98; BRECQ 69.37 / 68.47,
  75.88 / 75.32; OBQ 69.56 / 68.69, 75.72 / 75.24; GPTQ 69.37 / 67.88, 75.71 / 74.87. GPTQ takes "< 1 minute
  rather than ~1 hour" (Sec. 5). Weight-only quantization; granularity per OBC setup (asymmetric per-channel).
- OBQ vs GPTQ (Table 8, appendix): BERT-base F1 88.53 FP; OBQ 88.23 / 85.29, GPTQ 88.18 / 86.02 (4/3 bit);
  OPT-125M PPL 27.66 FP; OBQ 32.52 / 69.32, GPTQ 31.12 / 53.85.
- Memory (Sec. 5, Practical Speedups): 3-bit OPT-175B ~63GB incl. FP16 embeddings and output layer; KV cache
  ~9GB at 2048 tokens; fits one 80GB A100 ("the model would not fully fit using 4 bits"). FP16 needs 5x80GB
  GPUs; LLM.int8() needs 3.
- Speed (Table 6; OPT-175B 3-bit, batch size 1, average per-token latency generating sequences of length 128;
  HuggingFace-accelerate layer split; custom quantized-matrix x FP16-vector kernel):
  A6000 48GB: FP16 589ms -> 3bit 130ms, 4.53x, GPUs 8 -> 2. A100 80GB: 230ms -> 71ms, 3.24x, GPUs 5 -> 1.
  Abstract/Sec. 1 round these to "around 3.25x" (A100) and "4.5x" (A6000).
- Large batch (App. A.2.2): decompressing the matrix first takes < 1.5ms on A100 and < 3ms on A6000 vs
  76ms / 365ms for the OPT-175B FC2 layer at batch 16 x 1024 tokens.

## 9. Limitations stated
- Speedups come from less memory movement only; no compute reduction; no FP16 x INT4 hardware support (Sec. 1,
  Sec. 6).
- No activation quantization (Sec. 1, Sec. 6).
- Kernels target batch size 1 (memory-bound matrix-vector); for large batch "our kernels thus not directly
  applicable" (App. A.2.2).
- Later fix from given sources: AWQ (lin2024) argues GPTQ's reconstruction overfits the calibration set and
  needs a reorder trick for LLaMA-7B and OPT-66B; AWQ Fig. 8(b) shows GPTQ +2.33 / +4.89 PPL under calibration
  shift vs AWQ +0.60 / +0.50. AWQ Table 9 combines both (AWQ+GPTQ) at INT2.

## 10. Outline feedback
- "GPTQ 175B in ~4 GPU hours": Table 2 gives 4.2h (OPT-175B) and 3.8h (BLOOM-176B) on one A100 80GB. Quote these.
- "175B on a single 80GB A100" is the 3-bit result; the paper says 4-bit would not fully fit. The chapter is
  titled "four-bit"; word it so the single-GPU claim is attached to 3 bit.
- Equation list is right: Eq. 1 objective, Eq. 2 OBQ/OBS update, Eq. 3 inverse update, Eq. 4-5 block versions.
  The Cholesky reformulation has no number; cite "Sec. 4 Step 3 and Algorithm 1". Note the `_{-p}` typo in Eq. 3.
- `viz-gptq` recall of `viz-adaround`: GPTQ Table 1 compares GPTQ with AdaRound/BRECQ/OBQ on ResNet18/50 at
  4/3 bit, a direct vision row for that callback.
- Detector relevance (our reading): GPTQ speedups are for batch-1 memory-bound matrix-vector products. A
  detector backbone at normal batch and resolution is closer to compute-bound, so W4 alone saves memory, not
  time (App. A.2.2 says the kernels are "not directly applicable" to large batch). Useful for ch-guide.
