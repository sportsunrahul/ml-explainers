# frantar2023sparsegpt -- SparseGPT (BRANCH)

## Identity
- Title: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
- arXiv 2301.00774, read v3. v1 2023-01-02; v3 (read) 2023-03-22 (abs.html).
- Authors: Elias Frantar (first), Dan Alistarh (last). Venue: not stated in abs.html or the PDF read; do not assert one.
- Repo: https://github.com/IST-DASLab/sparsegpt (abstract). Source: `src/arxiv.tex`; macro `\sparsegpt{}` = SparseGPT.

## Key idea (two sentences)
Pruning a layer is a sparse regression on the same layer-wise objective as GPTQ, and the OBS update makes the
reconstruction exact for a fixed mask; SparseGPT makes it affordable by sharing one sequence of inverse Hessians
across all rows (column-by-column, left to right, "Hessian synchronization"), with the same Cholesky / lazy-batch
(B = 128) machinery as GPTQ. The mask is chosen adaptively every B_s = 128 columns from the OBS error, so one pass
gives 50-60% unstructured sparsity (or 2:4 / 4:8 with B_s = m) without retraining, and it can quantize in the same
pass (Sec. 3).

## Defining equation (`eq:obs-update`, PDF Eq. 3, Sec. 3.1), verbatim
```
\boldsymbol{\delta_m} = - \frac{w_m}{[\mathbf{H}^{-1}]_{mm}} \cdot \mathbf{H}^{-1}_{:, m}, \quad \varepsilon_m = \frac{w_m^2}{[\mathbf{H}^{-1}]_{mm}}.
```
Gloss: removing weight m costs epsilon_m; delta_m moves the remaining weights to compensate. Algorithm 1 caption
uses H^-1 = (X X^T + lambda I)^-1 (no factor 2, unlike GPTQ); dampening 1% of the average diagonal (App. A).

## Setup facts
- Calibration: 128 2048-token segments from the first shard of C4, "follow frantar2022gptq" (Sec. 4).
- Hardware: single NVIDIA A100 80GB; 175B models in "approximately 4 hours" (Sec. 4); abstract says "under 4.5 hours".

## Key rows
- OPT WikiText2 PPL (Table 1; 2.7B / 6.7B / 13B / 30B / 66B / 175B):
  Dense 12.47 / 10.86 / 10.13 / 9.56 / 9.34 / 8.35; Magnitude 50% 265. / 969. / 1.2e4 / 168. / 4.2e3 / 4.3e4;
  SparseGPT 50% 13.48 / 11.55 / 11.17 / 9.79 / 9.32 / 8.21; 4:8 14.98 / 12.56 / 11.77 / 10.30 / 9.65 / 8.45;
  2:4 17.18 / 14.20 / 12.96 / 10.90 / 10.09 / 8.74.
  Note: dense OPT-175B is 8.35 here vs 8.34 in GPTQ Table 3. Do not mix the two.
- Joint 50% + 4-bit on OPT-175B: 8.29 vs GPTQ 3-bit 8.68 (same storage) (Sec. 4, Fig. 6 text).
- 2:4 GPU matmul speedups, OPT-175B shapes, 2048 tokens, CUTLASS vs cuBLAS (App. Table 8): Q/K/V/Out 2.84ms ->
  1.59ms 1.79x; FC1 10.26 -> 6.15 1.67x; FC2 10.23 -> 6.64 1.54x. GPU model not named (text: "Ampere and newer").

## Outline feedback
- One line for ch-gptq-awq: "SparseGPT reuses GPTQ's column-by-column OBS machinery to prune OPT-175B to 50% with
  WikiText2 PPL 8.35 -> 8.21 (Table 1)". The 2:4 row (8.74) is the one tied to GPU speed.
