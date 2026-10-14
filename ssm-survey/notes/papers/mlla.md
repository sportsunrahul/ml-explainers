# MLLA / MILA: "Demystify Mamba in Vision: A Linear Attention Perspective" (branch, analysis lens)

## 1. Identity
- arXiv 2405.16605, read **v2** (2024-12-02). v1 2024-05-26 (abs.html).
- Authors: Dongchen Han, ..., Gao Huang (corresponding). Tsinghua University, Alibaba Group (PDF p. 1).
- Venue: NeurIPS 2024 (abs.html comment; README "Accepted to NeurIPS 2024 as a poster").
- Repo: https://github.com/LeapLabTHU/MLLA. README: "Mamba-Like Linear Attention (MLLA) ... renamed as
  Mamba-Inspired Linear Attention (MILA) in the final manuscript". v2 text uses MILA.

## 2. Core problem
Mamba works well in vision; linear attention, which has the same linear cost, usually underperforms. Why?

## 3. Prior work (per the paper)
Linear attention (Katharopoulos et al.) replaces softmax with a kernel and normalization so
(QK^T)V = Q(K^T V), O(N); later works found it lacks expressive power (Sec. 1).

## 4. The claimed equivalence
Written in one form (Eq. 11 vs Eq. 12), selective SSM is single-head linear attention with:
h_i ~ S_i (state = K^T V memory), B_i ~ K_i^T, x_i ~ V_i, C_i ~ Q_i. Six distinctions:
input gate (Delta), forget gate (Abar), shortcut (D), no attention normalization, single-head, and
Mamba's block design. "Mamba can be seen as a variant of linear attention Transformer with specialized
linear attention and modified block design." (Sec. 4.4)

## 5. How tested
1. Baseline: Swin-T with softmax attention replaced by linear attention (Sec. 5.1).
2. Add or remove **one** distinction at a time; ImageNet-1k top-1 and throughput (Table 1).
3. Try replacing the forget gate with positional encodings (Table 2).
4. Combine the useful designs into MILA and compare with vision Mamba models (Table 3, COCO Table 4, ADE Table 5).

## 6. Equations (src/sec/4_method.tex; PDF eq. numbers (11), (12))
Macros from mathcommand.tex: `\vh`, `\vx`, `\vy` bold vectors; `\mA`, `\mB`, `\mC`, `\mD`, `\mQ`, `\mK`,
`\mV`, `\mS`, `\mZ` bold matrices.
Eq. (11), `\label{eq:compare_ssm_2}` (selective SSM):
```
\vh_i&=\widetilde{\mA}_i\odot \vh_{i-1}+\mB_i (\mathbf{\Delta}_i\odot \vx_i), \\
\vy_i&={\mC_i}\vh_i\ / \ 1+\mD\odot \vx_i.
```
Eq. (12), `\label{eq:compare_linear_2}` (causal linear attention):
```
\mS_i&=\mathbf{1} \odot\mS_{i-1}+{\mK_i^{\top}(\mathbf{1} \odot\mV_i)}, \\
\vy_i&={\mQ_i\mS_i}\ /\ {\mQ_i\mZ_i}+\mathbf{0}\odot \vx_i.
```
Gloss: A-tilde_i = exp(diag(A) Delta_i), elements in (0, 1) because A is negative diagonal (forget gate);
Delta_i = Softplus(x W_1 W_2) > 0 scales the input (input gate); D (.) x is the shortcut; linear attention
has forget gate 1, input gate 1, shortcut 0, and divides by Q_i Z_i (normalization). Shapes: h_i, S_i in
R^{d x C}; B_i in R^{d x 1}; C_i in R^{1 x d}.

## 7. Measurement setup
Swin-T-based, ImageNet-1k, 224^2; throughput and params/FLOPs reported per row (Table 1). Each distinction
added separately. Forget gate uses Mamba's hardware-aware scan (Sec. 5.2). Training details in the appendix
(not extracted).

## 8. Results
- Table 1 (top-1 / throughput img/s): Baseline 77.6 / 1152; + Input Gate 77.8 / 1069; **+ Forget Gate
  78.4 / 743**; + Shortcut 77.8 / 1066; - Normalization 72.4 / 1215; - Multi-head 73.5 / 1540;
  + Block Design all 79.4 / 915; **+ Block Design sub 80.9 / 1010**.
- The two credited factors: forget gate and block design ("the forget gate and block design tend to be the
  two core contributors", Sec. 1).
- Table 2 (forget gate replaced by position encoding): + APE 80.0; + LePE 81.6; + CPE 81.7; + RoPE 80.0, all
  at 1074-1132 img/s vs 743 for the forget gate. Paper: PE gives local bias and position information with a
  global (non-causal) receptive field.
- Forget gate values (Fig. 4b): about 0.2 in shallow layers, about 0.6-0.8 in deep layers.
- Table 3: MILA-T 83.5 (25M, 4.2G), MILA-S 84.4 (43M, 7.3G), MILA-B 85.3 (96M, 16.2G) vs VMamba-T 82.5
  (31M, 4.9G; this is the VMambaV9-T number, not VMamba v4's 82.6), VMamba-S 83.6, VMamba-B 83.9,
  MambaOut-T 82.7, Vim-S 80.3.
- Key number for the page: forget gate alone 77.6 -> 78.4 at 1152 -> 743 img/s; Mamba-style block
  ("sub") 80.9.

## 9. Limitations stated
Not extracted beyond the main text; the paper argues the forget gate's recurrence suits causal data and
"may not be as suitable for non-causal data like images" (Sec. 4.2).

## 10. Outline feedback
- The outline's one line ("behaves like linear attention with a forget gate") is incomplete. Suggested
  line: "Mamba's selective SSM is single-head linear attention plus an input gate (Delta), a forget gate
  (A-bar) and a skip (D), without normalization; in vision the forget gate and Mamba's block design carry
  the gain, and a position encoding can replace the forget gate (MLLA Tables 1-2)."
- Links to ch 8 (SSD): same Q/K/V reading of C/B/x. Consider a back-reference rather than re-deriving.
