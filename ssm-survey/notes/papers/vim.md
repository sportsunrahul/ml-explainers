# Vision Mamba (Vim)

## 1. Identity
- Title: "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model"
  (running head in v3 drops "Bidirectional").
- arXiv 2401.09417, read **v3** (2024-11-14). v1 2024-01-17, v2 2024-02-10 (abs.html submission history).
- Authors: Lianghui Zhu (equal), Bencheng Liao (equal), ..., Xinggang Wang (corresponding). HUST, Horizon Robotics, BAAI.
- Venue: ICML 2024 (PMLR 235; abs.html comment and PDF footer).
- Repo: https://github.com/hustvl/Vim
- Sources used: src/main.tex + src/sub_files/{1intro,3method,4exps}.tex (equations, tables), paper.pdf
  (figure values, page and equation numbers), README.md (code-only values).

## 2. Core problem
Mamba is a 1-D causal sequence model. Images need position awareness and global context. The paper asks
whether a pure-SSM backbone (no attention) can match ViT/DeiT and stay efficient at high resolution.

## 3. What prior methods did and why that hurt (per the paper)
- ViT/DeiT: global, data-dependent context through self-attention, but speed and memory become a problem
  at high resolution (Sec. 1). For detection at 1024^2 the DeiT baseline needs ViTDet-style 2D window
  attention, which "injects 2D prior and breaks the sequential modeling nature of Transformer" (Sec. 4.3).
- Earlier SSM vision work (2-D SSM, SGConvNeXt, ConvSSM, S4ND) is hybrid with CNN/Transformer or uses a
  global 2D convolution kernel; not data-dependent (Sec. 1, Sec. 5).
- Plain Mamba has two problems for vision: "unidirectional modeling and lack of positional awareness" (Sec. 1).

## 4. The one key idea
Treat the image as a ViT-style patch sequence with position embeddings, and run two selective SSMs per
block, one forward and one backward over the same sequence, sharing the input projection and the gate.

## 5. Mechanism, step by step (Vim-Ti numbers; shapes from Algorithm 1)
1. Patchify: image 224 x 224 x 3 (H, W, C), patch P = 16 (16x16 kernel projection, Sec. 3.4).
   J = (224/16)^2 = 196 patches, each P^2 C = 768 values (our reading: arithmetic).
2. Project: W in R^{768 x D}, D = 192 (Ti) / 384 (S). Patch tokens (196, D).
3. Class token: one learnable t_cls is inserted. Default is the **middle** of the sequence (Table 5;
   README news 2024-02-10: "By placing the class token at middle, Vim achieves improved results").
   M = J + 1 = 197 tokens. Index 98 is our reading of "middle" (the paper does not give the index).
   Note: Eq. 4 still writes t_cls first ([t_cls; t_p^1 W; ...]); Fig. 2 draws it at position 5 of 0..9.
4. Position embeddings: E_pos in R^{(J+1) x D} added to all tokens (Eq. 4). The paper does not say
   "learnable" explicitly; the README model name contains "abs_pos_embed" (code-only).
5. Stack L = 24 Vim blocks, each T_l = Vim(T_{l-1}) + T_{l-1}. Per block (Algorithm 1), with batch B:
   - T' = Norm(T_{l-1}): (B, M, D)
   - x = Linear^x(T'), z = Linear^z(T'): each (B, M, E); E = 384 (Ti) / 768 (S), E = 2D.
   - For o in {forward, backward}, each with its own weights:
     - x'_o = SiLU(Conv1d_o(x)): (B, M, E). Conv kernel width is not stated in the paper.
     - B_o = Linear^B_o(x'_o), C_o = Linear^C_o(x'_o): (B, M, N), N = 16.
     - Delta_o = softplus(Linear^Delta_o(x'_o) + Parameter^Delta_o): (B, M, E).
     - Parameter^A_o: (E, N). Abar_o = Delta_o (x) Parameter^A_o, Bbar_o = Delta_o (x) B_o: (B, M, E, N).
     - h_o: (B, E, N), zero-initialized; for i = 0..M-1: h_o = Abar_o[:, i] * h_o + Bbar_o[:, i] * x'_o[:, i, :, None];
       y_o[:, i] = h_o (x) C_o[:, i]. y_o: (B, M, E).
   - y'_f = y_f * SiLU(z), y'_b = y_b * SiLU(z) (same z gates both).
   - T_l = Linear^T(y'_f + y'_b) + T_{l-1}: (B, M, D).
   - How the backward branch reverses the sequence is not spelled out in Algorithm 1 (it shows only the
     loop over directions); our reading: reverse along M, scan, reverse back.
   - The README model name ends in "div2" (code-only); our reading, unverified: the two branch sums are halved.
6. Head: f = Norm(T_L^0) (the class token output), p_hat = MLP(f) (Sec. 3.2).
7. State per direction per layer (Vim-Ti): E x N = 384 x 16 = 6144 numbers per image (our reading).
8. Efficiency (Sec. 3.5): IO-aware kernel reads O(BME + EN) from HBM instead of O(BMEN); recompute the
   (B, M, E, N) states and activations in backward; cost Eq. 5 vs Eq. 6.
9. Trainable (our reading of Algorithm 1 / Eq. 4): W, E_pos, t_cls, Norms, Linear^x, Linear^z,
   Linear^T, and per direction Conv1d_o, Linear^{B,C,Delta}_o, Parameter^Delta_o, Parameter^A_o; MLP head.
   Signal: ImageNet-1k classification loss (label smoothing, mixup). A/Delta initialization is not stated.

Notation: Vim's E is the expanded **dimension** (384 for Ti). Mamba's E is the expansion **ratio** (2).
Vim's L is depth; Vim's M is sequence length; Vim's B is batch (and bold B is the SSM input matrix).

## 6. Equations (src/sub_files/3method.tex; numbers from PDF p. 4-5)
Macros: `\name{}` = "Vim"; `\mathtt` used for dimension letters.

Eq. (1), `\label{eq:zoh}` (ZOH):
```
\mathbf{\overline{A}} &= \exp{(\mathbf{\Delta}\mathbf{A})}, \\
\mathbf{\overline{B}} &= (\mathbf{\Delta} \mathbf{A})^{-1}(\exp{(\mathbf{\Delta} \mathbf{A})} - \mathbf{I}) \cdot \mathbf{\Delta} \mathbf{B}.
```
Paper: "zero-order hold (ZOH)" transforms continuous A, B to discrete Abar, Bbar using timescale Delta.
Gloss: Abar = per-step decay; Bbar = per-step input weight; Delta = step size. Page notation already.
Note: Algorithm 1 line 13-14 writes Abar = Delta (x) A and Bbar = Delta (x) B (no exp, first-order B);
our reading is that the kernel applies exp as in Mamba; the algorithm listing is schematic.

Eq. (2), `\label{eq:discrete_lti}`:
```
h_t &= \mathbf{\overline{A}}h_{t-1} + \mathbf{\overline{B}}x_{t}, \\
y_t &= \mathbf{C}h_t.
```
Gloss: state h (size N) decays by Abar and adds the new input; output y reads the state through C.

Eq. (3), `\label{eq:conv}`:
```
\mathbf{\overline{K}} &= (\mathbf{C}\mathbf{\overline{B}}, \mathbf{C}\mathbf{\overline{A}}\mathbf{\overline{B}}, \dots, \mathbf{C}\mathbf{\overline{A}}^{\mathtt{M}-1}\mathbf{\overline{B}}), \\
\mathbf{y} &= \mathbf{x} * \mathbf{\overline{K}},
```
Paper: "M is the length of the input sequence x, and Kbar in R^M is a structured convolutional kernel."
Only valid for LTI SSMs; Sec. 3.5 says the data-dependent SSM "is no longer equivalent to convolution".

Eq. (4), `\label{eq:embed}`:
```
\mathbf{T}_0 &= [\mathbf{t}_{cls};\mathbf{t}_p^1\mathbf{W};\mathbf{t}_p^2\mathbf{W};\cdots;\mathbf{t}_p^{\mathtt{J}}\mathbf{W}] + \mathbf{E}_{pos}, \\
```
Paper: t_p^j is the j-th patch, W in R^{(P^2 C) x D} the learnable projection, t_cls the class token,
E_pos in R^{(J+1) x D}. Gloss: the input token sequence x for layer 1. (Order shown is head-cls; the
default model puts t_cls in the middle, see Sec. 5 step 3.)

Eq. (5) and (6), `\label{eq:self-attn}` (computation, sequence T in R^{1 x M x D}, E = 2D):
```
&\Omega (\text{self-attention}) = 4\mathtt{M}\mathtt{D}^2 + 2\mathtt{M}^2\mathtt{D}, \\
&\Omega (\text{SSM}) = 3\mathtt{M}(2\mathtt{D})\mathtt{N} + \mathtt{M}(2\mathtt{D})\mathtt{N},
```
Paper: "self-attention is quadratic to sequence length M, and SSM is linear to sequence length M
(N is a fixed parameter, set to 16 by default)." Gloss: the page can evaluate both at M = 197 and 6084.

## 7. Config / setup table
| item | value | loc |
|---|---|---|
| patch size P | 16 (16x16 kernel projection, non-overlapping) | Sec. 3.4 |
| train/test resolution | 224^2 (center crop at test) | Sec. 4.1 |
| tokens at 224 | 196 patches + 1 cls = 197 | our reading (14^2 = 196 also in MambaOut Sec. 3.2) |
| tokens at 1248 | 6084 | Fig. 1 caption |
| depth L | 24 (all sizes, "by default") | Sec. 3.4 |
| SSM dim N | 16 | Sec. 3.4, Sec. 3.5 |
| Vim-Ti D / E | 192 / 384 | Sec. 3.4 |
| Vim-S D / E | 384 / 768 | Sec. 3.4 |
| Vim-B D / E | **not stated** in v3 paper or README | -- |
| params Ti / S / B | 7M / 26M / 98M | Table 1 |
| conv1d kernel width | not stated | -- |
| optimizer | AdamW, momentum 0.9, wd 0.05 | Sec. 4.1 |
| batch / epochs / lr | 1024 / 300 / 1e-3 cosine, EMA | Sec. 4.1 |
| augmentation | random crop, h-flip, label smoothing, mixup, random erasing (follows DeiT) | Sec. 4.1 |
| hardware | 8 A800 GPUs | Sec. 4.1 |
| long-sequence FT (dagger rows) | patch stride 8 (patch size 16 kept), 30 epochs, const lr 1e-5, wd 1e-8 | Sec. 4.1 |
| tokens with stride 8 at 224 | 27^2 = 729 patches (+1 cls) | our reading |
| ADE20K | UperNet, AdamW wd 0.01, batch 16, lr 6e-5, warmup 1500 it, 160K it, 512 crop | App. B |
| COCO | Cascade Mask R-CNN (ViTDet framework), AdamW wd 0.1, batch 64, lr 1e-4, 380K it, LSJ 1024^2 | App. B |

## 8. Results worth remembering
- ImageNet-1k top-1, Table 1 (p. 5), 224^2:
  - Vim-Ti 76.1 (7M); Vim-Ti-dagger 78.3. DeiT-Ti 72.2 (6M).
  - Vim-S 80.3 (26M); Vim-S-dagger 81.4. DeiT-S 79.8 (22M).
  - Vim-B 81.9 (98M); Vim-B-dagger 83.2. DeiT-B 81.8 (86M).
  - S4ND-ViT-B 80.4 (89M); ResNet-50 76.2.
  - Text (Sec. 4.1): +3.9 (Ti), +0.5 (S), +0.1 (B) over DeiT.
  - **Conflict**: README lists Vim-small 80.5 and Vim-small+ 81.6 (code-only). VMamba (Table 1) and
    MambaOut (Table 1) cite Vim-S = 80.5; MLLA (Table 3) cites 80.3. Use 80.3 for v3.
- High resolution, Vim-Ti vs DeiT-Ti, backbone only, batch inference (Fig. 1b/c, p. 1; Sec. 4.1 text):
  - Similar FPS and memory at 512x512.
  - At 1248x1248 (6084 tokens): "2.8x faster" and "saves 86.8% GPU memory".
  - Fig. 1(b) y-axis is log-scaled FPS: at 1248, Vim-Ti 1.71 vs DeiT-Ti 1.26 (10^0.45 = 2.8, our reading).
  - Fig. 1(c) GB: DeiT-Ti 4.56 / 8.09 / 12.48 / 40.09 / OOM; Vim-Ti 3.32 / 4.22 / 5.03 / 8.13 / 11.14
    at 512 / 640 / 738 / 1024 / 1248. DeiT-Ti at 1248 is drawn as "OOM" with no value, so the 86.8% is
    not recomputable from the figure (our reading). GPU model for this benchmark is not stated.
  - With backbone + FPN (Fig. 3 and Fig. 4, p. 7): still "2.8x faster" at 1248 (log FPS 1.70 vs 1.25), but
    memory saving is **-73.2%** (Vim 22.59 GB; DeiT OOM). Do not mix 86.8% (backbone) with this.
- ADE20K mIoU, UperNet, 512^2, Table 2: Vim-Ti 41.0 (13M) vs DeiT-Ti 39.2 (11M); Vim-S 44.9 (46M) vs
  DeiT-S 44.0 (43M); ResNet-101 44.9 (86M). Text: +1.8 and +0.9 mIoU.
- COCO, **Cascade** Mask R-CNN, 1024^2, Table 3: Vim-Ti AP^box 45.7 vs DeiT-Ti 44.4; AP^mask 39.2 vs
  38.1; AP^box_l 63.2 vs 61.8. Vim-Ti runs without modification; DeiT-Ti uses ViTDet window attention.
  Not comparable with VMamba/MambaOut rows (those are Mask R-CNN 1x, hierarchical backbones).
- Bidirectional ablation, Table 4 (no class token; ImageNet top-1 / ADE20K mIoU with 2-layer Segmenter):
  None 73.2 / 32.3; Bidirectional Layer 70.9 / 33.6; Bidirectional SSM 72.8 / 33.2;
  Bidirectional SSM + Conv1d (default) 73.9 / 35.9.
- Classification design, Table 5: Mean pool 73.9; Max pool 73.4; Head class token 75.2; Double class
  token 74.3; **Middle class token 76.1**. Paper's reason: it "can fully exploit the recurrent nature of
  SSM and the central object prior in ImageNet" (Sec. 4.4).
- Hier-Vim (Swin with window attention replaced by bidirectional SSM), App. C Table 7: T 82.5 (30M),
  S 83.4 (50M), B 83.9 (89M) vs Swin-T/S/B 81.2 / 83.2 / 83.5 (as listed there).

Text/table inconsistencies to avoid on the page:
- Sec. 4.4 names five strategies ("Bidirectional Sequence", "Bidirectional Block", ...); Table 4 has four
  rows and calls one "Bidirectional Layer". No row for "Bidirectional Sequence".
- Sec. 4.4 says Bidirectional Block is "7 points lower top-1"; Table 4 gives 70.9 vs 73.2 (2.3 lower).
- Sec. 3.3 text and Algorithm 1 both add the residual, and Sec. 3.2 adds it again (T_l = Vim(T_{l-1}) + T_{l-1});
  our reading: one residual, described twice.

## 9. Limitations stated
- Only supervised ImageNet pretraining so far; MAE-style and CLIP-style pretraining are future work (Sec. 5, Conclusion).
- No explicit limitation section beyond future work. VMamba (Sec. 5.3, Fig. 7) reports Vim's ERF is global
  but less uniform and less 2D-aware than VMamba's; MambaOut argues each causal branch still loses
  fully-visible mixing (MambaOut Sec. 3.1).

## 10. Outline feedback
- ch 9 `viz-vimflow` wants Vim-Ti/S/B configs: v3 states D/E only for Ti and S. Vim-B's D/E is in neither
  the paper nor the README. Use Ti and S for shapes, or fetch `models_mamba.py` and label it code-only.
- Say "Vim's E = 2D is a dimension; Mamba's E = 2 is a ratio" once, since the page uses Mamba notation.
- The class token goes in the middle **because it scored best** (Table 5, 76.1 vs 75.2 head); the paper's
  explanation (recurrence + central-object prior) is a post-hoc reason. Eq. 4 shows it at the head.
- The "2.8x / 86.8%" pair is Vim-Ti vs DeiT-Ti, backbone only, at 1248^2. The FPN version is 2.8x / 73.2%.
  At 224^2 (the ImageNet setting) there is no speed claim; at 512^2 they are "similar".
- `viz-scan2d` "Vim (2 routes)": both routes are the same row-major flattening, forward and reversed; there
  is no column route. Each branch is causal (MambaOut's point).
