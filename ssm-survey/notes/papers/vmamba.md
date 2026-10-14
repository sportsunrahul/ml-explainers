# VMamba

## 1. Identity
- Title: "VMamba: Visual State Space Model"
- arXiv 2401.10166, read **v4** (2024-12-29). v1 2024-01-18, v2 2024-04-10, v3 2024-05-26 (abs.html).
- Authors: Yue Liu, ..., Yunfan Liu (corresponding, double dagger). UCAS, Huawei, Pengcheng Lab.
- Venue: NeurIPS 2024 spotlight (abs.html comment; README update 2024-09-25).
- Repo: https://github.com/MzeroMiko/VMamba
- Sources used: src/main.tex, src/appendix.tex (equations, tables), paper.pdf (figures, numbering), README.md.

## 2. Core problem
Mamba's selective scan is 1-D and ordered. Image patches have no natural order. The paper builds a
hierarchical vision backbone with linear cost that keeps global receptive fields and input-dependent weights.

## 3. What prior methods did and why that hurt (per the paper)
- Efficient attention either restricts the effective receptive field (Swin windows) or loses accuracy
  (linear attention, Linformer) (Sec. 1).
- S4ND extends the SSM kernel to 2D by outer product, which "restricts the weights from being
  input-dependent" (Sec. 4.2).
- A single 1-D scan gives each patch context only from patches earlier on its path (Fig. 1b).

## 4. The one key idea
SS2D: unfold the patch grid into four 1-D sequences (cross-scan), run a separate S6 scan on each, and fold
the four outputs back onto the grid and merge them (cross-merge), so every patch receives information from
all other patches through some route.

## 5. Mechanism, step by step (VMamba-T, 224^2; shapes from Table 4, our reading where noted)
1. Stem: conv 3x3 stride 2, LayerNorm, GELU, conv 3x3 stride 2, LayerNorm (Table 4). 224 -> 56; feature
   map 56 x 56 x 96 (H/4 x W/4, Sec. 4.1). No position embeddings ("Without incorporating additional
   positional embeddings", Sec. 4.1).
2. Four stages at H/4, H/8, H/16, H/32: 56^2 = 3136, 28^2 = 784, 14^2 = 196, 7^2 = 49 tokens (our reading:
   arithmetic). Channels 96, 192, 384, 768. Depths [2, 2, 8, 2]. Downsampling between stages:
   conv 3x3 stride 2 + LayerNorm (Table 4).
3. VSS block (Fig. 3d; Table 4 block): two residual sub-blocks, "mimicking the architecture of a vanilla
   Transformer block" (Sec. 4.1):
   - x = x + SS2D-block(LN(x)), where SS2D-block = Linear C -> ssm-ratio*C, DWConv 3x3, SiLU, SS2D,
     LN, Linear ssm-ratio*C -> C (Fig. 3d inset; Table 4).
   - x = x + FFN(LN(x)), FFN hidden = mlp-ratio*C = 4C.
   - No multiplicative (gate) branch: removed "as the effect of the gating mechanism has already been
     achieved by the selectivity of SS2D" (Sec. 4.1).
4. SS2D (Fig. 2, Sec. 4.2), on a h x w map with d = ssm-ratio*C channels:
   - Cross-scan into four length-hw sequences (route order read from Fig. 2 on a 3x3 grid):
     route 1 row-major from top-left (1,2,3,...,9); route 2 column-major from top-left (1,4,7,...,9);
     route 3 = route 1 reversed (9,8,...,1); route 4 = route 2 reversed (9,6,3,...,1).
   - Each sequence goes through a **separate** S6 block (own parameters), in parallel. S6 per Fig. 2:
     Delta from a Linear; B, C from a Linear; A, D are parameters; h_t = e^{Delta A} h_{t-1} + Delta B x_t;
     y_t = C h_t + D x_t.
   - Cross-merge: outputs are "reshaped and merged to form the output map". The merge operator (sum)
     is not stated in the text; our reading from Fig. 2 and the official code (code-only, not verified here).
5. State size: d_state = 1 in the final models (Table 5 Step(e.2)-(g); App. H.3 "we primarily set the
   value of d_state to 1"). Stage-1 state per route for VMamba-T: d x N = 96 x 1 = 96 numbers per image
   (our reading). Compare Vim-Ti: 384 x 16 per direction.
6. Head: average pool, 1000-d fc, softmax (Table 4).
7. Trainable / init: S4D-style init for A ("we adopted the initialization scheme ... in S4D", App. H.2);
   log(A) is the parameter; with d_state = 1 Mamba init of log(A) is all zeros. D = 1 for rand/zero init.
   Init choice barely matters: mamba 82.60, rand 82.58, zero 82.67 (Table 11). Loss: ImageNet CE with label
   smoothing 0.1.
8. Selective SSM is computed with an associative scan (App. B, Eq. recur_pA/pB) and the Mamba hardware kernel.

Sign/scaling: VMamba writes the input as u, state h, with Bbar = Delta B (first-order Taylor of ZOH,
App. A); Abar = e^{A Delta} is exact ZOH. In practice A is a vector in R^{1 x D_k} (App. C), i.e. diagonal.

## 6. Equations (src/main.tex; numbers from PDF p. 3, 8)
VMamba writes the input as u(t); the page's x. Otherwise same letters.

Eq. (1), `\label{eq:ssm}`:
```
\mathbf{h'}(t) &= \mathbf{A} \mathbf{h}(t) + \mathbf{B} u(t),  \\
y(t)  &= \mathbf{C} \mathbf{h}(t) + D u(t),
```
A in R^{N x N}, B in R^{N x 1}, C in R^{1 x N}, D scalar. Gloss: continuous SSM with input x (= u), state h, skip D.

Eq. (2), `\label{eq:ode_slt}`:
```
\mathbf{h}(t_b) = e^{\mathbf{A}(t_b-t_a)} \mathbf{h}(t_a) + e^{\mathbf{A}(t_b - t_a)} \int_{t_a}^{t_b} \mathbf{B}(\tau) u(\tau) e^{-\mathbf{A}(\tau-t_a)} \,d\tau.
```
Gloss: exact solution of the ODE over [t_a, t_b].

Eq. (3), `\label{eq:ode_dis}`:
```
\mathbf{h}_b = e^{\mathbf{A}(\Delta_a+...+\Delta_{b-1})} \left( \mathbf{h}_a + \sum^{b-1}_{i=a} \mathbf{B}_i u_i e^{-\mathbf{A}(\Delta_a+...+\Delta_i)} \Delta_i \right),
```
Paper: "this formulation approximates the result obtained by the zero-order hold (ZOH) method". With
b = a + 1 it gives h_{a+1} = e^{A Delta_a} h_a + B_a Delta_a u_a (App. A, eq:ode_dis_step_app).

Eq. (4), `\label{eq:ode_slt_yj}` (main text version; the appendix redefines the same label with a
rearranged RHS):
```
\mathbf{y_b}^{(j)} = \left( \mathbf{Q_T} \odot \mathbf{w_T}^{(j)} \right) 
{\mathbf{h_a}^{(j)}} +  \mathbf{Q_T} \sum_{i=1}^{T} \left(\frac{\mathbf{w_T}^{(j)}}{\mathbf{w_i}^{(j)}}  \odot \mathbf{K_i}\right)^\top \odot \left(\mathbf{V_i}^{(j)}\right).
```
Eq. (5), `\label{eq:VMamba_att}`:
```
\mathbf{Y}^{(j)} = \left[\mathbf{Q} \odot \mathbf{w}^{(j)}\right] \mathbf{h_a}^{(j)} + \left[ \left( \mathbf{Q} \odot \mathbf{w}^{(j)} \right) \left( \frac{\mathbf{K}}{\mathbf{w}^{(j)}} \right)^\top \odot \mathbf{M} \right] \mathbf{V}^{(j)},
```
Definitions (Sec. 5.3): V_i = u_i (.) Delta_i, K_i = B_i, Q_i = C_i; w_i = prod_{j=1}^{i} e^{A Delta_{a-1+j}^T}
"representing the cumulative attention weight"; M = lower-triangular T x T mask.
Paper: "the matrix multiplication process involving Q, K, and V closely resembles the self-attention
mechanism, despite the inclusion of w." App. C: if w had shape (T, D_k) it "reduce[s] to the form of
Gated Linear Attention (GLA)". Gloss: C acts as query, B as key, Delta-scaled input as value, and the
cumulative decay w is a causal, per-channel forget factor. This is the same bridge as ch 8 (SSD) and MLLA.

## 7. Config / setup table
Final VMamba (Table 4, App. E; d_state from Table 5 and App. H.3):
| model | dims per stage | depths | ssm-ratio | mlp-ratio | d_state | params | FLOPs |
|---|---|---|---|---|---|---|---|
| VMamba-T | 96, 192, 384, 768 | 2, 2, 8, 2 | 1 | 4 | 1 (Table 5 Step g) | 30.2M | 4.91G |
| VMamba-S [s2l15] | 96, 192, 384, 768 | 2, 2, 15, 2 | 2 | 4 | 1 (App. H.3 global statement) | 50.1M | 8.72G |
| VMamba-B [s2l15] | 128, 256, 512, 1024 | 2, 2, 15, 2 | 2 | 4 | 1 (App. H.3 global statement) | 88.6M | 15.36G |
The `\blockb` macro in Table 4 is passed a third argument "16" that is never rendered; ignore it.
Alternative S/B with ssm-ratio 1, stage-3 depth 20: VMamba-S[s1l20] 83.3, VMamba-B[s1l20] 83.8 (Table 6).

Vanilla VMamba (Table 3; first design, multiplicative branch, no FFN, d_state 16, ssm-ratio 2 via "2x" dims):
T: stem conv 4x4 stride 4, 96; depths [2, 2, 9, 2]; 22.9M; 5.63G. S: [2, 2, 27, 2]; 44.4M. B: dims 128..1024,
[2, 2, 27, 2]; 76.3M.

Training (App. E): 300 epochs, 20 warm-up, batch 1024, AdamW betas (0.9, 0.999), lr 1e-3, wd 0.05, cosine,
label smoothing 0.1, EMA 0.9999, drop path 0.2 (T) / 0.3 (S) / 0.6 (B s2l15). Hyper-parameters from Swin.
Hardware: 8 x A100 (Sec. 5). Throughput: A100 + AMD EPYC 7542, timm toolkit, batch 128 (App. E).
Detection: Mask R-CNN, mmdetection, AdamW lr 1e-4, batch 16, 12 or 36 epochs (App. F).
Segmentation: UperNet, lr 6e-5, 160k iterations, batch 16, 512^2 (App. F).

## 8. Results worth remembering
- ImageNet-1k, 224^2, Table 1 (p. 6): VMamba-T 82.6 (30M, 4.9G, 1686 img/s); VMamba-S 83.6 (50M, 8.7G,
  877); VMamba-B 83.9 (89M, 15.4G, 646). Swin-T/S/B 81.3 / 83.0 / 83.5 at 1244 / 718 / 458 img/s.
  DeiT-S 79.8 (1761), DeiT-B 81.8 (503). ConvNeXt-T 82.1 (1198). Vim-S 80.5 (811) as cited here
  (Vim v3 says 80.3). S4ND-Conv-T 82.2 (683).
  Text says VMamba-T throughput is "1.47x higher than S4ND-Conv-T and 1.08x higher than Vim-S"; the table
  ratios are 2.47x and 2.08x (our reading: the text means "% more"). Use raw img/s on the page.
- v1-style -> final (Sec. 4.3, Fig. 3e, Table 5): Vanilla-VMamba-T 82.17, 426 img/s ->
  Step(a) Triton cross-scan 467 -> (b) fp16 in / fp32 out 464 -> (c) einsum -> linear, (B,C,H,W) layout
  638 -> (d) add MLP, drop DWConv, [2,2,2,2]: 81.65, 813 -> (d.1) ssm-ratio 1: 81.05, 1336 ->
  (d.2) [2,2,5,2]: 82.24, 1137 -> (e) drop multiplicative branch: 82.17, 1179 -> (e.1) DWConv back: 82.31,
  1164 -> (e.2) d_state 16 -> 1: 81.87, 1942 -> (f) ssm-ratio 2: 82.49, 1340 -> (g) ssm-ratio 1, [2,2,8,2]:
  **82.60, 1686**. Net (our reading): about 4x throughput, +0.43 top-1. The paper never labels these "v1"/"v4"; MambaOut calls
  the final design "VMambaV9" and the vanilla one "VMamba".
- COCO Mask R-CNN 1x, Table 2: AP^b / AP^m VMamba-T 47.3 / 42.7, S 48.7 / 43.7, B 49.2 / 44.1;
  Swin-T 42.7 / 39.3, ConvNeXt-T 44.2 / 40.1. 3x MS: VMamba-T 48.8 / 43.7, VMamba-S 49.9 / 44.2.
- ADE20K UperNet 512^2, Table 2: mIoU SS / MS VMamba-T 47.9 / 48.8, S 50.6 / 51.2, B 51.0 / 51.6;
  Swin-T 44.5 / 45.8; Vim-S 44.9 (cited). **Conflict**: appendix Table 8 gives VMamba-T SS 48.0.
- Resolution scaling, Table 9 (batch 32, A100), trained at 224, tested without fine-tuning:
  VMamba-T at 768^2: 74.66 top-1 (79.22 linear tuning), 57.66G, 149 img/s. Swin-T at 768^2: 73.06, 53 img/s.
  DeiT-S at 768^2: 60.98, 90 img/s. ConvNeXt-T 69.50, 103 img/s. Text rounds to 74.7 / 79.2 (Sec. 5.2).
- ERF (Fig. 7, Sec. 5.3): "only DeiT, HiViT, Vim and VMamba demonstrate global ERFs"; "VMamba's ERF is more
  uniform and 2D-aware than that of Vim". Qualitative heat maps, no number. ERF goes local -> global during
  training (Fig. 11, Vanilla-VMamba-T).
- Scan pattern ablation, VMamba-T, Table 10 (Fig. 8): with DWConv: Unidi 82.26, Bidi 82.49, Cascade 82.42
  (998 img/s), Cross 82.60 (1686). Without DWConv: Unidi 80.88, Bidi 81.80, Cascade 81.71, Cross 82.25.
  The 4-route gain over 1 route is 0.34 with DWConv and 1.37 without (our reading: arithmetic).
- d_state, VMamba-T, Table 12: 1 / 2 / 4 at ssm-ratio 2: 82.49 / 82.50 / 82.51 (1340 / 1269 / 1147 img/s);
  8 at ratio 1.5: 82.69; 16 at ratio 1.0: 82.31.

## 9. Limitations stated (Sec. 6)
- Unsupervised / large-scale pretraining (e.g. ImageNet-21K) not explored for SSM backbones.
- No Large-scale model and no fine-grained hyper-parameter search (compute).
- A more general scanning pattern for other tasks/modalities is open.
- Later: MambaOut (2405.07992) shows a Gated-CNN without SSM matches or beats VMamba on ImageNet but not on
  COCO/ADE20K; MLLA (2405.16605) explains the gain as forget gate + block design.

## 10. Outline feedback
- `viz-scan2d` "VMamba (4 routes)": routes are row-major, column-major, and both reversed (Fig. 2). Each
  route has its own S6 parameters. The merge rule is not stated in text; label "sum" as our reading/code-only.
- State size N = 1 (d_state) in final VMamba is a striking contrast to Mamba's N = 16 and fits ch 10's
  "fixed state is lossy" story; worth one sentence in ch 9.
- VMamba is hierarchical (4 stages, 3136 -> 49 tokens, no position embedding, no class token); Vim is
  plain (197 tokens at every layer). The page should not draw VMamba as one flat sequence.
- The evidence that more routes help is Table 10 (small with DWConv, larger without). The ERF claim is a
  heat-map figure only.
- "What changed v1 -> v4": cite Sec. 4.3 / Table 5 (Vanilla -> Step g), not version numbers.
- Detection rows (Mask R-CNN 1x) are not comparable with Vim's Cascade Mask R-CNN rows.
