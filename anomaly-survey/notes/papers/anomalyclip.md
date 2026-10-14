# AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection

## 1. Identity
- Title: "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection".
- arXiv 2310.18961, version read: v12 (abs.html; LaTeX src and paper.pdf are v12, 31 pages).
- Dates (abs.html): v1 2023-10-29; v12 2026-01-01. Twelve versions in total (v2 2023-11-03 ... v8 2024-03-16, v9 2025-04-21, v10 2025-04-27, v11 2025-11-29).
- Authors: Qihang Zhou (Zhejiang University, first) ... Jiming Chen (Zhejiang University, last). Full list: Zhou, Pang, Tian, He, Chen.
- Venue: ICLR 2024 (abs.html comments).
- Repo: https://github.com/zqhang/AnomalyCLIP (abstract, "Code will be made available at"). The code was not read; nothing below is `code-only`.
- Sources: `sources/anomalyclip/src/iclr2024_conference.tex` (macros in `src/math_commands.tex`), `sources/anomalyclip/paper.pdf`.

### Version-sensitive points in v12 (record, do not reconcile)
- Orange text in the LaTeX marks revised passages: the Sec. 4.1 sentence "We use the top feature map as local visual details for anomaly segmentation."; all of Appendix C (DPAM analysis); parts of Related Work. Earlier versions were not read.
- Local layers: v12 says only the top visual feature map is used (Sec. 4.1, App. A, and the Sec. 3.1 footnote "Although we use only the top feature map of visual encoder in the code implementation, AnomalyCLIP is capable of incorporating multiple intermediate feature maps"). But Eq. (2) sums over a set of layers, Fig. 2 draws local maps from four blocks, and the Fig. 7(d) plot axis reads "Ablation on used layer of visial encoder N" (0 to 4, best at 4).
- Fig. 7 caption says (d) is "lambda analysis" and the text says "AnomalyCLIP achieves the overall performance gain when lambda = 4", but the plotted axis is N (layers). At x = 0 the plotted pixel AUPRO (about 77.8 MVTec AD, 78.1 VisA) matches Table 5's "global loss only" row, consistent with either reading.
- Fig. 7(b) text: "the performance drops when the refinement is excessive ... as seen in the case when M equals 9", yet 9 is the chosen depth and the plotted peak.
- Notation overload: M is the set of visual layers in Eq. (2), the text-token depth in Fig. 7(b), and App. A calls the prompt length "M" (= E in Sec. 4.1).

## 2. Core problem
Zero-shot anomaly detection (ZSAD): detect and segment anomalies in a target dataset with no training sample from it. The model may be trained on a different, auxiliary anomaly dataset. CLIP's embeddings encode object class semantics, not normal/abnormal state, so plain CLIP prompting is weak for ZSAD (Abstract, Sec. 1).

## 3. What prior methods did and why that hurt (per the paper)
- CLIP with "A photo of a normal/anomalous [cls]" and CLIP-AC (ImageNet prompt ensemble): weak, because CLIP aligns object semantics (Sec. 4.2, Table 1: MVTec AD pixel AUROC 38.4 / 38.2).
- WinCLIP: "a large number of hand-crafted text prompts" and "multiple forward passes of image patches for anomaly segmentation" (Sec. 5).
- VAND: learnable linear projections for local features, still hand prompts; textual embeddings not general enough for unseen object semantics (Sec. 5).
- CoOp-style learnable prompts: optimize global features for object alignment and miss fine-grained local abnormality; very poor segmentation (Table 1: CoOp MVTec AD pixel AUROC 33.3) (Sec. 1, Sec. 4.2).

## 4. The one key idea
Learn two text prompts whose only fixed words are "object" and "damaged object", so the learned embeddings encode generic normality and abnormality rather than any product. Train them once on one auxiliary dataset with image-level and pixel-level losses, then reuse them unchanged on unseen products and domains.

## 5. Mechanism, step by step
1. Prompts (Sec. 3.2). Normal: `[V_1]...[V_E][object]`. Abnormal: `[W_1]...[W_E][damaged][object]`. E = 12 learnable word embeddings each (Sec. 4.1).
2. Text-encoder refinement (Sec. 3.3, Eq. 3). In the frozen CLIP text encoder, the first Q token embeddings of a layer's input are replaced by fresh learnable tokens q_m; the encoder's outputs at those positions (r) are discarded and new learnable tokens are inserted at the next layer. Q = 4 tokens per layer ("each text token has a length of 4", App. A; called L in Fig. 7(c)). Attached to "the first 9 layers" (Sec. 4.1, App. A); the method text says replacement starts "from the bottom (the second layer)". Our reading: the layer-1 learnable input is the prompt itself, deep tokens enter layers 2 to 9. Whether q_m is shared between the normal and abnormal prompt is not stated.
3. Text outputs: g_n and g_a in R^D (Sec. 2).
4. Image side: CLIP ViT-L/14@336px, all parameters frozen; images resized to 518 x 518 after OpenCLIP normalization (App. A, App. B). Our reading: 518 / 14 = 37, so a 37 x 37 patch grid (position-embedding interpolation is not mentioned).
5. DPAM (Sec. 3.3, App. A, App. C). Starting from the 6th visual layer, Q-K attention is replaced by V-V attention (default; Q-Q and K-K are ablated). DPAM has no parameters. Fig. 2 draws two parallel paths through the frozen encoder: "DPAM layer" blocks produce the local visual embedding (patch tokens); "Original layer" blocks produce the global visual embedding (class token). Our reading of Fig. 2.
6. Scores (Sec. 2). Image: P(g_a, f_i) from Eq. (1), the softmax over {g_n, g_a}. Local: S_n and S_a in R^{H x W} with entries P(g_n, f_i^{m(j,k)}) and P(g_a, f_i^{m(j,k)}) per patch token.
7. Loss (Eq. 2): L_global (cross-entropy on the image-level P) + lambda x sum over layers of L_local (focal + two Dice terms on the upsampled maps vs the ground-truth mask). lambda = 4. Fig. 2 shows the image label obtained from the ground-truth mask through a "Maximum" node (our reading: label = 1 if any pixel is anomalous).
8. Training (App. A): Adam, lr 0.001, batch 8, 15 epochs, one NVIDIA RTX 3090 24GB, PyTorch 2.0.0. Training time not stated.
9. Inference (Sec. 3.3 "Training and Inference"): image score P(g_a, f_i); map = Gaussian-smoothed (sigma = 4, App. A) sum over layers of 0.5 (1 - Up(S_n)) + 0.5 Up(S_a). One forward pass per image (Sec. 5).

What is trained vs frozen:
- Trained: V_1..V_12, W_1..W_12, and the deep text tokens q_m (Fig. 2 flame icons on V, W and on t'_M ... t'_{M'}).
- Frozen: every CLIP weight, image and text ("Model parameters of CLIP are all frozen", Sec. 4.1; "with the visual encoder kept frozen during training", Sec. 3.3).
- Gradient path (our reading; the paper never states it in these words): both losses depend on the prompts only through g_n and g_a, which are outputs of the frozen text encoder. So the gradient flows backward through all 12 frozen text layers (our reading of CLIP depth; not stated) into the deep tokens of layers 2-9 and the input word embeddings. On the vision side there is nothing to train (DPAM is parameter-free), so the image features act as constants.
- Parameter count: not stated.

Auxiliary-data protocol (Sec. 4.1, App. A):
- To test on every dataset except MVTec AD: train on the MVTec AD TEST split (15 classes, 467 normal / 1258 anomalous images with masks, Table 6).
- To test on MVTec AD: train on the VisA TEST split (12 classes, 962 normal / 1200 anomalous, Table 6).
- Medical results in Table 2 also use MVTec AD-trained prompts. Table 3 uses a built medical set: ColonDB (380 anomalous, masks) + Endo test split normals, 163 normal + 380 anomaly (App. B).
- Results are per dataset, averaged over sub-datasets (Sec. 4.1). One set of prompts for all categories of a target dataset.

## 6. Equations (verbatim TeX from iclr2024_conference.tex)
Custom macros: `\R` = `\mathbb{R}` (math_commands.tex:480), `\sG` = `{\mathbb{G}}` (math_commands.tex:380).

Eq. (1), `\label{equ: softmax}`, Sec. 2 (Preliminary):
```tex
p(y = c | x_i) = P(g_c, f_i) = {\frac{exp(<g_c, f_i>/\tau)}{\sum_{c\in \mathcal{C}}exp(<g_c, f_i>)/\tau)}},
```
Typo as printed: the denominator's parenthesis closes before /\tau, and c is reused as the summation index. Paper: "CLIP predicts the probability of x_i belonging to c", tau a temperature, <.,.> cosine similarity; "We denote the probability of being abnormal P(g_a, f_i) as the anomaly score." Gloss: g_c = text embedding of class c (here only g_n and g_a); f_i = class-token image embedding.

Local maps, inline in Sec. 2 and in the unnumbered gather of Sec. 3.3:
```tex
S^{(j,k)}_{n, M_l} = P(g_n, f^{m(j, k)}_{i, M_l}), \enspace
S^{(j,k)}_{a, M_l} = \textstyle P(g_a, f^{m(j, k)}_{i, M_l}), \enspace \text{where} \enspace j\in[1, H], k\in[1, W]
```
Gloss: f^{m(j,k)}_{i,M_l} = patch token (j, k) of image i at visual layer M_l; S_n, S_a = per-patch normal / abnormal probabilities (they sum to 1 per patch, our reading of Eq. 1).

Object-aware and object-agnostic templates (unnumbered, Sec. 3.2):
```tex
g_n &= [V_1][V_2]\dots[V_E][cls] \nonumber\\
g_a &= [W_1][W_2]\dots[W_E] [damaged][cls],\nonumber
```
```tex
g_n&= [V_1][V_2]\dots[V_E][object] \nonumber\\
g_a &= [W_1][W_2]\dots[W_E][damaged][object].\nonumber
```
Paper: the object-agnostic templates "replace the class name in g_n and g_a with `object`, blocking out the class semantics of objects". Gloss: V_i, W_i = learnable word embeddings (E = 12); [cls] = product name; [damaged], [object] = fixed word tokens.

Eq. (2), `\label{eq:total_loss}`, Sec. 3.3:
```tex
L_{total} = L_{global} + \lambda \textstyle \sum_{M_l \in \mathcal{M}}L_{local}^{M_l},
```
Paper: "lambda is a hyperparameter to balance the global and local losses. L_global is a cross-entropy loss that matches the cosine similarity between the object-agnostic textual embeddings and visual embeddings of normal/abnormal images from auxiliary data." L_global has no printed formula. Gloss: \mathcal{M} = set of visual layers whose patch tokens are scored; lambda = 4.

L_local (unnumbered, Sec. 3.3):
```tex
L_{local} =  Focal(Up([S_{n, M_l}, S_{a, M_l}]), S) + Dice(Up(S_{n, M_l}), I-S) + Dice(Up(S_{a, M_l}), S), \nonumber
```
Paper: Up = upsampling, [.,.] = channel concatenation, I = all-ones matrix; S in R^{H_image x W_image} = ground-truth mask (1 = anomalous pixel). Focal loss "to address the imbalance problem" (anomalous regions are small); Dice "to measure the overlaps between the predicted segmentation ... and the ground truth mask". Gloss: focal grades the two-channel per-pixel normal/abnormal map against the mask; one Dice term asks the normal map to cover the normal region, the other asks the abnormal map to cover the defect.

Eq. (3), Sec. 3.3 (only the 4th line carries the number):
```tex
t_2 &= T_1(t_1)  \nonumber\\
&\ldots \nonumber\\
[r^i_{m+1}|_1^Q, t^i_{m+1}|_{Q+1}^P] &= T_m([q^i_m|_1^Q, t^i_m|_{Q+1}^P]) \nonumber\\
[r^i_{m+2}|_1^Q, t^i_{m+2}|_{Q+1}^P] &= T_{m+1}([q^i_{m+1}|_1^Q, t^i_{m+1}|_{Q+1}^P])\\
&\ldots \nonumber\\
t_{M'+1} &= T_{M'}([r^i_{M'}|_1^Q, t^i_{M'}|_{Q+1}^P]),\nonumber
```
Paper: "we replace t^i_m|_1^Q with q_m ... we discard the obtained r^i_{m+1}|_1^Q and initialize new learnable token embeddings q^i_{m+1}|_1^Q ... until we reach the designated layer M'". Gloss: T_m = text-encoder layer m (frozen); t_m = its P input tokens; q_m = Q learnable tokens that overwrite the first Q positions; r = outputs at those positions, thrown away. Note: the last line feeds r (not q) into T_{M'}, as printed.

Inference map (inline, Sec. 3.3 "Training and Inference"):
```tex
Map = G_{\sigma}( \sum_{M_l \in \mathcal{M}}(\frac{1}{2}(I - Up(S_{n, M_l}))+ \frac{1}{2}Up(S_{a, M_l})))
```
Gloss: per layer, average "not normal" and "abnormal"; sum over layers; Gaussian smoothing with sigma = 4.

DPAM (App. C, unnumbered, `\nonumber`):
```tex
Attention( {Q}, {K}, {V})_i =  softmax\left(\frac{q_iK^{\top}}{\sqrt{D}}\right)V =\frac{\sum\limits_{j=1}^n a_{ij} v_j}{\sum\limits_{j=1}^n a_{ij}}\label{eq:std-att-2}, \quad \quad a_{ij} =  e^{  \frac{q_i k_j^{\top}}{\sqrt{D}}}. \nonumber
```
```tex
a^{qq}_{ij} =  e^{ \frac{q_i q_j^{\top}}{\sqrt{D}}}, \quad  a^{kk}_{ij} =  e^{ \frac{k_i k_j^{\top}}{\sqrt{D}}}, \quad a^{vv}_{ij} =  e^{ \frac{v_i v_j^{\top}}{\sqrt{D}}}.  \nonumber
```
Paper: in CLIP's Q-K attention many queries attend to the same few tokens m, so output i "is dominated by v_m"; V-V "does not participate in computing the original attention map", so v_i keeps the largest weight a_ii, giving a "diagonally prominent attention map". Gloss: same values V are averaged; only the weights change, from q_i.k_j to v_i.v_j, so each patch mostly keeps its own content. No new weights.

## 7. Config / setup (method lens)
| Item | Value | Where |
|---|---|---|
| Backbone | CLIP ViT-L/14@336px (OpenCLIP), all parameters frozen | Sec. 4.1, App. A |
| Input | 518 x 518, OpenCLIP normalization | App. A, App. B |
| Patch grid | 37 x 37 | our reading (518/14) |
| Learnable word embeddings | E = 12 per prompt, 2 prompts | Sec. 4.1 (App. A calls it M) |
| Deep text tokens | 4 per layer, first 9 text layers, random init | Sec. 3.3, Sec. 4.1, App. A, Fig. 7(b)(c) |
| DPAM | V-V attention, from the 6th visual layer, parameter-free | Sec. 3.3, App. A |
| Local feature layers | top layer only (v12 text); Fig. 2 and Fig. 7(d) show up to 4 | Sec. 4.1 (orange), App. A, footnote Sec. 3.1 |
| Loss | L_global (CE) + 4 x sum L_local (focal + Dice + Dice) | Eq. 2, App. A |
| Optimizer | Adam, lr 0.001 | App. A |
| Batch / epochs | 8 / 15 | App. A |
| Hardware | 1 x RTX 3090 24GB, PyTorch 2.0.0 | Sec. 4.1, App. A |
| Test smoothing | Gaussian, sigma = 4 | App. A |
| Aux data | MVTec AD test (to test others); VisA test (to test MVTec AD) | Sec. 4.1 |
| Latency / throughput | not reported | whole paper |

Tensor trace (training step, our reading for anything not in the table above):
- Text: 2 prompts x (12 learnable + fixed word tokens, padded to the CLIP context) -> frozen text layers 1-12 with 4 learnable tokens overwritten at layers 2-9 -> g_n, g_a in R^D.
- Image: 8 x 3 x 518 x 518 -> frozen ViT-L/14 -> class token f_i (original-attention path) and 37 x 37 patch tokens (DPAM path, top layer per v12) projected to R^D.
- Scores: P(g_a, f_i) per image; S_n, S_a in R^{37 x 37} per image -> Up to 518 x 518 -> focal + Dice vs mask.
- Backward: gradient to V, W, q_m only, through the frozen text encoder; no gradient needed in the vision tower.

## 8. Results worth remembering
Protocol for every AnomalyCLIP number: zero-shot on the target dataset; prompts trained once on another dataset's test split (VisA test -> MVTec AD; MVTec AD test -> all others); one prompt pair for all categories; 518 px; ViT-L/14@336px.

Table 1 (p.7), industrial. Image (AUROC, AP); pixel (AUROC, PRO). Dagger = "results taken from original papers".
| Dataset | CLIP | CLIP-AC | WinCLIP | VAND | CoOp | AnomalyCLIP |
|---|---|---|---|---|---|---|
| MVTec AD img | (74.1, 87.6) | (71.5, 86.4) | (91.8, 96.5)+ | (86.1, 93.5)+ | (88.8, 94.8) | (91.5, 96.2) |
| VisA img | (66.4, 71.5) | (65.0, 70.1) | (78.1, 81.2)+ | (78.0, 81.4)+ | (62.8, 68.1) | (82.1, 85.4) |
| MVTec AD pix | (38.4, 11.3) | (38.2, 11.6) | (85.1, 64.6)+ | (87.6, 44.0)+ | (33.3, 6.7) | (91.1, 81.4) |
| VisA pix | (46.6, 14.8) | (47.8, 17.3) | (79.6, 56.8)+ | (94.2, 86.8)+ | (24.2, 3.8) | (95.5, 87.0) |
(+ = dagger, copied from the original paper. The four WinCLIP dagger cells equal WinCLIP's own Table 1 / Table 4. CLIP, CLIP-AC, CoOp, and WinCLIP/VAND on the other five datasets are AnomalyCLIP's runs, "All parameters are kept the same as in their paper", App. A.)
- WinCLIP beats AnomalyCLIP on MVTec AD image AUROC (91.8 vs 91.5) and AP (96.5 vs 96.2); AnomalyCLIP wins MVTec AD pixel metrics and every VisA metric.
- Other industrial AnomalyCLIP rows: MPDD (77.0, 82.0) / (96.5, 88.7); BTAD (88.3, 87.3) / (94.2, 74.8); SDD (84.7, 80.0) / (90.6, 67.8); DAGM (97.5, 92.3) / (95.6, 91.0); DTD-Synthetic (93.5, 97.0) / (97.9, 92.3).
- WinCLIP re-run by AnomalyCLIP: MPDD (63.6, 69.9) / (76.4, 48.9); BTAD (68.2, 70.9) / (72.7, 27.3); SDD (84.3, 77.4) / (68.8, 24.2); DAGM (91.8, 79.5) / (87.6, 65.7); DTD-Synthetic (93.2, 92.6) / (83.9, 57.8).

Table 2 (p.8), medical, prompts trained on MVTec AD: e.g. HeadCT image (93.4, 91.6); BrainMRI (90.3, 92.2); Br35H (94.6, 94.7); COVID-19 (80.1, 58.7); ISIC pixel (89.7, 78.4); CVC-ColonDB (81.9, 71.3); CVC-ClinicDB (82.9, 67.8); Kvasir (78.9, 45.6); Endo (84.1, 63.6); TN3K (81.5, 50.4).
Table 3 (p.9), prompts trained on the ColonDB-based medical set: HeadCT (93.5, 95.1); BrainMRI (95.5, 97.2); Br35H (97.9, 98.0); COVID-19 (70.9, 33.7); ISIC (83.0, 63.8); CVC-ClinicDB (92.4, 82.9); Kvasir (92.5, 61.5); Endo (93.2, 84.8); TN3K (79.2, 47.0).

Table 4 (p.9), module ablation; pixel (AUROC, PRO), image (AUROC, AP):
| Row | MVTec AD pix | MVTec AD img | VisA pix | VisA img |
|---|---|---|---|---|
| Base | (46.8, 15.4) | (66.3, 83.3) | (47.9, 17.1) | (54.4, 61.7) |
| + T1 DPAM | (68.4, 47.4) | (66.3, 83.3) | (54.8, 32.7) | (54.4, 61.7) |
| + T2 object-agnostic prompts | (89.5, 81.2) | (90.8, 96.0) | (95.0, 85.3) | (81.7, 85.2) |
| + T3 text-encoder tokens | (91.1, 81.4) | (91.5, 96.2) | (95.5, 87.0) | (82.1, 85.4) |
"Base" is not defined in the text. DPAM changes pixel metrics only (image metrics unchanged, as expected: the class token uses the original path). T2 adds learned prompts and object-agnostic wording together, so it does not isolate "agnostic vs aware".

Table 5 (p.9), context optimization; same column order:
| Local | Global | MVTec AD pix | MVTec AD img | VisA pix | VisA img |
|---|---|---|---|---|---|
| no | no | (71.7, 57.7) | (68.8, 85.8) | (74.7, 62.1) | (61.1, 69.1) |
| no | yes | (80.3, 77.8) | (89.9, 95.4) | (86.6, 78.1) | (82.2, 84.9) |
| yes | no | (91.0, 80.4) | (89.9, 96.0) | (95.2, 86.5) | (79.5, 83.2) |
| yes | yes | (91.1, 81.4) | (91.5, 96.2) | (95.5, 87.0) | (82.1, 85.4) |

Fig. 5 (p.9), object-agnostic minus object-aware prompt learning (bar labels as printed; image AUROC / AP / pixel AUROC / AUPRO):
MVTec AD 0.8 / 0.0 / 0.6 / 1.0; VisA 0.3 / 0.5 / 0.5 / 0.2; MPDD 4.4 / 3.3 / 0.8 / 0.9; BTAD 0.3 / 0.3 / 0.1 / 1.8; SDD 0.8 / 0.0 / 0.4 / 0.9; DAGM 0.5 / 0.3 / 0.7 / 0.3; DTD 0.1 / 0.0 / 0.0 / 0.6.
Absolute object-aware numbers are NOT printed. Derived (our reading, AnomalyCLIP minus gain): MVTec AD image AUROC 90.7, pixel AUROC 90.5, PRO 80.4; VisA image AUROC 81.8, pixel AUROC 95.0, PRO 86.8.

Fig. 8 (p.19), object ablation (trained object-agnostic, `object` replaced by `[cls]` at test = AnomalyCLIP_re), AnomalyCLIP vs _re: MVTec AD image AUROC 91.5 vs 89.1, AP 96.2 vs 94.5, pixel AUROC 91.1 vs 90.5, AUPRO 81.4 vs 80.6; VisA 82.1 vs 82.0, 85.4 vs 85.0, 95.5 vs 95.0, 87.0 vs 86.6.

Fig. 6 (p.10), DPAM ablation (radar chart, no printed values): Q-Q matches V-V on segmentation but is worse at image level; K-K is good at image level but worse at segmentation; V-V is the default.

Table 7 (p.18), vs full-shot methods (provenance of the PatchCore / RD4AD numbers not stated): MVTec AD image (AUROC, AP) PatchCore (99.0, 99.7), RD4AD (98.7, 99.4); pixel PatchCore (98.1, 92.8), RD4AD (97.8, 93.6). VisA image PatchCore (94.6, 95.9), RD4AD (95.3, 95.7); pixel PatchCore (98.5, 92.2), RD4AD (98.4, 91.2). (WinCLIP Table 2 prints PatchCore full-shot MVTec AD as 99.6 / 98.2; different source, do not reconcile.)

Table 8 (p.19), replacing "damaged": MVTec AD image AUROC damaged 91.5, anomalous 91.4, flawed 91.3, defective 91.4, blemished 91.5; VisA 82.1, 80.7, 80.7, 80.9, 80.7. "not sensitive" per the paper.

Hyperparameter plots (Fig. 7, p.18; read from plots, no printed values): E sweep 10-16, peak at 12; depth sweep 5-11, peak at 9; per-layer token length sweep 2-8, peak at 4; panel (d) see version notes.

## 9. Limitations the paper states
- Medical ZSAD is weaker than industrial, "partly due to the impact of auxiliary data" (Sec. 4.2); training on a medical auxiliary set helps polyp and brain data but hurts ISIC and COVID-19 (Table 3).
- Zero-shot here still needs an auxiliary anomaly dataset with pixel masks (the other benchmark's test split).
- No efficiency numbers. No explicit limitations section.
- Later fixes: not covered by the sources read (AdaCLIP 2407.15795 and UniVAD 2412.03342 are not in `sources/`).

## 10. Outline feedback (ch-anomalyclip)
- viz-anomalyclip "transfer grid" is not a grid in the paper. There are only three training sources: VisA test -> MVTec AD; MVTec AD test -> VisA, MPDD, BTAD, SDD, DAGM, DTD-Synthetic and 10 medical sets (Tables 1-2); ColonDB+Endo -> 9 medical sets (Table 3). A two-row "trained on A -> tested on B" chart fits the data; a full A x B grid would be mostly empty.
- The object-aware vs object-agnostic toggle: the paper prints only the gains (Fig. 5), not the object-aware scores. On MVTec AD and VisA the gains are 0.0 to 1.0 points; the largest is MPDD image AUROC +4.4. Show the gains with the "derived" label, or use Fig. 8 (91.5 vs 89.1 on MVTec AD image AUROC). The big jump in Table 4 (+T2, MVTec AD image AUROC 66.3 -> 90.8) mixes "learn prompts" with "drop the object name"; do not present it as the agnostic-vs-aware effect.
- L_global has no printed equation; describe it in words (cross-entropy on P(g_a, f_i) vs the image label). L_local and L_total are printed.
- "Frozen but the gradient passes through" is our reading, well supported: all CLIP weights frozen, learnable tokens only in text layers 1-9 and the prompt, losses depend on g_n, g_a only. The vision tower has no trainable part (DPAM is parameter-free).
- DPAM gloss: V-V attention from the 6th visual layer, in a parallel path that feeds only the local patch features (Fig. 2); the class token keeps the original attention. Table 4 confirms: DPAM moves pixel metrics only.
- Scoreboard protocol differs from WinCLIP: 518 px vs 240 px, ViT-L/14@336px vs ViT-B/16+, trained on the other benchmark's test split vs no training, no category name vs category name in the prompt.
- Do not say AnomalyCLIP beats WinCLIP everywhere: on MVTec AD image AUROC WinCLIP leads, 91.8 vs 91.5 (Table 1).
- Version: cite v12; flag "top feature map only" as a v12 statement that conflicts with Fig. 2 and Fig. 7(d).
- Branches AdaCLIP and UniVAD are not in `sources/`; their claims need their own extraction.
