# UniAD: A Unified Model for Multi-class Anomaly Detection

## 1. Identity
- Title: "A Unified Model for Multi-class Anomaly Detection" (method name UniAD).
- arXiv 2206.03687, read v3 (2022-10-25). v1 submitted 2022-06-08 (abs.html).
- Authors: Zhiyuan You (first, equal contribution with Lei Cui) ... Xinyi Le (last, corresponding). Shanghai Jiao Tong Univ., Tsinghua, CUHK, SenseTime.
- Venue: "Accepted by NeurIPS 2022" (abs.html Comments). main.tex loads `neurips_2022` with `[final]`.
- Repo: https://github.com/zhiyuanyou/UniAD (abstract).
- Sources: `sources/uniad/src/` (sections/*.tex, supp_sections/*.tex; `rebuttal.tex` is in the tarball but NOT compiled into the paper, not used here), `paper.pdf` (19 pages).

## 2. Core problem
One-class-one-model AD needs a separate model per product; this costs memory as classes grow and fails when one "class" is internally diverse. UniAD trains ONE model on normal images of all 15 MVTec AD classes, with no class label at train or test time ("unified case"), and asks it to detect and localize anomalies in every class.

## 3. What prior methods did and why that hurt (per the paper)
- Existing methods (US, PaDiM, CutPaste, FCDD, PSVDD, ADTR, DRAEM) train separate models per class (Sec. 1).
- Reconstruction networks (image or feature) assume the model only outputs normal content. In practice they fall into the "identical shortcut": they return a copy of the input, so defects are reconstructed too and give low error (Sec. 1, Sec. 3.1).
- The unified case "magnifies" the shortcut: learning the joint distribution of 15 classes is hard, so copying is the easier solution (Sec. 1; Fig. 4).
- Wording: the paper uses "identical shortcut" (abstract, Sec. 3.1, Fig. 2/3/4 captions, conclusion) and also "identity shortcut" (Sec. 1, twice). Sec. 3.1 definition, verbatim: `We attribute this to the problem of ``identical shortcut'', where both normal and anomalous regions can be well recovered, thus failing to spot anomalies.` Fig. 2 caption: `the model learning an ``identical shortcut'', which tends to directly copy the input as the output regardless of whether it is normal or anomalous.`
- Evidence for the shortcut:
  - Fig. 2a (p. 4): MLP, CNN and a transformer with one query embedding reconstruct frozen-backbone features on MVTec AD (unified). Training loss (blue) falls toward zero over 1000 epochs, while test image AUROC (green) and pixel AUROC (red) peak early and then drop. "The metric is evaluated every 10 epochs. Note that the periodic evaluation is impractical since anomalies are not available during training." The transformer drops much less than MLP and CNN.
  - Fig. 2b (p. 4): features rendered back to images by a separate decoder (trained on normal and anomalous samples, used only for visualization; reversed ResNet-34, App. B.3). MLP/CNN/transformer reproduce the defect; UniAD reconstructs it as normal. More in Fig. A2.
  - Fig. 4 (p. 5): same MLP experiment, unified vs separate case. Separate case: AUROC keeps rising as the loss falls. Unified case: smaller loss, much worse AUROC. Paper: "the unified case is more challenging and hence magnifies the ``identical shortcut'' problem".
  - Sec. 3.1 one-layer argument: FC layer can learn w -> I, b -> 0 (Eq. 1); a 1x1 conv is an FC layer and larger kernels can do the same; attention with a learnable query (Eq. 2) must make softmax(q x^T / sqrt(C)) ~ I, so q must be tied to normal x+, and cannot reproduce x-.
  - Table 5a / A2: no query embedding 69.5 / 79.4 (Det. / Loc.) vs one query 87.6 / 92.8.
- Pseudo-anomaly methods (CutPaste, FCDD, DRAEM) rely on how well fakes match real defects (Sec. 2). In the unified case DRAEM "ceases to be effective" (Sec. 2); FCDD drops 28.7 pixel AUROC, "reflecting the pseudo-anomaly is not suitable for the unified case" (Sec. 4.3).

## 4. The one key idea
Reconstruct frozen CNN features with a transformer that is structurally prevented from copying its input: a learnable query embedding in every decoder layer (LQD), attention that hides each token's own 7x7 neighborhood (NMA), and Gaussian noise on the input tokens so the task is denoising (FJ). The reconstruction error is the anomaly map.

## 5. Mechanism, step by step
1. Input image 224x224 -> frozen ImageNet EfficientNet-b4. Take stage-1 to stage-4 outputs ("stage" = blocks with the same feature-map size).
2. Resize all four maps to 14x14 and concatenate on channels -> f_org in R^{272 x 14 x 14} (C_org = 272, Sec. 4.2).
   - `our reading`: stage strides 2/4/8/16 give 112/56/28/14 px maps, channels 24 + 32 + 56 + 160 = 272. The paper gives only the total. Cross-check: the CIFAR-10 config uses stages 1-5 = 720 channels, and 272 + 448 = 720.
3. Tokenize: H x W = 14 x 14 = 196 tokens (`our reading`: arithmetic from the stated 14x14), each 272-d. Linear projection 272 -> C = 256.
4. Feature jittering (training): add Gaussian noise D (Eq. 3) to each token with probability p = 1. Fig. 3 draws the perturbation added to the feature tokens at the encoder input. `our reading`: f_tok in R^C means after projection to 256; noise is a training-only step (paper says it turns training into a denoising task).
5. Neighbor Masked Encoder (NME): 4 layers, each NMA + FFN (256 -> 1024 -> 256, ReLU; Table A1), residual + layer norm. Learnable position embeddings are added in the attention modules.
6. Neighbor Masked Attention: in the attention map, each query token cannot attend to keys in its own 2D neighborhood (7x7 by default), itself included (Fig. 5 shows a toy 3x3 mask on a 5x5 grid; the default is 7x7). `our reading`: on 14x14 an interior token masks 49 of 196 keys and sees 147.
7. Layer-wise Query Decoder (LQD): 4 layers. In layer i, a learnable query embedding q^i attends (NMA) to the encoder output (K, V = encoder embeddings); the result then attends (NMA, as query) to the previous decoder layer's output as K, V (layer 1: self-fusion); then a 2-layer FFN; residuals throughout. This is design (a) of six tried (Fig. A1, Table A3).
8. Linear projection 256 -> 272, reshape -> f_rec in R^{272 x 14 x 14}.
9. Loss: MSE between f_org (clean, un-jittered backbone features) and f_rec (Eq. 4). Gradient flows into: the two linear projections, NME, LQD, query embeddings, position embeddings. The backbone is frozen; Table A5: training it gives 65.9 / 65.5 vs 96.5 / 96.8 frozen ("the backbone [would] extract some indiscriminative features that are easy to reconstruct").
10. Test: anomaly map s = per-token L2 norm of f_org - f_rec (Eq. 5), 14x14, bilinear upsample to image size. Image score = maximum of the average-pooled s (pool size not stated in the paper). Fig. 3 labels this "Avg Pool & Max".

## 6. Equations
Custom macros: none beyond `\bm` (bm package) and `\mathtt`. Main-text equations carry no `\label` except Eq. 4 and 5; numbers below are from the PDF.

Eq. (1), Sec. 3.1, FC layer:
```
\bm{y} = \bm{x}^+ \bm{w} + \bm{b} \in \mathbb{R}^{K \times C}.
```
Paper: "With the MSE loss pushing y to x+, the model may take shortcut to regress w -> I (identity matrix), b -> 0. Ultimately, this model could also reconstruct x- well, failing in anomaly detection."
Gloss: x+ = K normal feature tokens of C channels; w = C x C weights; b = bias; y = reconstruction.

Eq. (2), Sec. 3.1, attention with learnable query:
```
\bm{y} = \mathtt{softmax}(\bm{q} (\bm{x}^{+})^T / \sqrt{C}) \bm{x}^+ \in \mathbb{R}^{K \times C}.
```
Paper: "To push y to x+, the attention map ... should approximate I (identity matrix), so q must be highly related to x+. Considering that q in the trained model is relevant to normal samples, the model could not reconstruct x- well."
Gloss: q in R^{K x C} = learnable query embedding (one per token position); keys and values are the input tokens.

Eq. (3), Sec. 3.2, feature jittering:
```
D \sim N(\mu=0, \sigma^2=(\alpha \frac{||\bm{f}_{tok}||_2}{C})^2),
```
Paper: "where alpha is the jittering scale to control the noisy degree. Also, the sampled disturbance is added to f_tok with a fixed jittering probability, p."
Gloss: f_tok in R^C = one feature token; ||f_tok||_2 = its length; C = channels; noise std = alpha * ||f_tok|| / C. Defaults alpha = 20, p = 1.
`our reading`: with C = 256, std = (20/256) ||f_tok|| = 1.25 x the token's per-channel RMS (||f|| = 16 x RMS when C = 256). The noise is larger than the signal per channel.

Eq. (4), `equ:mse_loss`, objective:
```
\mathcal{L} = \frac{1}{H \times W} ||\bm{f}_{org} - \bm{f}_{rec}||_2^2.
```
Paper: "Our model is trained with the MSE loss". Gloss: f_org = frozen backbone feature map (C_org x H x W); f_rec = reconstruction; averaged over the H x W = 196 positions. Same as Eq. (A1). Appendix alternatives: normalized MSE (A2) 96.4 / 96.9, cosine (A3) 96.6 / 96.8, vs MSE 96.5 / 96.8 (Table A4).

Eq. (5), `equ:score_map`, anomaly map:
```
\bm{s} = ||\bm{f}_{org} - \bm{f}_{rec}||_{2} \in \mathbb{R}^{H \times W}.
```
Paper: "the anomaly score map, s, is calculated as the L2 norm of the reconstruction differences". Then "up-sampled to the image size with bi-linear interpolation". Image score: "taking the maximum value of the averagely pooled s".

No equation is given for the neighbor mask itself; it is defined in words and Fig. 5 ("a feature point relates to neither itself nor its neighbors"; "the neighbor region is defined in the 2D space").

## 7. Config / setup (MVTec AD, Sec. 4.2 unless noted)
| item | value | where |
|---|---|---|
| input | 224 x 224 | Sec. 4.2 |
| backbone | EfficientNet-b4, ImageNet-pretrained, frozen ("fixed") | Sec. 3.3; Table A5 |
| layers | stage-1 to stage-4, resized to 14 x 14, concatenated | Sec. 3.3, 4.2 |
| C_org | 272 | Sec. 4.2 |
| per-stage channels / sizes | 24/32/56/160 at 112/56/28/14 | `our reading`, not in paper |
| tokens | 14 x 14 = 196 | `our reading` (arithmetic) |
| reduced dim C | 256 | Sec. 4.2 |
| FFN | 256 -> 1024 (ReLU) -> 256 | Table A1 |
| encoder / decoder layers | 4 / 4 | Sec. 4.2; Table 5b |
| attention heads | not stated | - |
| position embedding | learnable, added in attention modules | Sec. 3.3 |
| neighbor size | 7 x 7 | Sec. 4.2; Table 5c |
| jitter scale alpha / prob p | 20 / 1 | Sec. 4.2; Table 5e, 5f |
| loss | MSE on features | Eq. 4 |
| optimizer | AdamW, weight decay 1e-4 | Sec. 4.2 |
| LR | 1e-4, x0.1 after 800 epochs | Sec. 4.2 |
| epochs | 1000 | Sec. 4.2 |
| batch | 64 | Sec. 4.2 |
| hardware | 8 GPUs, NVIDIA Tesla V100 16GB | Sec. 4.2 |
| seeds | 5 | Sec. 4.2 |
| learnable params | 7.48 M | Table A7 (p. 17) |
| FLOPs | 6.46 G at 224 x 224 | Table A7 |
| CIFAR-10 config | stages 1-5, 720 ch, batch 128, rest same | App. B.4 |
| trained from scratch | reconstruction models, both cases | Sec. 4.2 |

No latency or throughput is reported. No VisA results and no PRO / AU-PRO: the paper reports only AUROC (MVTec AD image and pixel, CIFAR-10 image).

## 8. Results worth remembering
Protocol for all MVTec AD rows: 224 px, MVTec AD 15 classes, AUROC (%). "unified" = one model for all 15 classes; "separate" = one model per class.

Table 1 (p. 7), image AUROC, Mean row, unified / separate:
| method | unified | separate | drop (our reading, separate - unified) |
|---|---|---|---|
| US | 74.5 | 87.7 | 13.2 |
| PSVDD | 76.8 | 92.1 | 15.3 |
| PaDiM | 84.2 | 95.5 | 11.3 |
| CutPaste | 77.5 | 96.1 | 18.6 (stated in text) |
| MKD | 81.9 | 87.8 | 5.9 |
| DRAEM | 88.1 | 98.0 | 9.9 (text: "near 10%") |
| UniAD | 96.5 +- 0.08 | 96.6 | 0.1 |

Table 2 (p. 8), pixel AUROC, Mean row, unified / separate (FCDD replaces CutPaste):
| method | unified | separate | drop (our reading) |
|---|---|---|---|
| US | 81.8 | 93.9 | 12.1 (stated) |
| PSVDD | 85.6 | 95.7 | 10.1 |
| PaDiM | 89.5 | 97.4 | 7.9 |
| FCDD | 63.3 | 92 | 28.7 (stated) |
| MKD | 84.9 | 90.7 | 5.8 |
| DRAEM | 87.2 | 97.3 | 10.1 |
| UniAD | 96.8 +- 0.02 | 96.6 | -0.2 (gain) |

Provenance (Sec. 4.2 "Baselines"): separate-case numbers are COPIED from each method's own paper, except US, which is copied from the DRAEM paper. Unified-case numbers are RE-RUN by UniAD with public code: US (denguir/student-teacher-anomaly-detection), PSVDD (nuclearboy95), PaDiM (xiahaifeng1995), CutPaste (Runinho/pytorch-cutpaste), MKD (Niousha12), DRAEM (VitjanZ/DRAEM), FCDD (liznerski/fcdd). `our reading`: the US, PaDiM and CutPaste repos are third-party reimplementations. PatchCore does not appear in any UniAD table.

Text claims: UniAD beats DRAEM by 8.4 (image, unified) and PaDiM by 7.3 (pixel, unified); abstract: "from 88.1% to 96.5%" (detection) and "from 89.5% to 96.8%" (localization).

Table 4 (p. 9), transformer competitors, Det. / Loc., unified / separate: InTra 65.3/95.0, 70.6/96.6; VT-ADL 55.4/78.7, 64.4/82.0; AnoViT 69.6/78, 68.4/83; UniAD baseline (1 query) 87.6/94.7, 92.8/95.8; UniAD 96.5/96.6, 96.8/96.6.

Table 5 (p. 10), ablations, unified, Det. / Loc.:
- (a) w/o query 69.5 / 79.4; 1 query (vanilla transformer) 87.6 / 92.8; layer-wise query 95.0 / 96.5; 1 query + NMA 96.1 / 96.3; 1 query + FJ 95.0 / 95.8; all three 96.5 / 96.8. Text deltas: query +18.1 / +13.4; layer-wise +7.4 / +3.7; NMA +8.5 / +3.5; FJ +7.4 / +3.0.
- Table A2 adds: 1 query + NMA + FJ 96.2 / 96.6; layer-wise + NMA 95.8 / 96.5; layer-wise + FJ 94.9 / 96.2 ("FJ must cooperate with NMA").
- (b) #Enc, #Dec (ours): 4,0 94.9/96.0; 0,4 96.1/96.3; 2,2 95.1/96.0; 4,4 96.5/96.8; 6,6 96.5/96.7. Vanilla: 69.8/79.2, 80.5/88.3, 84.7/90.6, 87.6/92.8, 86.1/91.9.
- (c) neighbor size: 1x1 94.6/96.3; 5x5 96.4/96.8; 7x7 96.5/96.8; 9x9 96.3/96.7.
- (d) NMA place: Enc 95.8/96.3; Enc+Dec1 96.4/96.8; Enc+Dec2 96.5/96.7; All 96.5/96.8.
- (e) alpha: 5 96.1/96.7; 10 96.4/96.7; 20 96.5/96.8; 30 95.7/96.6.
- (f) p: 0.25 95.6/96.5; 0.50 95.8/96.7; 0.75 96.3/96.7; 1 96.5/96.8.
Table A6 (p. 17) backbone, Det. / Loc.: Res-18 92.4/95.8, Res-34 93.0/96.2, Res-50 92.4/96.0, Res-101 92.2/95.9, Eff-b0 96.1/96.4, Eff-b2 96.2/97.0, Eff-b4 96.5/96.8, Eff-b6 96.0/96.7.
Table A7 (p. 17), FLOPs (G) / learnable params (M) at 224: US 60.32/9.55, PSVDD 149.74/0.41, PaDiM 23.25/950.36, CutPaste 3.65/13.61, FCDD 13.16/4.51, MKD 32.11/0.34, DRAEM 245.15/69.05, UniAD 6.46/7.48.
Table 3 (p. 9), CIFAR-10 unified (5 normal classes vs 5 anomalous), mean image AUROC: UniAD 87.2 +- 0.03; FCDD+OE 78.9; PANDA 72.4; MKD 72.1; US 55.9; FCDD 55.8.

## 9. Limitations stated
- Class labels are never used; "How to incorporate the unified model with category labels should be further studied" (Sec. 5 Discussion).
- Real normal data is less consistent than MVTec AD (Sec. 5).
- Privacy risk in video surveillance (Sec. 5).
- Not stated but visible in the sources: 1000 epochs; ADer (2406.03262) reports UniAD "require[s] more epochs to converge" and scores 92.5 image AUROC at 100 epochs vs 96.8 at 300 (ADer Table 3).
- Later fixes: the sources here do not name a paper that fixes these. ADer's tables show later multi-class methods (ViTAD, InvAD, MambaAD) above UniAD at 100 and 300 epochs.

## 10. Outline feedback
1. ch-uniad intuition names only neighbor masking and noise. The paper's own ablation makes the learnable query embedding the largest effect (69.5 -> 87.6 for one query, -> 95.0 layer-wise; Table 5a). Add "a learned query in every decoder layer" to the intuition, and do not let `viz-uniad` imply that masking alone fixes the shortcut. NMA alone on top of one query does give 96.1 / 96.3.
2. `viz-uniad` 1D row of tokens: the real mask is a 2D 7x7 window on a 14x14 grid, and the paper's Fig. 5 draws a toy 3x3. Say "toy" and keep 2D if possible; Table 5c shows 1x1 (mask only self) is too small (94.6), because neighbors leak the token.
3. "it learns to copy its input" is supported by Fig. 2a/2b and Fig. 4; Fig. 4 (unified vs separate MLP curves) is the cleanest single visual for "the unified case magnifies the shortcut" and could be redrawn as a schematic.
4. The outline's "how much one-class methods drop" should use UniAD Table 1/2 (re-run baselines) for PaDiM, CutPaste, DRAEM, US, PSVDD, MKD, FCDD. PatchCore is NOT in UniAD; only ADer has PatchCore multi-class.
5. UniAD has no VisA and no PRO numbers. The closing scoreboard can only take MVTec AD image and pixel AUROC from UniAD itself; VisA and AU-PRO for UniAD come only from ADer (different protocol).
6. Architecture walkthrough: say the gradient also reaches the learnable query and position embeddings, and that the MSE target is the clean (un-jittered) frozen feature.
