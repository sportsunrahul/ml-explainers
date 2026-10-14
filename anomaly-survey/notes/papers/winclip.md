# WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

## 1. Identity
- Title: "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation".
- arXiv 2303.14814, version read: v1 (the only version; abs.html submission history).
- Published: v1 2023-03-26 (abs.html).
- Authors: Jongheon Jeong (KAIST, first, equal contribution) ... Onkar Dabeer (AWS AI Labs, last). Full list: Jeong, Zou, Kim, Zhang, Ravichandran, Dabeer.
- Venue: CVPR 2023 (abs.html comments; `\confName{CVPR}`, `\confYear{2023}` in main.tex).
- Repo: none given in the paper. It uses OpenCLIP and the official PatchCore code for a baseline.
- Sources: `sources/winclip/src/main.tex`, `src/appendix.tex`, `src/tables/*.tex`; `sources/winclip/paper.pdf` (21 pages).
- Table numbering trap: there are TWO "Table 8"s. Main Table 8 (p.8) = `tables/al_0shot.tex` (AS ablation + time). Supplement Table 8 (p.13) = `tables/ablation_patchcore.tex` (caused by `\setcounter{table}{7}` in appendix.tex). Supplement Table 9 (p.13) = backbones. Every loc below carries the page.

## 2. Core problem
Industrial inspection trains one model per product from hundreds of normal images. WinCLIP asks for anomaly classification (AC, image level) and segmentation (AS, pixel level) with zero normal images (zero-shot) or 1 to 4 normal images (few-normal-shot), with no training at all.

## 3. What prior methods did and why that hurt (per the paper)
- One-class methods (PatchCore, PaDiM, SPADE, CutPaste, DRAEM, ...) fit a model per task on many normal images; not scalable across the long tail of tasks (Sec. 1, Sec. 2).
- Few-shot methods (DifferNet, TDG) augment a small support set; RegAD pretrains a registration network on other categories. Still need in-task images (Sec. 2).
- Naive CLIP zero-shot with "normal [o]" vs "anomalous [o]" is weak: 74.0 AUROC on MVTec AD (Table 1, p.7). The CLIP ImageNet prompt ensemble does not help (74.1).
- CLIP aligns only the global embedding with text. Its last-layer patch tokens are not aligned with language: 22.4 pAUROC on MVTec AD (main Table 8, p.8, "Patch-token").
- Generic CLIP localizers (Trans-MM, MaskCLIP) give poor anomaly maps (Table 4, p.8).

## 4. The one key idea
Ask CLIP to compare each image window, not each patch token, with an ensemble of "normal" and "anomalous" text prompts. A window's class token is directly aligned with text, so the window score is a valid zero-shot anomaly score; spreading window scores back to pixels gives a map.

## 5. Mechanism, step by step
Nothing is trained, in either WinCLIP or WinCLIP+. CLIP stays frozen; no weight gets a gradient. "Fitting" for WinCLIP+ is storing features of K normal images.

Text side (once per category):
1. Build all compositions of state words x templates (supp. Fig. 6, p.12). Normal states: 7 (`[o]`, `flawless [o]`, `perfect [o]`, `unblemished [o]`, `[o] without flaw`, `[o] without defect`, `[o] without damage`). Anomaly states: 4 (`damaged [o]`, `[o] with flaw`, `[o] with defect`, `[o] with damage`). Templates: 22 (10 in the first column + 12 "cont'd"), e.g. "a cropped photo of the [c].", "a photo of a [c] for visual inspection.", "a photo of the [c] for anomaly detection.".
2. `[o]` is replaced by the object name word of the category, available in MVTec AD and VisA (supp. Sec. A). So zero-shot WinCLIP uses the category name. Optional task-specific states (e.g. "bad soldering") are used only in Table 7.
3. Encode every prompt with the CLIP text encoder g, then "compute the average of text embeddings per label" (Sec. 4.1). Result: one normal and one anomalous class prototype. Prompt counts 7 x 22 = 154 normal and 4 x 22 = 88 anomalous are computed (our reading). Whether embeddings are L2-normalized before averaging is not stated.
4. ascore_0 : R^d -> [0, 1] is the two-class zero-shot score on an image embedding (Sec. 4.1). The paper defines it in words only; our reading: Eq. (1) with S = {normal prototype, anomalous prototype}, returning the probability of the anomalous class. tau is not given.

Image side, zero-shot (WinCLIP):
5. Preprocess: OpenCLIP normalization, bicubic resize so the shorter edge is 240 (supp. Sec. A). Non-square VisA images: overlapping square tiles of side = shorter edge, stride at most 0.8 x shorter edge, predictions averaged over tiles (supp. Sec. A (ii)).
6. Backbone: OpenCLIP ViT-B/16+ pretrained on LAION-400M (Sec. 5). 240^2 input, 225 patch tokens (supp. Sec. A (i)) = 15 x 15 grid of 16 px patches (our reading).
7. Windows (Eq. 2): binary masks w_ij active on a k x k patch kernel. Small scale 2 x 2 patches (32 x 32 px), mid scale 3 x 3 patches (48 x 48 px), stride 1 patch = 16 px (Sec. 4.2, Sec. 5). Each window is a separate forward pass of the frozen CLIP image encoder on only the unmasked patches ("dropping all the masked patches before forwarding them, in a similar manner to masked autoencoder", Sec. 4.2). The window embedding is the output class token (Fig. 3). Window counts at stride 1 with no padding: 14 x 14 = 196 small, 13 x 13 = 169 mid (our reading; padding not stated).
8. Image scale: the class token of the full image (Sec. 4.2 "Multi-scale aggregation" (c); Fig. 4 shows "Class token" entering "Multi-scale Aggregation"). Ambiguity: the ablation text says "the scores from last patch embeddings of ViT encodes global context ... contributes to a comprehensive localization" (Sec. 5.4). Our reading: the zero-shot image scale is the class-token score, constant over pixels.
9. Window score M^W_{0,ij} = ascore_0(F^W_ij), "distributed to every pixel of the local window" (Sec. 4.2).
10. Per pixel, harmonic mean over all windows that cover it (Eq. 3), which "weight[s] more on scores towards normality prediction (zero value)". Scales are then also combined by harmonic averaging (Sec. 4.2, last sentence of "Multi-scale aggregation").
11. Image score (zero-shot AC): ascore_0(f(x)) on the full-image embedding, plus "multi-crop prediction" (Table 3 row (d), p.7). Multi-crop is not described further.

Few-shot (WinCLIP+), K = 1, 2, 4 normal images (8 only in Tables 5-6):
12. Build three reference memories from the K images: R^W_s (small-window embeddings), R^W_m (mid-window embeddings), R^P (last-layer patch tokens F^P, not language aligned but "still useful") (Sec. 4.3). No subsampling is stated. Per reference image: 196 + 169 + 225 = 590 vectors (our reading).
13. Per scale, score each query feature by its cosine-distance to the nearest memory entry (Eq. 4).
14. Average the three maps (Eq. 5), then "fusing with our language-guided prediction" \bar{M}^W_0. The fusion formula is NOT given (prose only); Fig. 4 draws a circled plus.
15. Image score: mean of ascore_0(f(x)) and the max of the vision map (Eq. 6).

## 6. Equations (verbatim TeX from main.tex)
Custom macros: `\def\rvx{{\mathbf{x}}}`, `\def\rvs{{\mathbf{s}}}` (main.tex preamble). `\dev{x}` = footnotesize plus-minus x (std over 5 seeds).

Eq. (1), `\label{eq:clip_0shot}`, Sec. 3:
```tex
p(\rvs = s_i|\rvx; \rvs \in S) := \frac{\exp(\langle f(\rvx), g(s_i) \rangle / \tau) }{\sum_{s \in S} \exp(\langle f(\rvx), g(s) \rangle / \tau) },
```
Paper: CLIP "can perform zero-shot classification via a k-way categorical distribution", tau > 0 the temperature. Gloss: f = CLIP image encoder, g = text encoder, <.,.> = cosine similarity, S = the set of class texts. WinCLIP uses k = 2 (normal vs anomalous), with each class text replaced by its CPE prototype (our reading, see step 4). The one-class variant scores `-<f(x), g(s_-)>` (Sec. 4.1, inline).

Eq. (2), `\label{eq:winclip}`, Sec. 4.2:
```tex
\mathbf{F}^{\tt W}_{ij} := f(\rvx \odot \mathbf{w}_{ij}),
```
Paper: "Collect each output embedding F^W_ij, computed from the active area of x after applying each w_ij", odot = element-wise product. Gloss: w_ij = binary k x k window mask around (i, j); F^W_ij = CLIP embedding (class token) of only that window.

Eq. (3), `\label{eq:harmonic}`, Sec. 4.2:
```tex
\bar{\mathbf{M}}^{\tt W}_{0,ij} := \left(\frac{1}{\sum_{u,v} {(\mathbf{w}_{uv})_{ij}}} \sum_{u,v} \frac{(\mathbf{w}_{uv})_{ij}}{\mathbf{M}^{\tt W}_{0,uv}} \right)^{-1}.
```
Paper: "at each pixel, we aggregate multiple scores from all overlapping windows ... by harmonic averaging, weighting more on scores towards normality prediction (zero value)". Gloss: (w_uv)_ij = 1 if window uv covers pixel ij; M^W_{0,uv} = zero-shot score of window uv; the result is the harmonic mean of the scores of all windows covering the pixel. One near-zero (confidently normal) window pulls the pixel toward normal.

Eq. (4), `\label{eq:memory_al}`, Sec. 4.3:
```tex
\mathbf{M}_{ij} := \min_{r \in \mathbf{R}} \tfrac{1}{2} (1 - \langle \mathbf{F}_{ij}, r \rangle).
```
Paper: the reference association "can simply store and retrieve the memory features R of D based on the cosine similarity". Gloss: F_ij = query feature at location ij; R = stored features of the K normal images; score = half the cosine distance to the nearest stored feature, in [0, 1].

Eq. (5), `\label{eq:winplus_al}`, Sec. 4.3:
```tex
\mathbf{M}^{\tt W} := \tfrac{1}{3}(\mathbf{M}^{\tt P} + \mathbf{M}^{\tt W}_{\tt s} + \mathbf{M}^{\tt W}_{\tt m}),
```
Paper: "we average their multi-scale predictions (4) for anomaly segmentation for a given query, ... and then fusing with our language-guided prediction". Gloss: M^P, M^W_s, M^W_m = Eq. (4) maps from the patch-token, small-window and mid-window memories.

Eq. (6), `\label{eq:winplus_ad}`, Sec. 4.3:
```tex
\mathrm{ascore}_{\tt W}(\rvx) := \tfrac{1}{2}\left(\mathrm{ascore}_0(f(\rvx)) + \max_{ij} \mathbf{M}^{\tt W}_{ij} \right).
```
Paper: "we combine the maximum value of M^W and the WinCLIP zero-shot classification score", which carry "complementary information". Gloss: half language score of the whole image, half the worst (highest) vision-memory score anywhere in the map.

## 7. Config / setup (method lens)
| Item | Value | Where |
|---|---|---|
| Backbone | OpenCLIP ViT-B/16+, LAION-400M weights, frozen | Sec. 5 "Implementation details" |
| ViT-B/16+ vs ViT-B/16 | image width 768 -> 896, text width 512 -> 640, input 224^2 -> 240^2, tokens 196 -> 225 | supp. Sec. A (i) |
| Input | shorter edge 240, bicubic (Pillow), OpenCLIP mean/std | supp. Sec. A "Data pre-processing" |
| Patch grid | 15 x 15 of 16 px | our reading (225 tokens, 240/16) |
| Window kernels | 2 x 2 patches (32 px), 3 x 3 patches (48 px), plus image-scale class token | Sec. 4.2 |
| Window stride | 1 patch = 16 px | Sec. 5 |
| Windows per scale | 196 (2 x 2), 169 (3 x 3) | our reading, no padding assumed |
| Window masking | drop masked patches before the forward pass (MAE style) | Sec. 4.2 |
| Prompts | 7 normal states, 4 anomaly states, 22 templates; class prototype = mean text embedding | supp. Fig. 6, Sec. 4.1 |
| Aggregation | harmonic mean over windows (Eq. 3) and over scales | Sec. 4.2 |
| Few-shot memories | R^W_s, R^W_m, R^P; nearest neighbour by cosine | Sec. 4.3 |
| Shots | 0, 1, 2, 4 (8 in ablation Tables 5-6, MVTec AD only) | Tables 1, 4, 5, 6 |
| Seeds | 5 random seeds for few-shot; zero-shot std 0.0 | Table 1/4 captions |
| Trainable | nothing | Sec. 1 "our zero-shot models do not require any tuning" |
| Baseline input | 240 shorter edge for all baselines | supp. Sec. A |
| Latency | 389.4 +- 18.5 ms per image (zero-shot WinCLIP), Amazon EC2 G4dn; batch size not stated; GPU model not named | main Table 8, p.8 |
| Compute other | none stated |  |

Tensor trace (zero-shot, one 240 x 240 image; widths per supp. Sec. A; joint embedding dim d is not stated):
- 3 x 240 x 240 -> 225 patch tokens + CLS at width 896 (full image) -> CLS -> d-dim image embedding -> ascore_0 -> image score.
- For each of 196 small windows: 4 tokens + CLS -> CLS -> d-dim -> ascore_0 -> one score painted on a 32 x 32 px block.
- For each of 169 mid windows: 9 tokens + CLS -> same, 48 x 48 px block.
- Harmonic means -> 240 x 240 map (our reading for the output grid; resizing to the original resolution not stated).
- Backward: none.

## 8. Results worth remembering
Protocol for all rows: one prompt set and (for WinCLIP+) one memory per category; MVTec AD and VisA; 240 px shorter edge; ViT-B/16+; few-shot = mean of 5 seeds.

Abstract headline: zero-shot AC/AS AUROC 91.8/85.1 (MVTec AD) and 78.1/79.6 (VisA); 1-shot WinCLIP+ 93.1/95.2 (MVTec AD) and 83.8/96.4 (VisA).

Table 1 (p.7), AC, AUROC / AUPR / F1-max:
| Setup | Method | MVTec AD | VisA |
|---|---|---|---|
| 0-shot | CLIP-AC | 74.0 / 89.1 / 88.5 | 59.3 / 67.0 / 74.4 |
| 0-shot | + Prompt ens. | 74.1 / 89.5 / 87.8 | 58.2 / 66.4 / 74.0 |
| 0-shot | WinCLIP | 91.8 / 96.5 / 92.9 | 78.1 / 81.2 / 79.0 |
| 1-shot | PatchCore | 83.4 / 92.2 / 90.5 | 79.9 / 82.8 / 81.7 |
| 1-shot | WinCLIP+ | 93.1 (+-2.0) / 96.5 / 93.7 | 83.8 (+-4.0) / 85.1 / 83.1 |
| 2-shot | PatchCore | 86.3 / 93.8 / 92.0 | 81.6 / 84.8 / 82.5 |
| 2-shot | WinCLIP+ | 94.4 / 97.0 / 94.4 | 84.6 / 85.8 / 83.0 |
| 4-shot | PatchCore | 88.8 / 94.5 / 92.6 | 85.3 / 87.5 / 84.3 |
| 4-shot | WinCLIP+ | 95.2 / 97.3 / 94.7 | 87.3 / 88.8 / 84.2 |
SPADE 1/2/4-shot MVTec AD AUROC 81.0 / 82.9 / 84.8; PaDiM 76.6 / 78.9 / 80.4. Baselines are WinCLIP's own runs (own SPADE/PaDiM reimplementation, official PatchCore code, 240 px input; supp. Sec. A (iii)).

Table 4 (p.8), AS, pAUROC / PRO / F1-max:
| Setup | Method | MVTec AD | VisA |
|---|---|---|---|
| 0-shot | Trans-MM | 57.5 / 21.9 / 12.1 | 49.4 / 10.2 / 3.1 |
| 0-shot | MaskCLIP | 63.7 / 40.5 / 18.5 | 60.9 / 27.3 / 7.3 |
| 0-shot | WinCLIP | 85.1 / 64.6 / 31.7 | 79.6 / 56.8 / 14.8 |
| 1-shot | PatchCore | 92.0 / 79.7 / 50.4 | 95.4 / 80.5 / 38.0 |
| 1-shot | WinCLIP+ | 95.2 (+-0.5) / 87.1 / 55.9 | 96.4 (+-0.4) / 85.1 / 41.3 |
| 2-shot | WinCLIP+ | 96.0 / 88.4 / 58.4 | 96.8 / 86.2 / 43.5 |
| 4-shot | PatchCore | 94.3 / 84.3 / 55.0 | 96.8 / 84.9 / 43.9 |
| 4-shot | WinCLIP+ | 96.2 / 89.0 / 59.5 | 97.2 / 87.6 / 47.0 |
SPADE 1-shot MVTec AD 91.2 / 83.9 / 42.4 is the runner-up on PRO.

Table 2 (p.7), MVTec AD, AC / AS AUROC; non-WinCLIP rows are cited from their papers (not re-run): WinCLIP 0-shot 91.8 / 85.1; WinCLIP+ 1-shot 93.1 / 95.2; 4-shot 95.2 / 96.2; DifferNet 16-shot 87.3 / -; TDG 10-shot 78.0 / -; RegAD-L 2-shot 81.5 / 93.3; RegAD 4 + agg. 88.2 / 95.8; MKD full 87.7 / 90.7; P-SVDD full 92.1 / 95.7; CutPaste full 95.2 / 96.0; PatchCore full 99.6 / 98.2.

Table 3 (p.7), zero-shot AC ablation on MVTec AD, AUROC: (a) One-class 34.2; Two-class 74.0; (b) + State ens. 89.8; (c) + Prompt ens. 90.8; (d) + Multi-crop 91.8. The state ensemble gives most of the gain.

Main Table 8 (p.8), zero-shot AS on MVTec AD, pAUROC / PRO / F1-max / ms per image (EC2 G4dn):
Patch-token 22.4 / 2.3 / 8.0 / 95.5; Image tiling 77.9 / 57.5 / 25.5 / 1442.1; WinCLIP 85.1 / 64.6 / 31.7 / 389.4; w/o image-scale 82.0 / 63.0 / 29.5 / 378.6; w/o mid-scale 84.0 / 61.6 / 30.5 / 190.7; w/o small-scale 84.7 / 63.6 / 30.6 / 265.4; w/o Harmonic avg. 81.5 / 60.5 / 27.3 / 279.9.
(Unused source file `tables/ablation_multiscale.tex` has different numbers; it is not `\input` anywhere; ignore it.)

Table 5 (p.8), WinCLIP+ AC ablation, MVTec AD AUROC at 1 / 2 / 4 / 8 shots: max M^W only 87.9 / 91.0 / 92.6 / 94.5; ascore_0 only 91.8 at every shot; both 93.1 / 94.4 / 95.2 / 96.3.
Table 6 (p.8), WinCLIP+ AS ablation, MVTec AD pAUROC at 1 / 2 / 4 / 8: M^P only 94.5 / 94.8 / 95.4 / 95.8; M^P + M^W_m 95.1 / 95.7 / 96.3 / 96.6; all three 95.2 / 96.0 / 96.2 / 96.5.
Table 7 (p.8), VisA zero-shot AC AUROC with task-specific state words: mean 78.1 -> 78.9; PCB2 51.2 -> 56.5; PCB4 79.6 -> 82.7; Pipe fryum 69.7 -> 70.4.
Supp. Table 9 (p.13), zero-shot MVTec AD AC / AS by backbone: RN50 224 79.8 / 65.6; RN101 224 79.2 / 62.4; RN50x4 288 81.9 / 71.3; RN50x16 384 82.3 / 65.3; ViT-B/16 224 86.1 / 71.1; ViT-B/16+ 240 90.8 / 85.1; ViT-L/14 224 86.1 / 64.4. Note: AC 90.8 here equals Table 3 row (c), not the headline 91.8 (our reading: no multi-crop in this table).
Supp. Table 8 (p.13), CLIP-feature PatchCore: its column headers are garbled (pAUROC/PRO under "classification") and its WinCLIP+ AUROC (92.7 / 94.0 / 94.8 at 1/2/4 shots) differs from Table 1 (93.1 / 94.4 / 95.2). Do not use for headlines.

## 9. Limitations the paper states
- Low-shot segmentation is "still not solved": best F1-max < 60% on both datasets although pAUROC > 95% (supp. Sec. A "Evaluation metrics").
- Zero-shot failure cases (supp. "Failure cases", Fig. 11): (a) logical anomalies (misplaced cable, missing capacitor) need a normal reference; WinCLIP+ alleviates them; (b) tiny defects (carpet, wood, capsule); (c) irrelevant deviations that are not defects (dots on pill, designed holes on PCB2); (d) imprecise boundaries.
- Some defects are only definable against a normal image (e.g. metal nut "flipped upside-down"), motivating WinCLIP+ (Sec. 4.3).
- Future work: vision-language pretraining on industrial data (Sec. 6).
- Later fixes per sources: AnomalyCLIP (2310.18961v12, Related Work) says WinCLIP needs "a large number of hand-crafted text prompts and involves multiple forward passes of image patches"; VAND adds learnable linear projections; AnomalyCLIP replaces the prompts with two learned object-agnostic prompts and segments in "a single forward pass".

## 10. Outline feedback (ch-winclip)
- No equation defines the two-class score directly. Show Eq. (1) with S = {normal, anomalous prototypes} and say ascore_0 = p(anomalous) is our reading of Sec. 4.1.
- The prompts include the category name `[o]` (e.g. "bottle"), taken from the dataset. Zero-shot here means "no images", not "no product knowledge". This is the exact contrast AnomalyCLIP uses ("object" instead of the name).
- CPE numbers for the page: 7 normal states, 4 anomaly states, 22 templates; 154 and 88 prompts are computed.
- viz-winclip window buttons match the paper: 2 x 2 patches (32 px), 3 x 3 patches (48 px), and the image-scale class token. The prediction prompt is answered by main Table 8 and the harmonic-mean rule: a whole-image window gives one score for every pixel, and a confidently normal window wins the harmonic mean.
- The headline 91.8 includes multi-crop (Table 3 row d). Supp. Table 9 prints 90.8 for the same backbone. Use 91.8 with its loc.
- For ch-speed: 389.4 ms per image on an EC2 G4dn instance, batch not stated, GPU model not stated in the paper.
- For ch-dinov2 shots slider: WinCLIP+ has 1/2/4 shots on both datasets (Tables 1, 4) and 8 shots only on MVTec AD (Tables 5, 6). No 16-shot.
- Naming clash: in WinCLIP, "CLIP-AC" = the bare two-class baseline and "+ Prompt ens." = CLIP's ImageNet ensemble. In AnomalyCLIP, "CLIP" = bare and "CLIP-AC" = ImageNet ensemble. Pick one name per baseline on the page.
