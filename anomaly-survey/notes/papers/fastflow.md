# FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows

## 1. Identity
- Title: "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows".
- arXiv 2111.07677, version read: v2 (PDF stamp "arXiv:2111.07677v2 [cs.CV] 16 Nov 2021"; LaTeX src `sources/fastflow/src/fastflow.tex`).
- Dates (abs.html): v1 2021-11-15, v2 2021-11-16.
- Authors: Jiawei Yu (first, equal contribution) ... Liwei Wu (last); 7 authors; SenseTime Research, ICT CAS, UCAS, CUHK-Shenzhen.
- Venue: not stated in abs.html (comment: "11 pages,8 figures"). The LaTeX uses the AAAI-22 template (`aaai22.sty`); that is a template, not an acceptance statement.
- Repo: none given in the paper or abs page. Do not use third-party reimplementations as a source.
- PDF: 11 pages; main text p1-6, supplementary p7-11.

## 2. Core problem
Detect and localize anomalies with normal-only training. Represent the normal feature distribution with a learnable, exact density (a normalizing flow) that works on the 2D feature map directly, so that one forward pass gives a per-location likelihood map.

## 3. Prior methods and why that hurt (per the paper, Sec. 1, 2.3)
- Gaussian / kNN estimators (PaDiM, CutPaste, SPADE, PatchCore): non-parametric or Gaussian modelling; "can not effectively map image features to a tractable base distribution" (abstract).
- 1D flows (DifferNet, CFLOW-AD): flatten the 2D feature into a 1D vector, which "destroys the inherent spatial positional relationship"; DifferNet "failed to obtain the exact anomaly localization results"; CFLOW-AD uses hard-coded position embeddings and slices the image (sliding window), so inference is slow.
- Multi-scale CNN features need hand-designed fusion and do not use global-local relations; sliding windows cost many forward passes.

## 4. The one key idea
Make the flow 2D: the coupling-layer subnets are fully convolutional (3x3 and 1x1 convs alternating), so the flow maps a C x H x W feature map to a C x H x W latent of the same shape, trained so that normal features become standard normal. The per-position likelihood is the anomaly map, in one end-to-end pass.

## 5. Mechanism, step by step
1. Frozen backbone, ImageNet-pretrained: "All these backbone are initialized with the ImageNet pre-trained weights and their parameters are frozen in the following training process" (Sec. 4.7 Implementation Details).
2. Feature choice (Table 6): ViT backbones use the feature of ONE block (CaiT-M48-distilled block 40, DeiT-base-distilled block 7). ResNet18 / WRN50 use "the features of the last layer in the first three blocks", each fed to its own FastFlow model (Sec. 3.2).
3. 2D flow f: X -> Z, bijective (Sec. 3.3). Built as K transformation blocks, each of multiple steps; each step is an affine coupling (Eq. 3) whose subnets s(.) and b(.) are 2D conv nets. Steps alternate a "3x3 flow" and a "1x1 flow" (Fig. 2a).
4. One flow step, per Fig. 2(b) (figure reading, not in the text): ActNorm -> Channel Permute -> Split into two C/2 halves -> subnet (Conv2d -> ReLU -> Conv2d) on half 1 -> split output into b1, s1 -> half 2 := exp(s1) * half 2 + b1 -> a second subnet on the new half 2 -> s2, b2 -> half 1 := exp(s2) * half 1 + b2 -> concat -> C channels. The "Conv 2d" is 3x3 or 1x1 depending on the step type. Eq. 3 in the text shows only one coupling and no exp.
5. Number of steps (Sec. 4.7): "we use 20-step flows in CaiT and DeiT and 8-step flows for ResNet18 and Wide-ResNet50-2".
6. Subnet kernel choice (Sec. 4.4, Table 5): alternating 3x3/1x1 ("3-1") for CaiT, DeiT, WRN50; all 3x3 ("3-3") for ResNet18.
7. Hidden width of the subnet (Supp. Sec. 6.1, Table 7): channel ratio 0.16x for CaiT and 1.0x for WRN50 ("ratio of the number of channels in the hidden layer to the number of channels in the input and output layers for subnet"). DeiT and ResNet18 ratios: not stated.
8. Training: only the flow parameters theta are trained, by maximizing the log-likelihood of normal features (Sec. 1, 3.1). The paper does not print a loss equation (see Sec. 6 below).
9. Test-time map (Sec. 3.3): "we sum the two-dimensional probabilities of each channel to get the final probability map and upsample it to the input image resolution using bilinear interpolation." For ResNets, the three per-scale results are averaged: "finally take the average value as the final result" (Sec. 4.7).
10. Image-level score: NOT stated in the paper.

### Which weights get a gradient
- Backbone: frozen, no gradient (Sec. 4.7).
- Flow subnets (and ActNorm per Fig. 2b): trained by the negative log-likelihood of normal features (our reading of "maximizes the log-likelihood", Sec. 1).

### Tensor walkthrough
Input size, block index and feature size are from Table 6. Channel widths are NOT stated in the paper; values below are the standard architecture widths (our reading).

| Backbone | Input | Feature layer(s) | Feature map C x H x W | Flow | Output map |
|---|---|---|---|---|---|
| CaiT-M48-distilled | 448 | block 40 | 768 x 28 x 28 (C our reading; 28 from Table 6) | 20 steps, 3-1, ratio 0.16x | 28 x 28 -> bilinear to 448 x 448 |
| DeiT-base-distilled | 384 | block 7 | 768 x 24 x 24 (C our reading; 24 from Table 6) | 20 steps, 3-1, ratio not stated | 24 x 24 -> 384 x 384 |
| ResNet18 | 256 | blocks 1,2,3 | 64 x 64 x 64, 128 x 32 x 32, 256 x 16 x 16 (C our reading) | 3 separate flows, 8 steps each, 3-3 | three maps -> upsample -> average |
| WRN50-2 | 256 | blocks 1,2,3 | 256 x 64 x 64, 512 x 32 x 32, 1024 x 16 x 16 (C our reading) | 3 separate flows, 8 steps each, 3-1, ratio 1.0x | three maps -> upsample -> average |

Every flow step keeps the shape C x H x W (bijection). Inside a step, the subnet maps C/2 x H x W -> hidden x H x W -> C x H x W (split into s and b, C/2 each).

Parameter arithmetic check (our reading, counting conv weights + biases only, not in the paper): with 8 steps, alternating 3x3/1x1, ONE Conv-ReLU-Conv subnet per step and hidden = ratio x C/2, WRN50 gives 41.31M (ratio 1.0), 82.60M (2.0) and 20.66M (0.5), matching Table 7's 41.3 / 82.6 / 20.7. The same rule does not reproduce WRN50 0.25x (10.34M vs printed 10.9), ResNet18 (4.65M vs 4.9; 2.59M vs 2.7) or CaiT (7.04M vs 14.8). The two-couplings-per-step reading of Fig. 2(b) gives CaiT 14.09M but doubles WRN50 to 82.6M. Conclusion: quote the printed parameter counts; do not derive them on the page.

## 6. Equations
No custom macros needed. `\mathbf{det}` is standard.

Eq. (1), change of variables (Sec. 3.3, p4):
```tex
{p}_{X}(x) ={p}_{Z}(z) \left\vert \mathbf{det}(\frac{\partial z}{\partial x})\right\vert
```
Paper: "the change of the variable formula defines the model distribution on X". Gloss: x = a feature vector from the backbone; z = f(x) its image under the flow; p_Z = the base density (standard normal); the |det| of the Jacobian dz/dx corrects for how much the flow stretches or squeezes volume around x.

Eq. (2), log-likelihood (Sec. 3.3, p4):
```tex
\log {p}_{X}(x) &= \log p_{Z}(z) + \log \left\vert \mathbf{det}(\frac{\partial z}{\partial x}) \right\vert \\
& = \log p_{Z}(f_{\theta}(x)) + \log \left\vert \mathbf{det}(\frac{\partial f_{\theta}(x)}{\partial x}) \right\vert ,
```
followed by "where $z \sim \mathcal{N}(o, I)$" (printed as `o`, copy verbatim). Paper: "We can estimate the log likelihoods for image features from p_Z(z)". Gloss: theta = flow parameters; f_theta = the 2D flow; first term = how close the warped feature is to the centre of the standard normal; second term = log volume change of the warp.

Eq. (3), affine coupling (Sec. 3.3, p4, numbered (3) in the PDF):
```tex
y_{a}, y_{b} &= \text{split}(y) \\
y'_{a} &= y_{a} \\
y'_{b} &= s(y_{a}) \odot y_{b} + b(y_{a}) \\
y' &= \text{concat}(y'_{a}, y'_{b}),
```
Paper: "where $s(y_a)$ and $b(y_a)$ are outputs of two neural networks. The split(.) and concat(.) functions perform splitting and concatenation operations along the channel dimension." Gloss: y = input to the step (C x H x W); y_a, y_b = the two channel halves; y_a passes through unchanged; y_b is scaled by s(y_a) and shifted by b(y_a), both computed by a conv subnet from y_a. Because y_a is untouched, the step is trivially invertible and its Jacobian is triangular (log-det = sum of log|s|). Note: Fig. 2(b) applies exp to s before multiplying (exp(s1), exp(s2)); Eq. 3 does not.

Training loss: NOT printed in the paper. The only loss equation in the LaTeX is commented out (fastflow.tex lines 355-361), so it is not in the PDF. The text says the flow is trained "through a trainable process that maximizes the log-likelihood of normal image features" (Sec. 1, said of flows in general) and that FastFlow uses a "two-dimensional loss function" (contribution 1). Our reading: loss = - log p_X(x) from Eq. 2, applied at every spatial position of the feature map and averaged over positions and training images.

Composition of blocks (Sec. 3.3, unnumbered):
```tex
X \xrightarrow{f_{1}} H_{1} \xrightarrow{f_{2}} H_{2} \xrightarrow{f_{3}} \cdots \xrightarrow{f_{K}} Z,
```
with "$f_{2d}=f_{1}\circ f_{2}\circ f_{3}\circ \cdots \circ f_{K}$ with $K$ transformation blocks".

## 7. Config / setup table (method lens)
| Item | Value | Location |
|---|---|---|
| Backbones | CaiT-M48-distilled, DeiT-base-distilled, ResNet18, Wide-ResNet50-2 | Table 6 (p7) |
| Input size | 448 / 384 / 256 / 256 | Table 6 |
| Feature size (H = W) | 28 / 24 / [64, 32, 16] / [64, 32, 16] | Table 6 |
| Block index | 40 / 7 / [1,2,3] / [1,2,3] | Table 6 |
| Backbone | ImageNet-pretrained, frozen | Sec. 4.7 |
| Flow steps | 20 (CaiT, DeiT), 8 (ResNet18, WRN50) | Sec. 4.7 |
| Subnet kernels | 3-1 alternating (CaiT, DeiT, WRN50); 3-3 only (ResNet18) | Sec. 4.4, Table 5 |
| Subnet hidden ratio | 0.16x (CaiT), 1.0x (WRN50); others not stated | Supp. Sec. 6.1, Table 7 |
| Flow step parts | ActNorm, channel permute, two affine couplings with exp(s) | Fig. 2(b) only |
| Optimizer | Adam, lr 1e-3, weight decay 1e-5 | Sec. 4.7 |
| Schedule | 500 epochs | Sec. 4.7 |
| Batch | 32 | Sec. 4.7 |
| Train augmentation | random hflip 0.5, vflip 0.3, rotation 0.7 (probabilities); not for categories like transistor | Supp. Sec. 6.2 |
| Anomaly map | sum of per-channel probabilities, bilinear upsample; ResNet: average over 3 scales | Sec. 3.3, 4.6 |
| Image score | not stated | -- |
| Timing hardware | Intel Xeon E5-2680 V4 @ 2.4GHz + NVIDIA GeForce GTX 1080Ti | Sec. 4.2 |
| Timing batch size | not stated | -- |
| Protocol | one model per category, normal-only training (MVTec AD) | Sec. 4.1 |

## 8. Results worth remembering
Protocol for all MVTec AD rows: one-class (one model per category), full-shot, normal-only training; resolution per backbone (Table 6). Metrics: image AUROC and pixel AUROC only. No PRO, no VisA.

Table 1 (p5), complexity + accuracy per backbone. Columns: FPS, A.d. Time (ms), A.d. Params (M), image AUC, pixel AUC. "A.d." = additional, i.e. excluding the backbone. Hardware: Xeon E5-2680 V4 + GTX 1080Ti, batch not stated.
| Row | FPS | A.d. ms | A.d. M | Img | Pix |
|---|---|---|---|---|---|
| CaiT-M48-distilled + FastFlow | 3.08 | 9 | 14.8 | 99.4 | 98.5 |
| DeiT-base-distilled + FastFlow | 30.14 | 8 | 14.8 | 98.7 | 98.1 |
| ResNet18 + FastFlow | 30.8 | 27 | 4.9 | 97.9 | 97.2 |
| WRN50-2 + FastFlow | 21.8 | 34 | 41.3 | 99.3 | 98.1 |
| CaiT + Patch Core | 2.39 | 107 | 0 | 97.9 | 96.5 |
| CaiT + CFlow | 2.76 | 42 | 10.5 | 97.7 | 96.2 |
| DeiT + Patch Core | 15.45 | 39 | 0 | 96.5 | 97.9 |
| DeiT + CFlow | 16.91 | 34 | 10.5 | 95.6 | 97.9 |
| ResNet18 + SPADE | 3.92 | 250 | 0 | - | - |
| ResNet18 + CFlow | 20.3 | 44 | 5.5 | 96.8 | 98.1 |
| WRN50 + SPADE | 0.67 | 1481 | 0 | 96.2 | 96.5 |
| WRN50 + Patch Core | 5.88 | 159 | 0 | 99.1 | 98.1 |
| WRN50 + CFlow | 14.9 | 56 | 81.6 | 98.3 | 98.6 |
Our reading: the ViT-backbone baseline rows are FastFlow's own runs (Sec. 1: "we found that performing this simple replacement in other methods actually degrade the performance"). Whether the CNN baseline rows are re-runs or copied is not stated.

Text claims (Sec. 4.2): "up to $10\times$ faster than other methods"; vs CFlow "$1.5\times$ speedup and $2\times$ parameter reduction"; additional inference time "up to $4\times$ reduction compared to Cflow and $10\times$ reduction compared to Patch Core".

Table 2 (p6), MVTec AD mean (image AUC, pixel AUC), row "AUCROC":
- FastFlow (99.4, 98.5). Table 2 does not name the backbone; the numbers equal the CaiT row of Table 1 (our reading: CaiT-M48-distilled, 448 px).
- Re-reported baselines in the same row: PatchSVDD (92.1, 95.7); SPADE* (96.2, 96.5); DifferNet (94.9, -); PaDiM (97.9, 97.5); Cut Paste (97.1, 96.0); PatchCore (99.1, 98.1); CFlow (98.3, 98.6). The paper does not say these were re-run.
- CFlow's pixel AUC 98.6 is bold and beats FastFlow's 98.5 in FastFlow's own table. FastFlow wins image AUC only.
- The Cut Paste entry 97.1 is CutPaste's EfficientNet-B4 level-7 fine-tuned number (CutPaste Table 3), not its ResNet-18 from-scratch headline 95.2 (CutPaste Table 1). The 96.0 is CutPaste's patch-level ResNet-18 pixel AUC (CutPaste Table 2).
- The PaDiM entry 97.9 differs from the PaDiM image AUC 95.5 that DRAEM prints (DRAEM Table 1). Do not reconcile; cite per paper.
- Abstract: "Our approach achieves 99.4\% AUC in anomaly detection".

Table 5 (p6), subnet ablation (A.d. Params M, image AUC, pixel AUC):
- DeiT 3-1: 14.8, 98.7, 98.1; DeiT 3-3: 26.6, 98.7, 98.3.
- CaiT 3-1: 14.8, 99.4, 98.5; CaiT 3-3: 26.6, 98.9, 98.5.
- ResNet18 3-1: 2.7, 97.3, 96.8; ResNet18 3-3: 4.9, 97.9, 97.2.
- WRN50 3-1: 41.3, 99.3, 98.1; WRN50 3-3: 74.4, 98.2, 97.6.

Table 7 (p8), hidden channel ratio (Params M, image, pixel): CaiT 0.16x 14.8 / 99.4 / 98.5; CaiT 0.33x 29.6 / 98.9 / 98.4; WRN50 0.25x 10.9 / 98.9 / 98.0; 0.5x 20.7 / 99.1 / 98.1; 1.0x 41.3 / 99.3 / 98.1; 2.0x 82.6 / 99.4 / 98.1.

Table 8 (p8), training augmentation: CaiT w/o 99.3 / 98.4, w 99.4 / 98.5; WRN50 w/o 98.9 / 98.2, w 99.3 / 98.1. So the headline 99.4 uses training augmentation.

Other datasets:
- BTAD (Table 3, p6): FastFlow mean pixel AUC 0.97 (printed as fraction); text says "97.0 pixel-wise AUC".
- CIFAR-10 (Table 4, p6): FastFlow AUC 66.7.
- Non-aligned MVTec AD (Supp. Sec. 8, p8): test set perturbed (zoom 0.85, rotation +-15 deg, translation 0.15, expanded 4x); FastFlow (CaiT) 99.2 image AUC, 98.1 pixel AUC.

## 9. Limitations the paper states
- Misses "a few small and unobvious anomalies"; false detections in background areas with hair and dirt; label ambiguity (Supp. Sec. 7, Fig. 5-7).
- Small-capacity backbones prefer 3-3 subnets; the 3-1 design is a speed/accuracy trade-off (Sec. 4.4).
- Data augmentation must be chosen per category ("the transistor can not be flipped upside down and rotated", Supp. Sec. 6.2).
- No later fix is named in this source.

## 10. Outline feedback (ch-fastflow)
- "Equations: ... the training loss": the paper prints no loss equation (it is commented out in the LaTeX). Present the loss as -log p_X(x) from Eq. 2, labelled "our reading", or quote Eq. 2 and say "trained to maximize this".
- The coupling in Eq. 3 has no exp; Fig. 2(b) uses exp(s) and two couplings per step, plus ActNorm and channel permute. If the toy shows a coupling layer, say which form it follows.
- Architecture walkthrough "frozen backbone (which ones the paper tries)": CaiT-M48-distilled (448 px, block 40, 28 x 28), DeiT-base-distilled (384, block 7, 24 x 24), ResNet18 and WRN50-2 (256; blocks 1-3; 64/32/16). Channel widths are our reading. ResNets use THREE separate flows, one per scale, averaged at the end.
- Steps: 20 for ViTs, 8 for ResNets. Do not compute parameter counts from config; the printed counts do not follow one rule (Sec. 5 above). Quote Table 1/5/7.
- The headline 99.4 / 98.5 is a CaiT number at 448 px, not a CNN. With WRN50 at 256 px it is 99.3 / 98.1. The scoreboard row must name the backbone.
- FastFlow does not lead pixel AUROC in its own Table 2 (CFlow 98.6 vs 98.5).
- Image-level score rule is not stated. Do not claim "max of the map" for FastFlow.
- ch-speed: Table 1 times are on a GTX 1080Ti with a Xeon E5-2680 V4; batch size is not stated; "A.d. Time" excludes the backbone, while FPS includes it (CaiT FastFlow is 3.08 FPS despite 9 ms additional). Label both.
- Fig. 4 (p7) shows the inverse flow generating a feature map from a perturbed latent; this could feed the "invertible" point in the viz if wanted.
- No VisA, no PRO for the closing scoreboard.
