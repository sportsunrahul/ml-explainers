# Finding defects you have never seen: a survey of industrial visual anomaly detection

Reader: knows basic deep learning (a network, a loss, gradient descent, what a CNN feature map is) and
knows DINO and CLIP from earlier explainers. Everything else (AUROC, kNN, Mahalanobis distance,
normalizing flows, knowledge distillation, prompt learning) is introduced where it is first needed.
No narration: a long page plus an atlas page. Cue ids and the player hooks stay, so narration can be
added later with `resume anomaly-survey scripts`. Wording is generic: no company or product names.

Core question, asked in every chapter: how do you find defects when you have almost no defect images?

Scan date for all field counts: 2026-10-09 (`notes/arxiv-scan-2026-10-09.json`, 1,165 unique records,
780 in the industrial visual subset; the rule is in `state.json`).

Color key (one meaning per color, whole page):
blue = normal (training data, normal features, the model of normal); red = anomaly (defects, anomaly
scores, distances that flag); yellow = the test sample under inspection; green = a learned component
(trained weights, learned prompts, the student); brown = frozen or stored (pretrained backbone,
memory bank, frozen CLIP).

3D figures (pe3.js, orbitable, reduced-motion safe): ch-padim Gaussian bell, ch-patchcore embedding cloud
with coreset, ch-fastflow density warp, and the hero. Everything else is canvas 2D or SVG. Each 3D
figure says whether it is a toy or computed from data.

Scoreboard policy: each paper's own number with its table and row. A reproduction view for
one-model-per-class methods comes from Anomalib 2202.08341 only if its paper tabulates them. ADer
2406.03262 is multi-class, so it appears only in ch-uniad.

Hero (canvas, toy): a stream of parts passes under a camera. Normal parts drop their patch features
into a blue cloud. A part with a scratch arrives; one patch lands far from the cloud, a red distance
line grows, and the matching spot on the part lights up as a heat map. Ends on a static diagram:
"normal cloud, test patch, distance = anomaly score". Replay button `btn-hero-replay`.

---

## Act I. The problem and the yardstick

### 1. ch-problem: Learn normal, measure surprise
Intuition to leave: you cannot train a classifier on defects you do not have, so model what normal
looks like and score how far a new sample is from it. The whole field is variations on "what is the
model of normal" and "how is distance measured".
From scratch: one-class setting (train on good parts only); anomaly score; threshold; false alarm vs
missed defect. The four training regimes as a small table: normal images only (most of this page);
normal plus synthetic defects (ch-synthetic); no images of this product at all (zero-shot, ch-winclip);
a handful of normal images (few-shot, ch-dinov2).
Visual `viz-normal-cloud` (2D canvas, toy): a cloud of normal patch features. Drag the yellow test point;
the readout shows the distance to the nearest normal point. A threshold slider draws the decision ring;
buttons "few normals" / "many normals" show that sparse coverage causes false alarms. Prediction prompt:
"with only 10 normal samples, where will a perfectly normal part get flagged?"
No equations beyond the score s(x) = min distance to the normal set.

### 2. ch-benchmark: MVTec AD, VisA, and how a detector is graded
Intuition to leave: image AUROC asks "does every defective part score above every good one?"; pixel
AUROC is swamped by the ocean of normal pixels, so it stays high even when small defects are missed;
PRO gives every defect region the same weight. Read every score on this page with that in mind.
Claim card: MVTec AD (CVPR 2019): categories (textures vs objects), train/test counts, defect types,
pixel-precise masks; what it measures and what would falsify "it is a fair benchmark" (saturation).
VisA (2207.14315): subsets, image counts, multiple instances and complex structures.
Visual `viz-metrics` (2D canvas, toy): a part with one large and one tiny defect, its score map, and live
ROC curves at image and pixel level plus a PRO curve. A threshold slider sweeps; a "miss the tiny defect"
button lowers its score: pixel AUROC barely moves, PRO drops. Prediction prompt: "how much will pixel
AUROC fall if we miss the tiny scratch entirely?"
Equations: AUROC as P(score of a random anomaly > score of a random normal); PRO as the mean per-region
overlap, integrated to FPR 0.3 (check the cut-off in the sources).
Step "after saturation": MVTec LOCO (logical anomalies: a missing or misplaced component, wrong count)
and MVTec AD 2 (2503.21622: harder scenes; how far scores drop). LOCO facts come from EfficientAD and
MVTec AD 2 until the IJCV PDF is in `sources/mvtec-loco/`.
Extract: MVTec AD counts and categories; VisA counts and subsets; the PRO definition and FPR limit;
MVTec AD 2 headline numbers; LOCO's five categories and its metric (sPRO).

### 3. ch-map: Six families on one map
Intuition to leave: the field split into families that answer "what is the model of normal" differently;
they did not replace each other, they coexist, and the growth since 2023 is in multi-class and
vision-language methods.
Visual `viz-map` (SVG, computed from the scan): papers per year in the industrial visual subset, with a
toggle per keyword family (heuristic, overlapping). Below it, the six family lanes with the spine papers
on a time axis. Every count is a `data-claim` whose `loc` is the query plus the subset rule.

---

## Act II. Remember normal

### 4. ch-spade: Pretrained features already know what normal looks like
Intuition to leave: a network trained on ImageNet maps a scratch and a clean surface to different
feature vectors, without ever seeing a defect. Nearest-neighbor distance in that feature space is a
strong anomaly score, first for whole images (DN2), then for every pixel (SPADE).
Mechanism: image-level kNN to find the K most similar normal images; pixel-level kNN against the
feature pyramids of those K images only.
Visual `viz-spade` (2D canvas, toy): a textured part with a defect. Click a pixel: its feature appears in
a 2D projection next to the pixel features of the K retrieved normal images; the readout gives the
mean distance to the k nearest. Toggle "image-level only" vs "pixel-level".
Equations: SPADE's image score (mean distance to K nearest) and pixel score (Eq. numbers from v3).
Architecture walkthrough: backbone (Wide-ResNet-50-2 per the paper; confirm) forward with feature-map
shapes per stage; backward: nothing is trained. "Which boxes get a gradient?" Answer: none; fitting
is storing features.
Branches: DN2 2002.10445 (image-level kNN), Patch SVDD 2006.16067, AE-SSIM 1807.02011 (the
reconstruction baseline MVTec AD shipped with).

### 5. ch-padim: One Gaussian per patch position
Intuition to leave: for an aligned part, the patch at position (i, j) always looks roughly the same, so
fit one Gaussian per position and score with Mahalanobis distance, which respects the shape of the
normal spread instead of treating every direction as equal.
Visual `viz-padim` (3D, pe3, toy with config-shaped dimensions): the normal features at one position
drawn as a 2D slice with a Gaussian bell rising above it. Drag the yellow test patch: its height on
the bell and its Mahalanobis distance update; a toggle "Euclidean" vs "Mahalanobis" shows a point that
is close in Euclidean terms but far along the thin axis of the ellipse. A position picker shows that
another position has a differently shaped bell.
Equations: per-position mean and regularized covariance (the epsilon I term); Mahalanobis distance.
Random dimensionality reduction (how many of how many channels) with the paper's ablation.
Architecture walkthrough: the three backbone layers concatenated, the reduced dimension, storage per
position (computed from the config); backward: none.
Branch: Gaussian AD 2005.14140 (one Gaussian per image, the predecessor).

### 6. ch-patchcore: A memory bank, thinned to a coreset
Intuition to leave: keep every normal patch feature instead of a Gaussian, so no shape is assumed; then
keep only a small subset that still covers the whole cloud (greedy coreset), so search stays fast.
Visual `viz-patchcore` (3D, pe3, toy): a cloud of normal patch embeddings. A slider sets the coreset
percentage; "run greedy selection" adds points one by one, each the farthest from the chosen set, and
draws the current coverage radius. A yellow test patch shows its nearest neighbor in the full bank vs
the coreset. Prediction prompt: "how few points can we keep before the coverage radius jumps?"
Equations: locally aware patch features (neighborhood aggregation); the coreset objective (minimax
facility location) and the greedy approximation; the patch score and the image score with the
reweighting term (Eq. numbers from v2).
Architecture walkthrough: input resolution, layer 2 and 3 feature shapes, aggregation, projection to
the target dimension, memory bank size per category (computed); backward: none.
Branches: CFA 2206.04325, PNI 2211.12634, ReConPatch 2305.16713.

Interlude `ch-patchcore-pipeline` (not narrated): PatchCore, end to end. A stepper through fit and test:
images -> backbone -> layer 2 + 3 -> local aggregation -> upsample and concatenate -> memory bank ->
random projection -> greedy coreset -> test image -> nearest neighbor per patch -> reweighting ->
upsample and Gaussian smoothing -> anomaly map and image score. Shapes in a readout, every config value
a `data-claim`. This takes the place of the "one training step" interlude: PatchCore has no training.

---

## Act III. Model the density, or fake the defect

### 7. ch-fastflow: Turn normal into a perfect Gaussian
Intuition to leave: a normalizing flow is an invertible network that warps the distribution of normal
features into a standard Gaussian; then the exact likelihood of any test feature is computable, and
low likelihood means anomaly. FastFlow does this in 2D, keeping the spatial layout.
Visual `viz-fastflow` (3D, pe3, toy): a twisted density surface over a 2D feature space; a scrubber
applies coupling layers one by one until the surface is a round bell; a yellow test point rides along
and ends up in the tail. Prediction prompt: "after the warp, where will the scratched patch land?"
Equations: change of variables (log-likelihood = Gaussian log-density + log-determinant); the affine
coupling layer; the training loss.
Architecture walkthrough: frozen backbone (which ones the paper tries), the 2D flow steps with their
3x3 / 1x1 subnets, output shapes; backward: the gradient reaches only the flow.
Branches: DifferNet 2008.12577, CFLOW-AD 2107.12571, CS-Flow 2110.02855, MSFlow 2308.15300.

### 8. ch-synthetic: Make fake defects (CutPaste, DRAEM)
Intuition to leave: if you have no defects, manufacture plausible ones and turn the problem into a
supervised one. CutPaste cuts and pastes patches and learns "normal vs pasted"; DRAEM paints
Perlin-noise blobs of foreign texture and learns to segment them. The risk: the model learns your fakes,
not real defects.
Visual `viz-synth` (2D canvas, toy, the Perlin noise is computed): a normal part; tabs "CutPaste" /
"DRAEM". CutPaste: drag the source and target rectangle. DRAEM: a Perlin threshold slider and an opacity
(beta) slider generate the mask and the blended image; "new noise" resamples (cue-exempt).
Equations: DRAEM's losses (reconstruction L2 + SSIM; focal loss on the mask); CutPaste's training
objective and its density-estimation scoring (Gaussian density estimator).
Architecture walkthrough (DRAEM): reconstructive subnet and discriminative subnet, input/output shapes;
backward: both subnets trained end to end, from which losses.
Branches: NSA 2109.15222, MemSeg 2205.00908, DSR 2208.01521, SimpleNet 2303.15140, GLASS 2407.09359.

---

## Act IV. Fast, and one model for all

### 9. ch-efficientad: A student that fails only on defects
Intuition to leave: train a small student network to imitate a teacher's features on normal images;
on a defect the student was never shown how to imitate, so the gap between them is the anomaly map.
An autoencoder adds a second gap for logical anomalies (a missing part), which local features miss.
The architecture is tiny (a patch description network), so it runs in a few milliseconds.
Visual `viz-efficientad` (2D canvas, toy): three panels: teacher map, student map, difference. Buttons
"structural defect" (a scratch) and "logical defect" (a missing component): the student-teacher gap
catches the first, the autoencoder gap catches the second. A slider sets the hard-feature quantile.
Equations: the hard feature loss (only the top quantile of distances is trained); the ImageNet penalty;
the autoencoder and student-vs-autoencoder losses; map normalization by quantiles.
Architecture walkthrough: the PDN layers with shapes for EfficientAD-S and -M; teacher distillation from
the pretrained backbone (which); backward per training stage.
Hook: EfficientAD's own latency/throughput table on its GPU (sourced); the full comparison is ch-speed.
Branches: Uninformed Students 1911.02357, STPM 2103.04257, Reverse Distillation 2201.10703.

### 10. ch-uniad: One model for every product, and the identical shortcut
Intuition to leave: training one model per product does not scale. A single reconstruction model for
all products fails in a specific way: it learns to copy its input, so defects get reconstructed too
("identical shortcut"). UniAD stops copying by masking each token's own neighborhood in attention and
by adding noise to the features it must reconstruct.
Visual `viz-uniad` (2D canvas, toy): a row of feature tokens with a defect token. "Full attention" lets
each output copy its own input: the defect reconstructs perfectly and the error map is blank. "Neighbor
mask" forces each token to be predicted from the others: the defect cannot be copied and lights up.
Equations: the reconstruction loss in feature space; the neighbor-masked attention mask; feature jitter.
Architecture walkthrough: frozen backbone (which), token count, transformer encoder and layer-wise query
decoder, shapes; backward: the gradient stops at the frozen backbone.
Multi-class comparison: ADer numbers for one-class methods run in the multi-class setting (clearly
labelled as multi-class). Branches: HVQ-Trans 2310.14228, DiAD 2312.06607, MambaAD 2404.06564.

---

## Act V. No training images

### 11. ch-winclip: Ask CLIP whether the part is damaged
Intuition to leave: CLIP already aligns images with words like "damaged" and "flawless", so compare each
image window with text prompts and you get an anomaly map with zero training images. WinCLIP adds a
prompt ensemble and multi-scale windows; WinCLIP+ adds a few normal images as a reference memory.
Visual `viz-winclip` (2D canvas, toy): a part with a defect; window size buttons (small, mid, whole
image); windows slide and each gets a "normal vs anomalous" score bar from text similarity; the
aggregated map builds up. Prediction prompt: "why does a whole-image window miss a tiny defect?"
Equations: the two-class softmax over normal/anomalous text embeddings; the compositional prompt
ensemble (states x templates); harmonic aggregation of window scores.
Architecture walkthrough: CLIP image and text encoders, window tokens, shapes; backward: none (zero-shot).
Branches: APRIL-GAN 2305.17382, AnomalyGPT 2308.15366, MuSc 2401.16753, InCTRL 2403.06495, PromptAD 2404.05231.

### 12. ch-anomalyclip: Prompts that know "damaged", not "which object"
Intuition to leave: hand-written prompts name the object; AnomalyCLIP learns two object-agnostic prompts
("[learned] object" and "[learned] damaged object") once, on one auxiliary dataset, and they transfer to
products it never saw, because "damaged" looks similar across objects.
Visual `viz-anomalyclip` (SVG, sourced): a transfer grid: train on dataset A (rows), test on B (columns),
with the paper's numbers; a toggle between object-aware and object-agnostic prompts (the ablation).
Equations: the learnable prompt templates; the glocal loss (global cross-entropy plus local focal and
Dice on the maps); DPAM attention as a short gloss.
Architecture walkthrough: CLIP frozen, the gradient passes THROUGH the frozen text encoder to the
learnable prompt tokens (and any learnable tokens inside text layers); a clear "frozen but the gradient
passes through" case. Branches: AdaCLIP 2407.15795, UniVAD 2412.03342.

### 13. ch-dinov2: DINOv2 brings nearest neighbors back (AnomalyDINO, Dinomaly)
Intuition to leave: with a strong self-supervised backbone, the simplest ideas win again. AnomalyDINO is
PatchCore-style patch kNN on DINOv2 features with a handful of normal images, and beats CLIP-based
few-shot methods. Dinomaly is a multi-class reconstruction model built from DINOv2 and a few deliberately
simple parts (a noisy bottleneck, loose reconstruction) that closes the gap to one-model-per-class.
Visual `viz-dinov2` (SVG, sourced): shots slider (1, 2, 4, 8, 16 normal images) with AnomalyDINO vs
WinCLIP+ / AnomalyCLIP numbers where both papers report the same setting; a second tab places Dinomaly
on the ch-uniad multi-class axis. Claim card style; no new architecture walkthrough beyond a short
"what is different from PatchCore / UniAD" diagram.

---

## Act VI. Choosing a method

### 14. ch-speed: Speed versus accuracy, on a desktop GPU and on a Jetson
Intuition to leave: latency differs by orders of magnitude between families, the ranking changes with
the device, and a line running N parts per second rules some methods out before accuracy matters.
Visual `viz-speed` (SVG, sourced): latency (log axis) vs image AUROC. A device toggle: desktop GPU
(EfficientAD's table: its GPU and averaging stated) vs Jetson Orin Nano (2605.24251v2 table) vs other
Jetson numbers (EdgeZSAD 2606.16119v1, KairosAD 2505.24334v1). A line-speed slider shades the feasible
region. Every point is labelled with its source and device; the caption says protocols differ.
Branches: PaSTe 2410.11591, Unified AD on Xavier NX 2407.02968, PatchCore-Lite 2603.20288.

### ch-timeline: The family tree
Every spine and branch paper on one tree by family and date; click a node for its one-line idea and its
one key number.

### ch-summary: The scoreboard and what to use when
A sortable scoreboard: image AUROC, pixel AUROC and PRO on MVTec AD and VisA per method, each cell with
its source; then a short "if you have X, start with Y" guide (all normal images and a GPU; a few
images; none; a Jetson at line speed).

### ch-deeper: Where to go deeper
Per paper: the section and equation to read first.

---

## Atlas (atlas page, about 9 stages, no narration)

1. problem (normal cloud) 2. benchmark (metrics, step tabs: image AUROC / pixel AUROC / PRO / after
saturation) 3. memory (step tabs: SPADE / PaDiM / PatchCore, sharing one 3D cloud) 4. flows
5. synthetic defects 6. student-teacher 7. one model for all 8. zero/few-shot (tabs: WinCLIP /
AnomalyCLIP / AnomalyDINO) 9. speed and scoreboard. Each figure is written once as a scene and mounted
in both pages where possible.
