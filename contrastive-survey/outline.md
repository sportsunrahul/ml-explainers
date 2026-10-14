# Outline: Learning by comparing

Working title: "Pull the pair, push the rest: a field guide to contrastive learning".
Reader: basic deep learning only (knows a CNN, softmax, cross-entropy, SGD). Vision only.
Long page `explainer.html` only. No atlas, no narration (checkpoint 0, 2026-10-13). Cue ids and reset buttons
are still built so audio can be added later.
Generic wording: no company names or internals on the page.

Cross-links (do not re-explain; one paragraph and a link):
- `../jepa-survey/explainer.html`: BYOL, SimSiam, VICReg, DINO, the energy-based view of collapse.
- `../vlm-survey/explainer.html`: CLIP inside VLMs, its InfoNCE toy and gradient figure.
- `../ovd-survey/explainer.html`: CLIP inside open-vocabulary detectors.

Color key for the whole page (one meaning each):
- blue = the anchor (one view of an image, or the image side of an image-text pair)
- yellow = the positive (the other view, or the matching caption)
- green = agreement: the pull between a positive pair, alignment, a kept structure
- red = negatives and their push, the loss, the gradient
- brown = no gradient: the momentum encoder, the memory bank or queue, stop-gradient branches

Running example: a small procedural "photo" drawn on canvas (a mug on a table, a few colored shapes). Every
augmentation figure crops, color-jitters, greys and blurs this one image, so the reader learns it once.

## Story

Problem: labels are expensive; can a network learn useful features from unlabeled images?
Key insight: make it a comparison task. Two views of the same image should map close together; views of
different images should not. The loss is a softmax over "which of these N is my partner".
Mechanism: InfoNCE -> how to get many negatives (memory bank, MoCo queue, SimCLR's big batch) -> what the
loss actually optimizes (alignment and uniformity) and what temperature does -> hard and false negatives
-> methods that drop negatives (SwAV, Barlow Twins) and the collapse they all fight -> labels (SupCon)
-> language (CLIP's softmax, SigLIP's sigmoid).
What came next: self-distillation (DINOv2/v3) took over image-only pretraining; contrastive image-text
training (CLIP/SigLIP family) still feeds VLMs and open-vocabulary detectors.

## Hero
`viz-hero`: 12 images as points on a circle (a 2D slice of the embedding sphere), each with a yellow twin
(its second view). Act 1: a green spring pulls each pair together. Act 2: red arrows push each point away
from the others. Act 3: real InfoNCE gradient steps on the 24 unit vectors; pairs meet, groups spread
evenly. Ends static on the diagram "pull the pair, push the rest". Replay `btn-hero-replay`.
Computed (a real loss on 2D unit vectors), drawn as a toy.

## Act I: the idea

### 1. `ch-compare` Learning by comparing
- Intuition: a label tells the network what an image is. A comparison only says "same source or not".
  Pulling pairs together with nothing else collapses every image to one point. A push term (negatives)
  or another trick (Act IV) prevents that.
- Visual `viz-pullpush` (computed): 8 images x 2 views as 2D points. Buttons `btn-pullpush-pull` (pull only:
  all points slide to one spot, readout "spread 0.00") and `btn-pullpush-both` (pull + push: pairs meet,
  groups spread). Real gradient descent on the two toy losses; run in node first (`notes/sim-pullpush.txt`).
- Prediction prompt: "what happens if we only pull?"
- Equation: the margin contrastive loss of Hadsell, Chopra, LeCun 2006 (DrLIM, CVPR; Eq. 4), with the
  margin m drawn as a ring in the figure. Mention Chopra 2005 (siamese) and the triplet loss
  (FaceNet 1503.03832) as the same idea with a margin.
- Bridge to ch-infonce: a margin needs a hand-picked distance; a softmax over many candidates does not.
- Aside "In practice": where comparison-based features show up today (retrieval, few-shot, zero-shot).

### 2. `ch-map` The field at a glance
- Intuition: the word "contrastive" exploded in 2020 (MoCo, SimCLR) and spread far beyond vision; the
  image slice is a small, steady share. Image-only contrastive pretraining peaked around 2020-2021; the
  image-text form (CLIP) is what lives on.
- Visual `viz-map` (sourced from the scan): per-year bars 2017-2026 for title "contrastive" vs abstract
  "contrastive learning"; a domain toggle (image/visual, text, graph, video/audio, medical, recommendation,
  anomaly). Every count is a data-claim with loc = query URL; labelled heuristic; 2026 partial.
- Family tree `viz-tree`: spine + branches on a time axis, lanes: negatives (Wu, CPC, CMC, MoCo, SimCLR),
  theory (Arora, Poole, Tschannen, Wang & Isola, Wang & Liu, HaoChen, Jing), hard negatives (Chuang,
  Robinson, DCL), no negatives (SwAV, BYOL, SimSiam, Barlow, VICReg, DINO, iBOT, DINOv2, DINOv3, Web-SSL),
  labels (Chopra, DrLIM, FaceNet, N-pair, SupCon), language (CLIP, ALIGN, OpenCLIP scaling, SigLIP, MetaCLIP,
  SigLIP 2, Perception Encoder), dense (DenseCL, DetCon). Edges "builds on".
- Claim card for the scan: what it measured, what it cannot see.

### 3. `ch-infonce` InfoNCE: find your partner in a lineup
- Intuition: the loss is ordinary softmax cross-entropy where the "classes" are the N candidates in the
  batch and the right answer is the partner. Temperature tau sets how peaked the softmax is.
  CPC (1807.03748) derived it by predicting the future in latent space, and showed that minimizing it
  raises a lower bound on mutual information: I >= log N - L. The bound can never exceed log N, and
  Tschannen et al. (1907.13625) show that tighter MI bounds can give worse features: the success depends
  on the encoder and the critic, not on MI alone (hedged as the paper hedges it).
- Visual `viz-lineup` (computed): an anchor and N candidate cards with similarity bars, their softmax
  probabilities and the loss. Sliders for N (2..64) and tau (0.05..1). A line marks log N, the ceiling of
  the MI bound. Buttons `btn-lineup-easy` / `btn-lineup-hard` set the negatives far or close.
- Equation build: CPC Eq. 4 (InfoNCE) and the bound (unnumbered display after Eq. 5, Sec. 2.3; derivation
  App. A.1 uses an approximation at Eq. 8; Poole Sec. 2.3 Eq. 10 proves it exactly); then our page notation
  with sim(u,v)/tau. CPC v1 has no temperature: say tau is SimCLR's addition.
- Architecture walkthrough "Inside CPC: forward, then backward" (ArchFlow): the image version (crop ->
  grid of overlapping patches -> encoder -> context network -> predictions k rows ahead -> InfoNCE). Shapes
  from the CPC image setup: 256x256 crop of a 300x300 image, 7x7 grid of 64x64 patches with 32 px overlap,
  60x60 random subcrop per patch, ResNet-v2-101 -> 7x7x1024, up to 5 rows ahead, 32 GPUs x batch 16.
  The paper does not state the image negatives: label our choice (other patches in the minibatch) as our
  reading. Gradient reaches encoder and context network; no frozen parts.
- Claim card: Tschannen (MNIST top/bottom halves; fixed-bound encoder test, invertible-encoder test; bound
  tightness vs linear accuracy; what would falsify).
- Warn: in Arora 1902.09229 tau is a class-collision probability, not the temperature.
- Branches named here: Poole 1905.06922 (variational MI bounds), CPC v2 1905.09272 (data-efficient
  recognition), CMC 1906.05849 (views = channels), Arora 1902.09229 (first theory), NCE (Gutmann &
  Hyvarinen 2010) as the ancestor of the name.

## Act II: the recipes

### 4. `ch-moco` Negatives at scale: a memory bank, then a momentum queue
- Intuition: more negatives make the lineup harder and the features better, but a batch holds only so many.
  Wu et al. 2018 (1805.01978) stored every image's embedding in a memory bank, updated once per visit, so the
  stored keys are stale. MoCo (1911.05722) keeps a queue of recent keys produced by a slowly moving copy of
  the encoder (momentum m, e.g. 0.999), so keys stay consistent while the queue stays large (65536).
- Visual `viz-queue` (toy, animated): a query encoder (trainable), a key encoder (brown, momentum copy),
  a FIFO queue sliding as each batch enqueues keys and dequeues the oldest. Buttons
  `btn-queue-step` (one training step), momentum chips m = 0, 0.9, 0.99, 0.999, 0.9999 that show the
  sourced MoCo momentum ablation (run at K = 4096, not 65536; m = 0 "fails", no number).
- Sourced chart `viz-mechs`: accuracy vs number of negatives K for end-to-end, memory bank and MoCo
  (MoCo Fig. 3 data labels, from the figure PDF). End-to-end stops at K = 1024 and uses K-1 negatives.
- Equations: MoCo Eq. 1 (InfoNCE with the queue) and Eq. 2 (momentum update).
- Architecture walkthrough "Inside MoCo": query and key encoders, the queue; backward: the gradient reaches
  only the query encoder; the key encoder and queue are brown (no gradient; updated by momentum).
  Shuffling BN noted as a detail.
- Deltas: MoCo v2 (2003.04297): MLP head + blur augmentation + cosine schedule (+0.2 only), tau 0.2 (v1:
  0.07). MoCo v3 (2104.02057): ViT backbone, keeps the momentum encoder (m = 0.99 for ViT) and adds a
  prediction head, drops the queue at batch 4096, symmetric loss scaled by 2 tau, the instability and its
  fix (frozen random patch projection); ViT-B 76.7 is 600 epochs (300: 76.5).

### 5. `ch-simclr` SimCLR: strong augmentation, a big batch, a throwaway head
- Intuition: no bank, no queue: the other 2(N-1) views in one huge batch are the negatives. Three findings
  matter: the augmentation composition (random crop + color distortion) matters most; a small MLP
  projection head used only for the loss improves the layer before it; bigger batches and longer training
  help contrastive learning more than supervised learning.
- Visual `viz-augment` (toy image + sourced table): the running-example image under each augmentation.
  Chips pick two augmentations; the figure shows the two views and, where the paper reports it, the sourced
  linear accuracy for that pair from SimCLR Fig. 5 (data labels from the figure PDF). Caption the
  asymmetric protocol: the transform goes on one branch only, which lowers accuracy (crop then color 56.3,
  crop alone 33.1). Table 1 (color-distortion strength) as the second sourced table. Prediction prompt: "crop alone or crop + color: which wins, and why?" (answer: with crop
  alone the network can match color histograms).
- Sourced: head ablation only as the text deltas (+3% over linear, >10% over none; Fig. 8 is a plot), batch
  size x epochs from Table B.1. Do not state a default tau (never given; Table 5 best 0.1) or the head's
  hidden width (not given).
- Equation: NT-Xent, SimCLR Eq. 1, with the 2N x 2N similarity matrix drawn: the diagonal masked, the
  positive per row marked.
- Architecture walkthrough "Inside SimCLR": x -> two augmentations -> ResNet-50 -> h (2048) -> MLP g ->
  z (128) -> NT-Xent; batch 4096 -> 8192 views (8190 negatives per anchor: our arithmetic); LARS lr 4.8,
  weight decay 1e-6, epochs, TPU cores from the paper.
  Backward: everything trainable; g is discarded after training.
- Branches: InfoMin 2005.10243 (good views share only task-relevant information), SimCLRv2 2006.10029
  (bigger nets + distillation for semi-supervised learning).

### Interlude `ch-step`: one SimCLR training step, end to end
Stepper through one step with N = 4096: sample, two augmentations, encoder, head, normalize, the
8192 x 8192 similarity matrix, masks, loss, which weights get gradient, LARS step. A "New batch" button
redraws the toy views. Shapes computed from the paper's config. Links back to ch-infonce and ch-simclr
for the equations.

## Act III: why it works

### 6. `ch-sphere` Alignment and uniformity on the hypersphere [3D]
- Intuition: with many negatives, the loss splits into two forces (Wang & Isola 2005.10242). Alignment
  pulls positive pairs together. Uniformity spreads all embeddings evenly over the sphere, which keeps the
  most information. Temperature sets how much the push focuses on the nearest negatives (Wang & Liu
  2012.09740): a small tau spreads points more uniformly but pushes apart semantically similar images too
  (the uniformity-tolerance dilemma).
- Visual `viz-sphere` (pe3, computed toy): ~90 points in 6 latent classes on S^2, trained live by
  InfoNCE with a tau control and two loss buttons (`btn-sphere-align` = alignment only -> collapse,
  `btn-sphere-both` = alignment + uniformity -> spread). Readout prints L_align and L_uniform each click.
  Simulated in node first (`notes/sim-sphere.txt`).
- Equations: L_align (Wang & Isola Sec. 4.1.1) and L_uniform (Sec. 4.1.2; unnumbered), the asymptotic
  decomposition (Theorem 1, Eq. 2, Sec. 4.2), and the gradient's hardness weights (Wang & Liu Eq. 4-5).
- Claim cards: Wang & Isola (vision only: Tables 1, 2, 3, 5; their text result, Table 4, is 3-4 points
  worse, say so). Wang & Liu (tau 0.05..1.0, peak 0.2-0.3; uniformity printed as -L_uniform; memory-bank
  negatives; ImageNet100 tau = 1.0 differs between main and supp tables: do not reconcile).
- Branch: HaoChen 2106.04156 (spectral contrastive loss, guarantees through the augmentation graph).

### 7. `ch-negatives` Hard negatives and false negatives
- Intuition: most negatives in a batch are easy and give almost no gradient; the useful ones are close.
  But without labels some "negatives" are the same class as the anchor (false negatives). With C equally
  likely classes about 1/C of them are (our arithmetic; the papers assume a uniform class prior tau+, set to
  0.1 for CIFAR10 and 0.01 for ImageNet-100). Chuang et al. 2007.00224 correct the loss for them; Robinson et
  al. 2010.04592 also up-weight hard negatives with a concentration beta (swept 0, 0.5, 1, 2, 6; best
  0.5-1). The papers write t for temperature: the page keeps tau for temperature and tau+ for the prior.
- Visual `viz-negatives` (computed toy): an anchor and 24 negatives on a circle; some carry the anchor's
  hidden class (false negatives, outlined). Bars show each negative's share of the push (its gradient
  weight). Buttons: `btn-neg-plain` (InfoNCE), `btn-neg-hard` (beta > 0), `btn-neg-debias` (tau+ > 0).
  Prediction prompt: "which negatives get most of the push, and is that good?"
- Equations: the gradient of InfoNCE w.r.t. one negative's similarity (Wang & Liu Eq. 4; bars = Eq. 5
  shares r_i); Chuang's debiased objective; Robinson's hardness-weighted estimator.
- Results: Chuang Table 1 (ImageNet-100) and its STL10 gain in the text; Robinson's STL10 hardness x
  debias grid and tinyImageNet Table 1. CIFAR10 gain vanishes at large N (say so).
- Branch: DCL 2110.06848 (decoupled: remove the positive from the denominator; works with small batches).

## Act IV: beyond pairs

### 8. `ch-beyond` Without negatives: clusters, redundancy, and the collapse they all fight [3D]
- Intuition: negatives cost memory and include false negatives. Two ways to avoid collapse without them:
  SwAV (2006.09882) predicts one view's cluster code from the other view, and forces the codes to use all
  prototypes equally (Sinkhorn); Barlow Twins (2103.03230) drives the cross-correlation matrix of the two
  views' embeddings to the identity (diagonal = invariance, off-diagonal = no redundancy). Asymmetric
  teacher-student methods (BYOL, SimSiam, DINO) are the third way: one paragraph + link to the JEPA survey.
  Collapse comes in two forms: complete (one point) and dimensional (Jing 2110.09348: embeddings fill only a
  low-dimensional subspace); contrastive methods suffer the second one too.
- Visual `viz-barlow` (computed toy): an 8 x 8 cross-correlation heatmap trained live toward the identity;
  lambda control for the off-diagonal weight. The paper sweeps 0.002..0.02 (default 0.005) for 8192-d
  outputs; the toy scales lambda for its 56 off-diagonal entries and says so. Simulated in node first.
- Visual `viz-dimcollapse` (pe3, computed toy): a 3D embedding cloud with its singular-value bars. Buttons
  `btn-dim-healthy`, `btn-dim-collapse` (complete), `btn-dim-dimensional` (augmentation variance larger
  than data variance along one direction drives that axis to zero: Jing Theorem 1, after their Fig. 3 toy).
  Jing measure the 128-d embedding; the 2048-d representation before the projector stays healthy, which
  links back to SimCLR's throwaway head. Simulated in node first.
- Small Sinkhorn figure `viz-sinkhorn` (computed): a batch x prototypes score matrix; eps 0.05, step
  through 3 iterations (rows to 1/K, columns to 1/B); the paper reports 1 iteration fails; codes get no
  gradient (brown).
- Equations: SwAV swapped-prediction loss and the Sinkhorn solution; Barlow Twins Eq. 1-2; Jing's
  singular-value spectrum measure.
- Sourced table: ImageNet linear-probe top-1 with ResNet-50 for SimCLR, MoCo v2, SwAV, BYOL, SimSiam,
  Barlow Twins, VICReg (notes/papers/imagenet-linear-table.md: each from its own paper, epochs and
  multi-crop noted; SwAV 75.3 is Fig. 2; SwAV without multi-crop is 70.1 at 400 epochs, while the 71.8
  others print is SimSiam's reproduction). Add a note that papers print different values for the same
  baseline (SimCLR 69.3 own, 70.0 in SwAV, 69.1 in DINO).

## Act V: labels and language

### 9. `ch-supcon` When you have labels: supervised contrastive learning
- Intuition: with labels, every same-class sample in the batch is a positive. SupCon (2004.11362) pulls all
  of them together, which makes tight class clusters; it matches or beats cross-entropy on ImageNet and is
  more robust to corruptions and to hyperparameters. The triplet loss (FaceNet) and the N-pair loss (Sohn
  2016) are its metric-learning ancestors; this is the form used for retrieval and few-label tasks.
- Visual `viz-supcon` (computed toy): 15 points, 3 classes. `btn-supcon-self` (one positive per anchor)
  vs `btn-supcon-sup` (all same-class positives); arrows redraw and a short live training run shows class
  clusters forming only in the supervised case.
- Equations: SupCon L_out (Eq. 2) vs L_in (Eq. 3) and why the paper prefers L_out (gradient argument; v5
  fixed a Jensen sign). Supp. Sec. 8: the triplet loss is a special case with margin 2 tau (link to
  ch-compare).
- Sourced: ResNet-50 / ResNet-200 top-1 vs cross-entropy; ImageNet-C mCE from Fig. 3 (left); the toy's
  anchor: Supp. Table 5, one positive per anchor 69.3 vs all (13 expected) 78.5.
- Aside "In practice": few labels per class, embedding search, open-set similarity (generic).

### 10. `ch-clip` Image and text: CLIP's softmax and SigLIP's sigmoid
- Intuition: CLIP (2103.00020) makes the positive a caption: for N image-text pairs, an N x N similarity
  matrix, softmax over each row and each column (symmetric InfoNCE), learnable temperature. Zero-shot
  classification is "which caption is my partner". SigLIP (2303.15343) replaces the softmax with an
  independent sigmoid on every cell ("does this pair match?") plus a learnable bias; no batch-wide
  normalization, so it works at small batch, saves memory and needs less communication.
- Visual `viz-clipsig` (computed toy): an 8 x 8 matrix of cosine similarities. `btn-clipsig-softmax`
  shows row-softmax probabilities (rows sum to 1, each cell depends on the whole row);
  `btn-clipsig-sigmoid` shows independent per-cell probabilities with bias b. A batch-size slider shows how
  the cost of the global normalization grows (computed counts, toy).
- Equations: CLIP's loss (our transcription of the paper's pseudocode, Fig. 3) and SigLIP Sec. 3.1
  (softmax) and Sec. 3.2 (sigmoid; unnumbered) with t and b; init t' = log 10, b = -10. The printed
  sigmoid loss and Algorithm 1 disagree on the sign of b; the toy follows Algorithm 1 (logits = t x.y + b),
  the only form where b = -10 starts every pair near "no match"; say so.
- Sourced: CLIP zero-shot ImageNet 76.2 (L/14@336px) and 59.6 (RN50), linear probe 85.4 kept separate;
  no vocabulary size (text and table disagree). SigLIP: 84.5 (g/14, 4 TPUv4 chips, 2 days); small batch
  from App. B Table 5 (batch 512, 3B examples: sigmoid 51.5 vs softmax 47.7). Claim values keep the
  paper's spacing ("32 k").
- Where it stands in 2025-26 (branches, one line each): ALIGN, OpenCLIP scaling laws, MetaCLIP, SigLIP 2,
  Perception Encoder (contrastive pretraining; best features inside the network), Web-SSL (language-free
  scaling), DINOv2/DINOv3 (self-distillation, no negatives, now the default for dense features).
  Links to the VLM and OVD surveys.

## Close
- `ch-guide` When to reach for it (generic decision guide, labelled "our reading" where it is ours):
  pretrained encoder first; SupCon / triplet for few labels and retrieval; CLIP/SigLIP for zero-shot and
  text queries; self-distillation (DINOv2/v3) for dense features; contrastive fine-tuning cautions
  (batch size, false negatives, temperature).
- `ch-summary`: one paragraph.
- `ch-deeper`: paper, section, equation per item.

## Extraction needs (per group)
- A (ch-compare, ch-infonce): Hadsell 2006 Eq. 4 + margin; Chopra 2005 setup; Gutmann 2010 NCE idea;
  Sohn 2016 N-pair loss eq; FaceNet triplet loss eq + margin alpha; CPC Eq. 4, the MI bound and its
  derivation location, the image setup (crop size, patch grid, stride, encoder, context network,
  prediction steps, negatives), CPC ImageNet linear result; CPC v2 headline; Tschannen claims + setup;
  Poole one-line; CMC one-line + number; Arora one-line; Wu 2018 memory bank, NCE approximation, m=4096?
  (verify), result.
- B (ch-moco, ch-simclr, ch-step): MoCo Eq. 1-2, K, m, momentum ablation table, Fig. 3 K sweep values if
  printed, shuffling BN, linear-probe numbers, config (batch, lr, epochs, GPUs); MoCo v2 table; MoCo v3
  config, instability finding, ViT numbers; SimCLR Eq. 1, Table 1 (color strength), Fig. 5 values if
  printed, head ablation numbers, batch size/epochs table, config (batch, LARS, lr, epochs, TPUs, tau),
  linear-probe numbers; SimCLRv2 headline; InfoMin headline.
- C (ch-sphere, ch-negatives): Wang & Isola Eq./Theorem, alpha, t, direct-optimization tables; Wang & Liu
  gradient equation, tau sweep table; Chuang debiased objective, tau+, tables; Robinson estimator, beta,
  tables; DCL headline; HaoChen one-line.
- D (ch-beyond): SwAV loss, Sinkhorn, epsilon, prototypes count, multi-crop, ImageNet number; Barlow loss,
  lambda, projector size, ImageNet number; Jing's mechanism (augmentation strength vs collapse, implicit
  regularization), spectrum figure facts, DirectCLR; ImageNet linear numbers from BYOL, SimSiam, VICReg,
  DINO, DINOv2, DINOv3, iBOT (one line each).
- E (ch-supcon, ch-clip, ch-guide): SupCon Eq. 2-3, gradient argument, tables; FaceNet (shared with A);
  CLIP data size, pseudocode loss, temperature init and clip, batch size, zero-shot ImageNet; SigLIP Eq.
  1-2, init, batch-size findings, SigLiT result; ALIGN, OpenCLIP scaling, MetaCLIP, SigLIP 2, PE, Web-SSL,
  DenseCL, DetCon (one line each).
