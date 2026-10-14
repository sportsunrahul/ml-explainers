# Branch tier: one delta card per paper (ch-family) plus open problems (ch-map)

Sources: `sources/<name>/abs.html` (title, v1 date = `citation_date`, first author) and `paper.pdf`
(first 8 pages via pdftotext; MJEPA and Graph-JEPA tables are on pages 9-12). No LaTeX source for
these papers, so loss descriptions are paraphrased from the PDF text. "Canonical recipe" below means
I-JEPA/V-JEPA: a context encoder sees a masked view, an EMA target encoder sees the full input, a
narrow transformer predictor with mask tokens regresses target patch embeddings, L1/L2 loss on the
masked positions, no decoder.

Delta card fields: context = what the model sees; target = what it predicts; predictor = what changes
in the predictor or loss; delta = what changed vs the canonical recipe (one sentence).

## MC-JEPA
Identity: MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion
and Content Features -- 2307.12698v1 -- published 2023-07-24 -- Bardes, Adrien.
- context: two consecutive video frames I_t, I_{t+1} (flow branch) and two augmented views of an
  ImageNet image (content branch), one shared ConvNeXt-T encoder (Fig 2, p.3).
- target: the optical flow f_{t,t+1} at every pyramid level, graded by warping the layer-l features
  of I_t onto I_{t+1} (Eq 2, L_reg) plus an image-level reconstruction of I_{t+1} (Eq 3) and a cycle
  consistency term; the content branch targets VICReg invariance between the two views (p.5).
- predictor: no mask-token predictor; a PWC-Net style coarse-to-fine flow estimator network
  g_phi predicts residual flow from a 4D correlation volume (Fig 2 caption, p.3).
- delta: not a masked-prediction JEPA at all; it is VICReg (content) multi-tasked with unsupervised
  optical flow regression in feature space (motion) through one shared encoder, so the "prediction"
  is the flow warp, not a transformer predictor.
- headline: Pascal VOC frozen linear segmentation 67.1 mIoU (Table 1, p.5, row MC-JEPA, column Pascal
  VOC Frozen; VICRegL 66.8 in the same column); KITTI 2015 train EPE 2.67 (same table, row MC-JEPA).

## A-JEPA
Identity: A-JEPA: Joint-Embedding Predictive Architecture Can Listen -- 2311.15830v3 -- published
2023-11-27 -- Fei, Zhengcong.
- context: visible patches of a log-mel audio spectrogram, masked with a curriculum that moves from
  random block masking to time-frequency aware masking (abstract; Fig 2, Fig 3 p.3).
- target: EMA target-encoder embeddings of masked spectrogram regions (abstract).
- predictor: same I-JEPA predictor; what changes is the mask schedule (curriculum, Table 3 p.6:
  curriculum 48.6 vs random 47.8 AS-2M mAP) and fine-tuning with "regularized masking" that drops
  patch tokens from the attention computation instead of zeroing the input (Fig 4, p.4).
- delta: I-JEPA applied to spectrograms with a time-frequency curriculum mask and a masking-based
  fine-tuning regulariser.
- headline: AudioSet-20K 38.4 mAP with ViT-B pretrained on AudioSet only (Table 1, p.5, row A-JEPA,
  column AS-20K; Audio-MAE 37.0 in the same column); AS-2M 48.6 (same row).

## MJEPA
Identity: MJEPA: A Simple and Scalable Joint-Embedding Predictive Architecture for Audio-Visual
Learning -- 2606.25225v1 -- published 2026-06-23 -- Teotia, Revant.
- context: one shared ViT encoder E_theta takes masked audio, masked video, or masked audio-video
  token sets (Fig 2a, p.6).
- target: (a) intra-modal: EMA-encoder embeddings of the masked tokens for a->a, v->v, av->av;
  (b) cross-modal: pooled last-layer features of a different modality (Fig 2b).
- predictor: one shared intra-modal predictor P_phi plus six lightweight cross-modal predictors
  C_psi (a<->v, a<->av, v<->av) that map pooled source features to pooled target features; only the
  JEPA prediction objective is used, no contrastive or reconstruction loss (Fig 1b, Fig 2).
- delta: V-JEPA with a single encoder shared across audio and video and a cross-modal pooled
  prediction term; the paper's ablation says that without the cross-modal term a shared encoder
  falls below unimodal baselines (abstract).
- headline: AudioSet-20K audio-video frozen mAP 45.44 for MJEPA ViT-g + data scaling vs 38.60 for
  EquiAV, the best prior frozen baseline (Table 1, p.11, column A-V); the abstract states this as
  "over 6.8 mAP".

## Point-JEPA
Identity: Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point
Cloud -- 2404.16432v6 -- published 2024-04-25 -- Saito, Ayumu.
- context: patch embeddings of a point cloud (FPS + KNN groups, 1024 points, 64 centres of 32 points,
  mini-PointNet encoder) ordered by a "sequencer" so that index-adjacent embeddings are spatially
  close; a contiguous run of sorted indices is the context block (Fig 2, p.3; Sec 4.1 p.5).
- target: target-encoder embeddings of other contiguous index runs (Eq 1-2, p.4).
- predictor: unchanged I-JEPA predictor (depth 6, width 192, mask tokens + positional encoding from
  the target centre points); loss is Smooth L1 (p.4-5).
- delta: I-JEPA on point clouds where the only new part is a sequencer that turns unordered
  patches into a 1-D order so I-JEPA block masking can be reused.
- headline: ModelNet40 linear SVM overall accuracy 93.7 (Table 1, p.5, row Point-JEPA; paper reports
  93.7 +- 0.2) after 7.5 hours of pretraining on one RTX A5500 (Sec 4.1, Fig 1).

## 3D-JEPA
Identity: 3D-JEPA: A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation
Learning -- 2409.15803v1 -- published 2024-09-24 -- Hu, Naiwen.
- context: one informative context block sampled by a multi-block sampling strategy over point
  tokens (Fig 3, p.3).
- target: teacher-model (EMA) representations of several target blocks (Fig 3).
- predictor: a "context-aware decoder" that re-injects the context representation at every decoder
  layer, trained with a negative cosine similarity loss (Fig 3 caption; Sec III-D p.4).
- delta: I-JEPA on point clouds with context fed into every predictor layer (not only the input) and a
  cosine instead of L1/L2 loss, which the authors say stops the encoder from memorising context.
- headline: ScanObjectNN PB_T50_RS fine-tuned accuracy 88.65 with 150 pretraining epochs (Table I,
  p.5, row 3D-JEPA #EP 150; Point2Vec 87.5 at 300 epochs in the same column); 89.52 at 300 epochs.

## Graph-JEPA
Identity: Graph-level Representation Learning with Joint-Embedding Predictive Architectures --
2309.16014v3 -- published 2023-09-27 -- Skenderi, Geri.
- context: one context subgraph (METIS partition, 1-hop expanded, GNN-encoded), fed to a context
  encoder (Fig 2, p.5; Eq 3, p.7).
- target: not the raw embedding; each target subgraph embedding is mapped to a point
  psi = (cosh(alpha), sinh(alpha)) on the unit hyperbola in 2-D (Eq 4, p.7).
- predictor: an MLP conditioned on the target subgraph positional embedding predicts the hyperbola
  coordinates; a latent distance loss D grades the prediction (Eq 5, p.7; Fig 2f).
- delta: I-JEPA on graphs where the target is compressed to a 2-D hyperbolic coordinate so the
  learned space carries the hierarchy of graph-level concepts.
- headline: REDDIT-B graph classification 91.99 accuracy (Table 1, p.10, row Graph-JEPA, column
  REDDIT-B; best in column, LaGraph 90.4 in the same column).

## LLM-JEPA
Identity: LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures --
2509.14252v2 -- published 2025-09-11 -- Huang, Hai.
- context: one view of a sample, e.g. the natural-language Text description, encoded by the LLM
  itself (Fig 2, p.2).
- target: the LLM embedding of the paired view, e.g. the Code (regex, SQL) for the same sample.
- predictor: a Pred applied to Enc(Text), compared with Enc(Code) by a distance d, added to the
  standard next-token loss with weight lambda:
  `L_{LLM-JEPA} = L_{LLM}(Text) + lambda * d(Pred(Enc(Text)), Enc(Code))` (Eq 2, p.3).
- delta: keeps the generative next-token loss and adds a JEPA regulariser between two natural
  views of one sample; no masking, no EMA encoder.
- headline: NL-RX-SYNTH pretraining accuracy 60.59 vs 54.38 for next-token prediction alone with
  Llama-3.2-1B-Instruct (Table 2, p.6, rows L_LLM and L_LLM-JEPA; paper reports 60.59 +- 1.01 and
  54.38 +- 1.70, p = 2.94e-4).

## VL-JEPA
Identity: VL-JEPA: Joint Embedding Predictive Architecture for Vision-language -- 2512.10942v2 --
published 2025-12-11 -- Chen, Delong.
- context: video frames through an X-Encoder into visual embeddings S_V, plus a text query X_Q
  (Fig 2, p.3).
- target: the continuous embedding S_Y of the target text from a Y-Encoder, not the text tokens.
- predictor: predicts S_Y from (S_V, X_Q); trained with an InfoNCE loss against the Y-Encoder
  embeddings (Sec 3, p.3); a small text decoder is invoked only when text output is needed,
  enabling "selective decoding" (Fig 2 right).
- delta: V-JEPA style embedding prediction where the target modality is text rather than masked
  video, replacing the autoregressive VLM head.
- headline: zero-shot video classification average top-1 52.5 for VL-JEPA_BASE vs 44.7 for the best
  CLIP-style baseline, PE-Core ViT-G (Table 1, p.5, column Average); the abstract claims 50% fewer
  trainable parameters than the matched token-space VLM and 2.85x fewer decoding operations.

## D-JEPA
Identity: Denoising with a Joint-Embedding Predictive Architecture -- 2410.03755v2 -- published
2024-10-02 -- Chen, Dengsheng.
- context: unmasked tokens of an image (raw or VAE latent tokens) (Fig 1, p.2).
- target: for each masked token, both the target-encoder feature g_i (prediction loss) and the token
  value itself through a per-token diffusion or flow-matching loss p(x_i | z_i) (Fig 1 caption).
- predictor: the feature predictor gamma; its outputs condition a per-token denoising head so the
  model can generate tokens autoregressively (Sec 3).
- delta: reinterprets I-JEPA masked prediction as generalised next-token prediction and adds a
  diffusion loss, turning the architecture into an image generator.
- headline: ImageNet 256x256 class-conditional FID 2.04 for D-JEPA-H (1.4B params, 320 epochs,
  no classifier-free guidance; Table 1, p.7, row D-JEPA-H, column FID; MAR-H 2.35); D-JEPA-L 2.32.

## Causal-JEPA (C-JEPA)
Identity: Causal-JEPA: Learning World Models through Object-Level Latent Masking -- 2602.11389v2 --
published 2026-02-11 -- Nam, Heejeong.
- context: object slots from a frozen object-centric encoder over a history of frames, with selected
  object slots masked across time except a minimal identity anchor (Fig 1, Fig 2, p.1 and p.4).
- target: the masked history object states and the future object latents (joint masked-history and
  forward-prediction objective, Fig 1 caption).
- predictor: a transformer predictor over slots, conditioned on optional auxiliary variables
  (actions); no reconstruction loss.
- delta: moves masking from image patches to object-level latents so each masked object must be
  inferred from its interactions, which the paper frames as counterfactual queries.
- headline: counterfactual VQA per-question accuracy 60.19 for C-JEPA vs 41.10 for OC-JEPA, the same
  model without object masking (Table 2, p.6, column "Counterfactual VQA per que."; abstract rounds
  this to "about 20%"); Push-T planning success 88.67 with 6x128 latent features vs DINO-WM 91.33
  with 196x384 (Table 3, p.6; abstract: "1%" of the latent input features).

## EB-JEPA library
Identity: A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures --
2602.03604v3 -- published 2026-02-03 -- Terver, Basile.
- context: three worked examples: augmented image views (Image-JEPA, ResNet-18 on CIFAR-10),
  past frames (Video-JEPA on Moving MNIST), past frames plus actions (AC-video-JEPA on Two Rooms)
  (Fig 1, p.2).
- target: embeddings of the other view / the next frames / the next latent state under the action.
- predictor: small predictors with explicit anti-collapse regularisers (SIGReg or VICReg variance +
  covariance terms, temporal similarity, and an inverse-dynamics term); full AC objective
  `L = L_pred + alpha L_var + beta L_cov + delta L_sim + omega L_IDM` (Eq 13, p.6).
- delta: not a new method; a single-GPU teaching codebase that walks the canonical recipe from
  images to video to action-conditioned world models and ablates each regulariser.
- headline: CIFAR-10 linear probe 91.02% with SIGReg (Table 1, p.6, row SIGReg, column Best acc.;
  abstract rounds to "91%"); Two Rooms planning success 97 (Table 4, p.8, full model, +- 2%);
  removing the IDM term drops success to 1 (same table).

## Var-JEPA
Identity: Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture -
Bridging Predictive and Generative Self-Supervised Learning -- 2603.20111v2 -- published 2026-03-20
-- Goegl, Moritz (abs.html spells the surname with o-umlaut).
- context: same context input as JEPA, but the context encoder outputs a distribution over latents
  (Fig 1, p.6).
- target: a distribution over the target latent, with a learned conditional prior given the context
  latent playing the predictor's role.
- predictor: replaced by the conditional prior inside a single ELBO; small decoders are added so the
  ELBO has a likelihood term; no SIGReg/VICReg style anti-collapse regulariser is needed (abstract;
  Fig 1).
- delta: makes the deterministic encoder/predictor pair probabilistic and trains it as a latent
  variable model with one ELBO, giving latent uncertainty for free.
- headline: tabular Covertype test accuracy with an MLP head 0.679 for +Var-T-JEPA vs 0.408 for
  +T-JEPA (Table 2, p.8, rows MLP +T-JEPA and +Var-T-JEPA, column CO; full-coverage rows, not the
  "selective evaluation" rows).

## iBOT (foil)
Identity: iBOT: Image BERT Pre-Training with Online Tokenizer -- 2111.07832v3 -- published
2021-11-15 -- Zhou, Jinghao.
- One line: masked patch tokens are predicted by self-distillation against a teacher "online
  tokenizer" (softmax over prototypes, DINO-style), plus a cross-view [CLS] distillation loss
  (Fig 3, p.4); the foil for I-JEPA because the target is a categorical distribution over learned
  prototypes with augmentation-based invariance, not a regressed embedding of the same view.
- headline: ImageNet-1K linear probing 82.3 with ViT-L/16 (abstract; Sec 1 text p.2; not the
  Table 2 ViT-S/16 fine-tune cell that also reads 82.3).

## Open problems (ch-map)
Fetched 2026-10-08. The arXiv API (`export.arxiv.org/api/query`) returned HTTP 429 "Rate exceeded"
on four attempts, so these come from `arxiv.org/abs/<id>` pages (citation_* meta tags). Dates are
v1 dates. Quoted wording is from each abstract; TeX cleaned to ASCII.

- 2607.05238 -- Branch-JEPA: Finite-Support Predictive Distributions for JEPA World Models --
  2026-07-06 -- Song, Zhi. Stochastic futures: "most JEPA world models return one latent successor,
  even when hidden intent, partial observation, or stochastic dynamics make several futures
  plausible"; the fix is "a context-weighted finite set of latent successors". NOTE: the outline
  calls this paper "MoP-JEPA"; the arXiv title is Branch-JEPA.
- 2610.04585 -- Frozen in a Frame: The Velocity Blind Spot in JEPA World Models -- 2026-10-03 --
  Zhang, Tinghe. Velocity blind spot: "a single-frame embedding carries no velocity information, for
  any encoder, including the official released LeWM weights"; "every linear velocity probe sits at
  or below chance" on the official checkpoints.
- 2608.06799 -- Is Forward Prediction Enough? Physical State Grounding for JEPA World Models --
  2026-08-07 -- Yan, Haodong. Physical grounding: "forward-prediction objectives do not explicitly
  enforce reliable identifiability of robot-centric physical state from individual latents or state
  changes from latent pairs, which can limit downstream planning and policy performance".
- 2607.25337 -- Temporal-Distance JEPA: Plan-Aware Representation Learning for Latent World Model
  Predictive Control -- 2026-07-28 -- Bai, Jiaxin. Planning metric mismatch: "JEPA-style training
  optimizes short-horizon latent prediction, whereas planning requires a multi-step ranking of
  imagined futures by goal progress"; prior planners "inherit that ranking from embedding geometry,
  typically latent Euclidean distance".
- 2610.02344 -- Drive vs. Decay: On the Training Dynamics of Joint-Embedding Predictive
  Architectures -- 2026-10-01 -- Costa, Jose Lucas De Melo. Training dynamics: JEPAs "are prone to
  representation collapse, typically mitigated through empirical heuristics"; a linearised gradient
  flow gives a per-mode stability ratio mu_i = gamma_i / sigma_i and a predicted "phase boundary" that
  the paper confirms on Tabular-JEPA configurations.
- 2601.14354 -- VJEPA: Variational Joint Embedding Predictive Architectures as Probabilistic World
  Models -- 2026-01-20 -- Huang, Yongchao. Probabilistic views: "existing formulations rely on
  deterministic regression objectives, which mask probabilistic semantics and limit its
  applicability in stochastic control".

## Outline feedback
- ch-family: MC-JEPA is VICReg + optical-flow regression with a shared encoder, not a masked
  predictor; place it as a sibling of VICReg on the tree rather than under I-JEPA, or say so on its
  delta card.
- ch-family: EB-JEPA is a library/tutorial, not a method; its card should say "same recipe, three
  scales, each regulariser ablated".
- ch-map: 2607.05238 is titled "Branch-JEPA", not "MoP-JEPA".
- ch-timeline: `sources/lecun2022/` is empty (no abs.html, no meta.json). The only primary-source
  date in the project is the ebm-tutorial bibliography entry "Version 0.9.2, 2022-06-27"
  (notes/papers/ebm-tutorial.md); the claim is marked uncertain until the OpenReview page is fetched.
