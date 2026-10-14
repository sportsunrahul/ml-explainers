# Fact-check of the page (fresh reader, primary sources only)

Method. All 520 `data-claim` spans in `parts/ch-*.html` were compared with `claims.json` (0 text/value
mismatches; expected, since the page is built from that file). About 150 claims, including every comparison
sentence and every headline result, were then checked against the primary source at the stated location
(tex, PDF text, README, released config, scan JSON). Every display equation was checked against the tex.
Every figure was checked for its badge, caption and (for Computed/Sourced) the literals in `parts/viz-*.js`.

Counts: 4 MUST FIX, 17 NITS. No wrong equation. Every dated paper with a directory under `sources/` has the
date its abs page gives. Unverified from the project (no primary source present): the five dates and five
quotations in ch-map "What the field is arguing about" (Branch-JEPA, Frozen in a Frame, PSG, Temporal-Distance
JEPA, Drive vs Decay); the LeWM Fig. 6 bar values (74 / 92 / 86 / 87 / 100 / 74 / 86) and Fig. 3 bars (0.98 s
/ 47 s), which exist only as figure graphics (the 96 / 78 Push-T values and 48x are in Table 5 and the text);
PLDM's "16-step horizon" in the ch-planning toy caption.

## MUST FIX

1. ch-vjepa (Sec. "What changes, and what does not", Table 4 sentence).
   Page: "a causal mask that hides only the last 6 of the 16 frames gives 61.3 and 49.8".
   Source: `causal multi-block[p]`: "x is restricted to the first p frames of the 16-frame video, which are
   then masked with a random set of spatio-temporal blocks" (V-JEPA tex, Table 4 caption and Sec. 4 text,
   `sources/vjepa/src`, lines 354 and 361 of the concatenated tex). Row `causal multi-block[6]` = 61.3 / 49.8.
   Correction: the context is limited to the first 6 frames (plus block masking inside them); the last 10
   frames are hidden, not the last 6.

2. ch-vjepa (Sec. "What it bought").
   Page: "Trained on about 2 million videos ... for 90K steps at batch 3072, the frozen ViT-H/16 at 384 pixels
   reaches 81.9 ...".
   Source: "We use a batch size of 3072 for the ViT-L/16 and ViT-H/16 models, and a batch size of 2400 for the
   ViT-H/16_384 model" (V-JEPA Sec. 3.4; Appendix hyper-parameter table: batch_size 2400 for ViT-H/16_384;
   iterations 90000 for all three).
   Correction: 2400 for the 384-pixel model (3072 is the 224-pixel runs), or drop the batch from this sentence.

3. ch-vjepa21 (Sec. "What it bought").
   Page: "Against V-JEPA 2's ViT-g, the 2.1 ViT-g gains +23.4 mIoU on ADE20K, +27.6 on Cityscapes and +20.7 on
   VOC with a frozen encoder (Table 8)".
   Source: V-JEPA 2.1 Sec. 3.5: "V-JEPA 2.1 ... obtaining 85.0 mIoU on VOC12, 73.5 mIoU on Cityscapes and 47.9
   mIoU on ADE20K. Compared with its previous version V-JEPA 2, the gains are ... +23.4 points in ADE20K,
   +27.6 on Cityscapes, and +20.7 on VOC12". Those are the ViT-G (2B) numbers. Table 8 rows: V-JEPA 2 ViT-g
   24.4 / 45.9 / 63.9; V-JEPA 2.1 ViT-g 47.8 / 71.8 / 84.7; V-JEPA 2.1 ViT-G 47.9 / 73.5 / 85.0.
   Correction: "the 2.1 ViT-G (2B)" (or recompute for the 1B model: +23.4 / +25.9 / +20.8). The 2B/1B mix in
   the next clause ("the 2B model's 0.307") then reads correctly.

4. ch-map (Sec. "The 2026 explosion").
   Page: "447 in the first nine months of 2026, 97 of them in September alone".
   Source: `notes/arxiv-scan-2026-10-08.json` (and the literals in `parts/viz-map.js`): 2026 = 447 papers,
   of which 31 are dated 2026-10; Jan-Sep 2026 = 416. The same file has one 2022 paper (slow-features,
   2022-11-20), so "7 in 2023, 38 in 2024, 76 in 2025, 447 in 2026" sums to 568 of the 569.
   Correction: "447 so far in 2026 (416 in its first nine months)" and add "1 in 2022", or say the years sum
   to 568 because one paper is from 2022.

## NITS

1. ch-ijepa, "worth 67.3 vs 56.1 in the paper's Table 11": that ablation is a ViT-H/16 for 300 epochs
   (1% low-shot linear probe), not the ViT-B/16 used by every other ablation in the paragraph. Name the model.
2. ch-ijepa, Clevr sentence: the I-JEPA row in Table 4 is ViT-H/14 (unnamed on the page) against DINO ViT-B/8
   and iBOT ViT-L/16. Also, MAE ViT-H/14 scores 90.5 on Clevr/Count (above I-JEPA's 86.7) and ties on
   Clevr/Dist; the page's "they lose" is true of the view-invariance methods only, which is what the
   sentence says, but a reader may infer I-JEPA is best overall.
3. ch-ijepa mask figure caption: "The alternatives are the paper's Table 6 ablations". The "Tiny targets"
   button (scale 0.0075-0.01 in `viz-ijmask.js`) is the page's own setting; the smallest paper setting is
   (0.075, 0.2) in Table 8 (19.2%). Say so.
4. ch-iwm equation: the paper's loss is summed over the target positions, L = sum_{i in M_x^C} || p_phi(...)_i
   - f^EMA(y)_i ||^2 (IWM Sec. 3, unnumbered). The page drops the sum and the per-position index. Not wrong
   as a sketch, but the caption claims it is the paper's loss.
5. ch-vjepa2, "From 2 million to 22 million videos (over 1 million hours, plus 1M images tiled into still
   clips)": VM22M is 22 million samples including the 1M ImageNet images (V-JEPA 2 Sec. 2.1, Table 1), so
   "22 million videos plus 1M images" double counts.
6. ch-vjepa21, "a 142M-image set and 19M more videos": the 19M YT-1B videos were already in V-JEPA 2's
   VideoMix22M (V-JEPA 2.1 Table 3 lists YT-1B 19M under both the V-JEPA 2 and V-JEPA 2.1 weight columns;
   V-JEPA 2 Table 1 has the same row). The new data is LVD-142M. Reword: "the 142M-image LVD set on top of the
   19M curated YT-1B videos".
7. ch-vjepa21, "with a longer horizon of 8 and more refinement it reaches 80%": that row also uses 300 samples
   (vs 800) and 14 s per step (vs 3 s) (Table 6). Mention the sample count or say "a different planner
   budget".
8. ch-planning, NWM: "trained on about 129 hours of robot navigation video" is a bare number. It is the sum of
   SCAND 8.7 h + TartanDrive 5 h + RECON 40 h + HuRoN 75 h from NWM App. (datasets list); wrap it or mark it
   "our reading".
9. ch-planning, NWM: "each plan takes 30.3 seconds". The 30.3 s (NWM rebuttal Table 8) is the runtime of
   simulating a single 4-second trajectory when ranking 32 of them; a plan from scratch uses 120 samples
   (Sec. 3.3). Reword to "simulating one candidate trajectory takes 30.3 s".
10. ch-lewm, "PLDM needs seven loss terms with six weights to tune": bare but correct (LeWM App. B: "seven-term
    training objective", six coefficients); wrap it in a claim.
11. ch-lejepa, "Theorem 1 shows that for nearest-neighbour and kernel probes ... unique minimiser": the main
    text Theorem 1 ("isotropic Gaussian Optimality") is the general statement; the k-NN and kernel versions
    are Theorems 7 and 8 in the appendix. Say "Theorem 1 (and Theorems 7-8 in the appendix)".
12. ch-lejepa, "(Table 3)" for Galaxy10 78.17 vs 71.38: that Table 3 is in the appendix (p. 42); the main-text
    Table 3 does not exist under that number in the body. Say "appendix Table 3" to avoid confusion with
    Table 2 (transfer) two sentences earlier.
13. ch-vjepa, "Delta from I-JEPA" table: "width 384, mask tokens with 2D positions" for I-JEPA is code-only
    (sin-cos positions) and is not marked as such here, while the chapter marks it elsewhere.
14. ch-vjepa2 robot paragraph: "Octo, a behaviour-cloning policy trained on 1M trajectories" is Octo-base-1.5
    pretrained on Open-X (over 1M trajectories) and then fine-tuned on Droid (V-JEPA 2 Sec. 4.1 footnote);
    "pretrained on" is more accurate than "trained on".
15. ch-vjepa21, "Nothing global was lost at 2B: Something-Something 77.7 against 77.3, Kinetics 87.7 against
    87.3": the same table (V-JEPA 2.1 "Performance on global tasks") has Diving-48 at 90.2 for V-JEPA 2 ViT-g
    (bold, best) and 89.2 for V-JEPA 2.1 ViT-G. Correction: "SSv2 and Kinetics held; Diving-48 dropped 90.2 to
    89.2."
16. body-top.html hero: the header animation (`viz-hero.js`: synthetic points, a hand-placed "guess") carries
    no badge and no caption. It is decorative, and the note below it explains the badge scheme, but the rule
    "every figure with synthetic content carries the Toy model badge" is not met there. Add the badge to the
    hero or say in the note that the hero is a toy.
17. ch-planning toy caption: "PLDM instead sums the distance over every step of a 16-step horizon with an
    ensemble-variance penalty (MPPI)". Sum over t = 0..H, ensemble variance penalty and MPPI match PLDM
    Sec. 3.3 (Eqs. for C_goal, C_uncertainty); the horizon H = 16 was not found in the text and is unwrapped.
    Wrap it or drop the number.

## Verified (sample log; value and sentence both correct unless noted above)

- ch-ssl: MAE 75% / 87.8; SimSiam 67.7 / 0.1 (Fig. 2); BYOL Eq. (1), tau_base 0.996; VICReg Eqs. (1) variance,
  (4) covariance, (5) invariance; data2vec K=6 for vision; MSN 1024 prototypes; recon 90%/45% vs 20%/55%
  (Sec. 1 and Fig. 4), ResNet9, Theorem 1, Eq. (7), Fig. 5 (R3, 50% to almost 90%), "exponentially slower";
  all six ancestor dates.
- ch-ebm: tutorial Sec. 3, 3.2, 4.1/4.2; Figs. 4, 5 ("can collapse!"), 9, 10 (four terms), 11; Eq. (3) hinge,
  Eq. (4) NLL; the latent-variable quote; Ref. [6] "Version 0.9.2, 2022-06-27"; JEPA equation matches Fig. 9
  labels (s_x, s_y, Pred, D).
- ch-ijepa: masking ranges and Fig. 4; Tables 1, 4, 6, 7 (66.9 at 500 ep vs 40.7 at 800 ep), 8, 10, 11
  numbering and values; loss equation (unnumbered, inner sum not averaged, as the page says); EMA 0.996 to 1.0;
  predictor width 384, depth 12 (6 for B/16); 16 A100 under 72 h; 1200 GPU hours; 2.5x / 10x; 7% / 5x;
  AdamW, lr 1e-4 to 1e-3 (15 ep) to 1e-6, wd 0.04 to 0.4; README 16 A100 80G, batch 2048; released config
  (batch 128, crop 0.3-1.0, min_keep 10, start_lr 2e-4, warmup 40, bf16); 626 steps/epoch is a correct
  derivation.
- ch-collapse: Littwin Theorem 4.2, Eqs. (7) JEPA / (8) MAE, Corollary 4.3 Eq. (9) rho^L vs rho^(L/(L+1)),
  width 100 depth 5; slow-features 17 frames, 28 px, Table 1 values; uniform-prior 960 vs 2 classes, Table 1
  deltas (-11.2, -17.7, -17.7, -0.8, -0.1), iNat +3.0.
- ch-vjepa: Eq. (1) with sg and Delta_y and L1; 16 frames stride 4 ~3 s; 2x16x16 tubelets, 8x14x14 = 1568;
  EMA 0.998 to 1.0; predictor 12 x 384; short 8 x 15%, long 2 x 70%; ~90% masked; Table 1 (73.7 vs 68.6; 85.6
  vs 85.4); Table 3 (+17.3, 56.7 to 73.7); Table 4 values; Table 6 (81.9 / 72.2 / 77.4; DINOv2 ViT-g 83.4 /
  50.6 / 86.2; "+21 points on SSv2" is the paper's own statement); abstract 77.9 with two-layer probe.
- ch-iwm: Table 2 MRR values (0.09, 0.25, 0.81 / 0.85, 0.00); Table 4 (83.3, +1.8, 83.3, 84.4, MAE +0.3,
  I-JEPA +0.9); linear 67.5 vs 74.5; bank 256; three-layer conditioning net; 300 epochs.
- ch-vjepa2: Fig. 3 ladder 84.2 / 85.2 / 86.7 / 87.5 / 88.2 (Sec. 2.4 text: +1.0, +1.5, +0.8, cumulative
  +4.0); Fig. 5 +1.7; 22M, >1M hours, 1M images; 252K vs 90K; 12K cooldown at 64 frames (16 s) and 384 px;
  60 GPU-years; 8.4x; EMA 0.99925; 22M predictor; 3D-RoPE; Table 4 (77.3 SSv2, 90.2 Diving-48); Table 5
  (39.7 vs PlausiVL 27.6, 44%, 1 s anticipation, probe on encoder + predictor); Table 8 (Llama 3.1 8B, 88.5M,
  59.5; PLM 8B wins two columns); AC: 62 h, 300M, 24 layers, 16 heads, 1024, 23k trajectories, 4 s clips at
  4 fps, 256x256, 7-DoF / 7D, 16x16x1408, Eqs. (2)-(4), T=15, T=2; batch 256, wd 0.04, lr 7.5e-5 to 4.25e-4
  over 4500, held 85500; CEM 800 / top 10 / 10 / horizon 1 / 16 s / RTX 4090; Table 2 averages (100, 65, 25,
  80, 65; Octo 15); Table 3 (Cosmos 4 min, 0%); 1.6 cm on 5 cm moves; two labs, 10 trials.
- IntPhys: 98 / 66 / 62; "only method significantly above untrained networks on all three"; 3 vs 10
  properties; chance 50; VideoMAEv2; n=20 untrained; 128 h over 70%; fixed context of 2 frames; Table S4
  0.28% (V-JEPA-H Avg, object permanence All) vs human 12.5%.
- ch-vjepa21: Eqs. (1)-(3); lambda 0.5 for video; 4 depths, blocks 12/24/36/48; predictor 24 vs 12; Table 1
  rows (72.8 / 22.2; 62.5 / 33.8; 72.1 / 38.6; 77.7 / 47.9); Table 2 constant 1.0 = 51.1; 142M; 2B; cooldown
  64 frames; Table 8 (24.4 vs DINOv2 49.5; 0.307 vs DINOv3 7B 0.309; 55.9 vs 47.9); global 77.7 vs 77.3, 87.7
  vs 87.3, 1B row slightly below; Table 6 robot 60 / 70 / 80 (horizon 8); 20% in abstract; nav 10.6 s vs
  103.2 s and matched ATE (1.146); v1 2026-03-15, v3.
- ch-lejepa: 17 knots on [-5, 5]; 1024 slices; "|A| as low as 16" (Sec. 4.3); Theorem 4 bound 4 sigma^2/N;
  Definition 2; (Epps-Pulley) and (LeJEPA) tags; Eq. (7); lambda 0.05; 2 global views; ~50 lines; 79%
  ViT-H/14; 77.1 / 78.5; Table 2 79.48 vs 78.50; 60+ architectures; 1.8B; Spearman ~85%; Galaxy10 11,000,
  78.17 vs DINOv3 ViT-S/16 71.38; CIFAR-100 86.93 vs 70.44; Table 4 (no predictor / teacher); "only justified
  when there exists an asymmetry". GE: Eq. (5), dims 64-512, Pearson 0.64-0.94, 2025-10-07. WM: 1024 dims,
  R^2 > 0.999, InfoNCE 0.720241 at 1024, zero sorry, 2026-05-25.
- ch-planning: DINO-WM 14x14x384, 19M, Table 1 (0.98 / 0.96 / 0.92 / 0.90; DreamerV3 0.30), Table 2 (CLS
  0.44, ResNet 0.20), Table 8 GD 0.22, CEM 100 x 10, 2024-11-07; PLDM 64 px, 2218672 params, six methods, 23
  datasets, Table 2 (97.8; HILP 100.0 both columns; 34.4 no-door), 26.2 / 24.0 (App. G), 5 layouts, ~4x
  slower, 13.4 without variance term, 2025-02-20; NWM 1B, CVPR 2025, 120 samples, Table 2 1.13 vs NoMaD 1.93,
  2024-12-04.
- ch-lewm: two terms, 15M / 5M / 10M, single GPU, Eqs. (1)-(3), lambda 0.1, 1024 directions, six to one, 192,
  final LayerNorm quote, zero-init AdaLN, CEM 300 / 30 / horizon 5, 96 / 78 (Table 5), 48x, probe 0.052,
  2026-03-13; H-JEPA 18% to 73%, 40.7 to 96.0, 32.0 to 60.0, HWM identity-encoder control 38.7, 2x frequency
  gap (Fig. 5), rank 1-14 of 384 (Fig. 17), frameskip 5, 2026-10-05.
- ch-family: all twelve dates; MC-JEPA VOC frozen 67.1; A-JEPA 38.4 vs Audio-MAE 37.0; Graph-JEPA REDDIT-B
  91.99; Point-JEPA 93.7, 7.5 h; 3D-JEPA 88.65 at 150 ep; D-JEPA-H FID 2.04, 1.4B; LLM-JEPA 60.59 vs 54.38
  (NL-RX-SYNTH); VL-JEPA 52.5 vs PE-Core ViT-G 44.7; EB-JEPA 97% / 1% without IDM; Causal-JEPA 60.19 vs 41.10
  (counterfactual VQA per question); Var-T-JEPA Covertype 0.679 vs 0.408; MJEPA 45.44 vs EquiAV 38.60.
- ch-map / closing: 569 unique ids, 7 / 38 / 76 / 447, 97 in 2026-09; arXiv versions in "Where to go deeper"
  match the abs pages.
- Figures: all 15 figures carry a badge; every Toy-model figure says "Toy model" in its caption; `viz-scale.js`
  holds the Fig. 3 ladder, `viz-tree.js` the abs-page dates, `viz-map.js` the scan counts, `viz-ijmask.js`
  the Fig. 4 ranges (plus the page's own "tiny" setting, see NIT 3), `viz-tube.js` the V-JEPA mask config,
  `viz-dense.js` lambda/sqrt(d_min) with 0.5, `viz-sigreg.js` 17 knots on [-5, 5], `viz-ijsched.js` the paper
  and config schedules.
