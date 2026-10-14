# V-JEPA 2 (2506.09985v1)

## 1. Identity
- Title: V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- arXiv: 2506.09985v1 (version read). Published 2025-06-11 (abs.html: "Submitted on 11 Jun 2025"; citation_date 2025/06/11).
- Authors: Mido Assran (first; README spells "Mahmoud Assran") ... Nicolas Ballas (last); 30 authors, Meta FAIR.
- Venue: none stated (FAIR tech report, fairmeta.cls). Repo: github.com/facebookresearch/vjepa2 (README: code released 2025-06-25, code-only).
- PDF: 48 pages. Sections: 2 pretraining, 3 V-JEPA 2-AC, 4 robot planning, 5 probe classification, 6 action anticipation, 7 video QA, App. A pretraining, App. B 2-AC post-training.
- Macros in main.tex: `\model` = "V-JEPA~2", `\acModel` = "V-JEPA~2-AC", `\oldModel` = "V-JEPA". `\coloneqq` needs mathtools.

## 2. Core problem
Learn a world model mostly by observation: internet-scale video (no actions) plus a small amount of robot interaction data, so one model can understand (classify), predict (anticipate) and plan (control a real arm) in the physical world. The stated bottleneck is that interaction data is scarce and expensive, while video is abundant but action-free.

## 3. What prior methods did and why that hurt (per the paper, Sec. 1)
- World models trained purely on interaction data (RL / robot datasets) need large amounts of expensive, environment-specific data.
- Action-conditioned video generation models (GAIA, UniSim, Genie, Cosmos) predict pixels; they are judged on visual fidelity, planning by generating video is computationally costly (Sec. 4.2: Cosmos needs 4 min per action), and pixel objectives spend capacity on unpredictable detail ("each blade of grass").
- The original V-JEPA recipe (2404.08471): 2 million videos, ViT-L/ViT-H, 90K iterations, 16 frames, cosine schedule with EMA/WD ramps, absolute sincos positions. Sec. 2.1 says sincos positions were a stability problem for the largest models.

## 4. The one key idea
Scale the V-JEPA mask-denoising objective along four axes (data, model, schedule, resolution), then freeze the encoder and post-train a small action-conditioned predictor on under 62 hours of robot video. Plan by searching for the action sequence whose predicted embedding is closest (L1) to the goal image's embedding.

## 5. Mechanism, step by step

Part A, pretraining (Sec. 2, App. A):
1. Sample a clip: primary phase 16 frames at 4 fps, random-resize-crop to 256x256 (scale [0.3,1.0], aspect [0.75,1.35]). Images (ImageNet) are repeated to a 16-frame "video".
2. Patchify into tubelets 2x16x16 (TxHxW) with a strided 3D conv. Our reading: 16x256x256 -> 8x16x16 = 2048 tokens; 64x384x384 (cooldown) -> 32x24x24 = 18432 tokens.
3. Multiblock mask (same strategy as V-JEPA): Table 9 gives spatial mask scale [0.15,0.7], temporal mask scale [1.0,1.0] (every mask block spans all frames), aspect [0.75,1.5]. The block counts (8 short-range at 0.15, 2 long-range at 0.7) are stated only in the V-JEPA paper and in 2502.11831 Table S1, not in this paper.
4. Encoder E_theta (ViT-L/H/g, 3D-RoPE: feature dim split in three near-equal segments for T, H, W) processes the visible tokens only.
5. Predictor P_phi (ViT-s: 22M, width 384, depth 12, heads 12; fixed for all encoder sizes) takes encoder outputs concatenated with learnable mask tokens Delta_y that carry the positions of the dropped patches, and outputs one vector per masked patch.
6. Targets: EMA encoder E_{\bar theta} (momentum 0.99925, fixed; no ramp) on the full clip, stop-gradient. Loss = L1, applied only at masked positions (Eq. 1).
7. Gradients: theta (encoder), phi (predictor) and the mask token get gradients from Eq. 1; the EMA encoder gets none (brown).
8. Schedule (warmup-constant-decay): linear warmup 12K steps (1e-4 -> 5.25e-4), constant 228K, cooldown 12K linear decay to 1e-6; total 252K iterations. Batch 3072 (global), AdamW, WD 0.04 fixed. Phase 1 is stopped when IN1K/COIN/SSv2 probes plateau (checked every 60K steps).
9. Progressive resolution: cooldown switches to 64 frames (still 4 fps, so 16 s clips) and crop size 256, 384 or 512. Only the 12K cooldown steps pay the long-clip cost: 8.4x less GPU time than full-resolution training throughout (Sec. 2.4; Fig. 5 caption says "up to 8x"); full-res ViT-g on 64x384x384 would be "roughly 60 GPU-years" (A100, Fig. 5 middle).

Part B, V-JEPA 2-AC post-training (Sec. 3, App. B):
1. Data: Droid raw dataset, left extrinsic camera only, 4-second clips at 4 fps -> 16 frames (x_k)_{k in [16]} at 256x256 (random-resize-crop aspect (0.75,1.35)); clips under 4 s discarded, leaving "under 62 hours" (abstract: "less than 62 hours"; Sec. 3: "approximately 62 hours"); footnote 1: 23k trajectories, successes and failures, no task labels or rewards. Robot: 7-DoF Franka Emika Panda with two-finger gripper.
2. Proprioception: end-effector state s_k, a 7D vector relative to the robot base: 3 cartesian position, 3 extrinsic Euler angles, 1 gripper state. Action a_k = change in end-effector state between frames k and k+1 (7D), k in [15].
3. Frozen V-JEPA 2 ViT-g encoder E encodes each frame independently ("as an image encoder") -> z_k = E(x_k) in R^{HxWxD} = 16x16x1408 (256 patch tokens of width 1408). The paper does not say how one frame fills a 2-frame tubelet; our reading: the frame is duplicated. Nothing upstream of the predictor is trainable.
4. Tokenisation: a_k, s_k and the flattened z_k each go through their own learnable affine map to the predictor width 1024. Patch tokens get 3D-RoPE; action and state tokens get only the temporal RoPE. Sequence is interleaved as (a_k, s_k, z_k)_{k in [15]}. Our reading of shapes: per step 1 action + 1 state + 256 patch tokens; 15 steps -> 3870 tokens.
5. Predictor P_phi: ~300M parameters, 24 layers, 16 heads, hidden 1024, GELU. Block-causal attention: a patch feature at step k attends to the action, state and patch tokens of step k and of all earlier steps. Output of the last block goes through an affine map back to 1408.
6. Teacher-forcing loss (Eq. 2): L1 between hat z_{k+1} and the frozen-encoder target E(x_{k+1}), averaged over T=15 steps. Ground-truth z_k are the inputs; there is no EMA teacher in this stage.
7. Rollout loss (Eq. 3): feed the predictor's own output back as the next input and compare the final prediction with z_{T+1}; T=2 in practice, so gradient flows through one recurrent step. Total loss Eq. 4, minimised w.r.t. phi only (Fig. 6 draws it with T=4 for illustration).
8. Post-training optimisation (App. B.1): AdamW, WD 0.04, LR warmup 7.5e-5 -> 4.25e-4 over 4500 iterations, constant 85500, decay to 0 over 4500 (94500 total, our sum), batch 256.
9. Planning (Sec. 3.2, Eq. 5): encode current frame x_k -> z_k and goal image x_g -> z_g with the frozen encoder. Energy = L1 between the T-step rollout P(hat a_{1:T}; s_k, z_k) and z_g. Minimise with the Cross-Entropy Method: per action coordinate and horizon step a Gaussian initialised at zero mean, unit variance; 800 samples; 10 refinement iterations; top-10 elites update mean and variance; return the mean; execute the first action; replan (receding horizon). Horizon T=1 for all robot tasks (App. B.2: "longer planning horizons also worked reasonably well, they require more planning time"). Each action is constrained to the L1-ball of radius 0.075 (about 13 cm max displacement). 16 seconds per action on one NVIDIA RTX 4090. Blocking control.
10. Multi-goal tasks: pick-and-place gets two sub-goal images plus the final goal; 4 steps on sub-goal 1, 10 on sub-goal 2, 4 on the final goal.

## 6. Equations (verbatim from src/main.tex)

Eq. 1, `\label{eq:loss}` (Sec. 2.1, p. 4), pretraining objective:
```
\text{minimize}_{\theta,\phi, \Delta_y}\quad \lVert P_\phi(\Delta_y, E_\theta(x)) - \text{sg}(E_{\overline\theta}(y)) \rVert_1,
```
Paper: "where Delta_y is a learnable mask token that indicates the locations of the dropped patches. The loss uses a stop-gradient operation, sg(.), and an exponential moving average, bar theta, of the weights theta of the encoder network to prevent representation collapse. The loss is applied only to the predictions of the masked patches."
Gloss: x = masked view of the video (tokens dropped); y = the full video; E_theta = encoder (trainable); P_phi = predictor (trainable); Delta_y = learnable mask tokens placed at the masked positions; E_{bar theta} = EMA copy of the encoder (not trained); sg = stop-gradient; ||.||_1 = L1 distance, summed over masked tokens only.

Eq. 2, `\label{eq:loss-wm-tf}` (Sec. 3.1, p. 9), teacher-forcing loss, "with T=15":
```
\mathcal{L}_{\text{teacher-forcing}}(\phi) \coloneqq \frac{1}{T} \sum^{T}_{k=1} \lVert \hat{z}_{k+1} - z_{k+1} \rVert_1 = \frac{1}{T} \sum^{T}_{k=1} \left\lVert P_\phi\left( \left(a_t, s_t, E(x_t)\right)_{t \leq k} \right) - E(x_{k+1}) \right\rVert_1,
```
Gloss: phi = predictor weights (the only trainable thing); x_t = frame t; E = frozen V-JEPA 2 encoder; z_t = E(x_t) in R^{16x16x1408}; a_t = action (7D delta of end-effector state) between t and t+1; s_t = end-effector state (7D); hat z_{k+1} = predicted feature map for frame k+1 given ground-truth inputs up to k; T = 15 predicted steps.

Eq. 3, `\label{eq:loss-wm-rollout}` (Sec. 3.1, p. 9), rollout loss; "In practice we use T=2 ... so that we only differentiate the predictor through one recurrent step":
```
\mathcal{L}_{\text{rollout}}(\phi) \coloneqq \lVert P_\phi(a_{1:T}, s_1, z_1) - z_{T+1} \rVert_1.
```
Gloss: P_phi(a_{1:T}; s_k, z_k) here means (overloaded notation) the final feature map after running the predictor autoregressively for T steps from (s_k, z_k), feeding its own predictions back; z_{T+1} = frozen-encoder features of the frame T steps later.

Eq. 4 (unlabelled, Sec. 3.1, p. 9), total objective, "minimized with respect to the predictor weights phi":
```
L(\phi) \coloneqq \mathcal{L}_{\text{teacher-forcing}}(\phi) + \mathcal{L}_{\text{rollout}}(\phi),
```

Eq. 5, `\label{eq:energy}` (Sec. 3.2, p. 10), goal-conditioned energy:
```
\mathcal{E}(\hat{a}_{1:T};\ z_k, s_k, z_g) \coloneqq \lVert P(\hat{a}_{1:T}; s_k, z_k) - z_g \rVert_1,
```
followed in the source by `(a^\star_i)_{i \in [T]} \coloneqq \text{argmin}_{\hat{a}_{1:T}}\ \mathcal{E}(\hat{a}_{1:T};\ z_k, s_k, z_g)`.
Paper: "the model infers an action sequence (a*_i) by selecting a trajectory that minimizes the L1 distance between the world model's imagined state representation T steps into the future and its goal representation." Gloss: hat a_{1:T} = candidate action sequence (what CEM searches over); z_k, s_k = current frame embedding and end-effector state; z_g = embedding of the goal image; P = T-step rollout of V-JEPA 2-AC; minimised by CEM, first action executed, then replanned.

## 7. Config / setup

| Item | Value | Loc |
|---|---|---|
| Encoders | ViT-L 300M (w 1024, d 24, h 16, MLP 4096); ViT-H 600M (1280, 32, 16, 5120); ViT-g 1B (1408, 40, 22, 6144); embedder 2x16x16 strided conv | Table 12 |
| Pretraining predictor | ViT-s 22M, width 384, depth 12, heads 12, MLP 1536; same for all encoders | Table 12 |
| Positions | 3D-RoPE (replaces sincos; stabilises largest models) | Sec. 2.1 |
| Tubelet / patch | 2 x 16 x 16 | Sec. 2.1, Table 9 |
| Frames / fps / crop, primary | 16 / 4.0 / 256 | Table 9 |
| Frames / fps / crop, cooldown | 64 / 4.0 / [256, 384, 512] | Table 9 |
| Mask | spatial scale [0.15, 0.7], temporal scale [1.0, 1.0], aspect [0.75, 1.5] | Table 9 |
| Iterations | 12K warmup + 228K constant + 12K cooldown = 252K (vs 90K in V-JEPA) | Sec. 2.4, Table 9 |
| LR | 1e-4 -> 5.25e-4 (warmup), constant, -> 1e-6 (cooldown) | Table 9 |
| Batch / optimiser / WD / EMA | 3072 / AdamW / 0.04 fixed / 0.99925 fixed | Table 9, Sec. 2.4 |
| Loss | L1 on masked tokens vs EMA-encoder targets, stop-grad | Eq. 1 |
| Data | VM22M: 22M samples; SSv2 168K clips/168 h (w 0.056); Kinetics 733K/614 h (0.188); HowTo100M 1.1M/134K h (0.318); YT-Temporal-1B 19M/1.6M h, curated (0.188); ImageNet 1M images (0.250). "over 1 million hours of video" (abstract); YT1B described as "1.4 million video-hours" in the curation paragraph vs 1.6M in Table 1 | Table 1, Sec. 2.3 |
| Curation | PySceneDetect scenes >= 4 s: 316M scenes; DINOv2 ViT-L embeddings; 1.5M clusters; keep 210K clusters hit by K710/SSv2/COIN/EpicKitchen -> 115M scenes | App. A.2, Table 11 |
| Compute | A100 GPU-days (Fig. 5 middle); full-res 64x384x384 ViT-g "roughly 60 GPU-years"; progressive = 8.4x less. GPU count not stated. | Sec. 2.4, Fig. 5 |
| Frozen eval | 4-layer attentive probe (3 self-attn blocks + cross-attn with learnable query); 6 tasks; SSv2 16x2x3 clips, K400 16x8x3, COIN 32x8x3, Diving-48/Jester 32x4x3 at 256; ViT-g_384 uses 384 and 64x2x3 for SSv2 | Sec. 5, Table 4 caption |
| 2-AC encoder | frozen ViT-g, per-frame feature map 16x16x1408 | Sec. 3.1 |
| 2-AC predictor | ~300M, 24 layers, 16 heads, hidden 1024, GELU, block-causal; separate affine in-maps for action, state, patches; affine out-map to 1408; 3D-RoPE on patches, temporal RoPE only on action/state tokens | Sec. 3.1 Architecture |
| 2-AC inputs | Droid left camera, 4 s clips, 4 fps, 16 frames, 256x256; s_k 7D; a_k 7D deltas; T=15 | Sec. 3.1 |
| 2-AC losses | Eq. 2 (T=15) + Eq. 3 (T=2) | Sec. 3.1 |
| 2-AC optimisation | AdamW, WD 0.04, LR 7.5e-5 -> 4.25e-4, 4500/85500/4500 steps, batch 256 | App. B.1 |
| 2-AC data | under 62 hours, 23k trajectories | Sec. 3.1, fn. 1 |
| CEM | 800 samples, 10 iterations, top-10 elites, horizon 1, Gaussian init N(0,1), action L1-ball radius 0.075 (~13 cm) | App. B.2, Table 3, Sec. 4.1 |
| Inference time | 16 sec per action, one RTX 4090 (Cosmos: 80 samples, 10 it., horizon 1, 4 min) | Table 3 |
| Robots | Franka Emika Panda + RobotiQ gripper, two labs not in Droid, uncalibrated monocular RGB, blocking control | Sec. 4.1 |
| Baselines | Octo (octo-base-1.5, Open-X 1M+ trajectories, BC fine-tuned on Droid with hindsight goals); Cosmos (latent diffusion 7B, 20M hours, action-conditioned fine-tune on Droid) | Sec. 4.1 |
| Code-only | README config for 2-AC is `configs/train/vitg16/droid-256px-8f.yaml` ("8f" vs the paper's 16-frame clips; not resolved here); checkpoints ViT-L/16 300M @256, ViT-H/16 600M @256, ViT-g/16 1B @256, ViT-g/16_384 1B @384; HF ids `facebook/vjepa2-vit{l,h,g}-fpc64-256`, `facebook/vjepa2-vitg-fpc64-384`; 2-AC checkpoint trained from ViT-g; README probe results match Table 4/5 (73.7, 89.0, 32.7; 77.3, 90.2, 39.7) | README |

## 8. Results worth remembering

Scaling ingredients (Fig. 3, bar labels in assets/v1_to_v2.pdf; text Sec. 2.2): V-JEPA 1 baseline ViT-L/16 on 2M videos, 90K it.: 84.2 avg over 6 tasks -> +Data (VM22M) 85.2 (+1.0) -> +Model (ViT-g) 86.7 (+1.5) -> +Long training (252K) 87.5 (+0.8) -> +Higher resolution (384, 64 frames, train and eval) 88.2; cumulative +4.0. The last step is +0.7 by our arithmetic (not stated for Fig. 3). Note: Fig. 5 Left caption gives model scaling as +1.7 (different setup: constant LR until plateau then 64f/256 cooldown); Sec. 2.4 "Scaling temporal and spatial video resolution" gives +0.7 for 64-frame pretraining at fixed 16-frame eval (Fig. 5 Right). Curation: +1.4 (Fig. 4 Right). Not a single-model ablation: baseline and data step are ViT-L; the long-training step is stated for ViT-g.

Table 4 (p. 16, frozen attentive probe, 256 px unless noted). Columns Param / Avg / SSv2 / Diving-48 / Jester / K400 / COIN / IN1K:
- V-JEPA 2 ViT-L: 300M / 86.0 / 73.7 / 89.0 / 97.6 / 85.1 / 86.8 / 83.5
- V-JEPA 2 ViT-H: 600M / 86.4 / 74.0 / 89.8 / 97.7 / 85.3 / 87.9 / 83.8
- V-JEPA 2 ViT-g: 1B / 87.5 / 75.3 / 90.1 / 97.7 / 86.6 / 90.7 / 84.6
- V-JEPA 2 ViT-g_384: 1B / 88.2 / 77.3 / 90.2 / 97.8 / 87.3 / 91.1 / 85.1 (different protocol: 384 px, SSv2 64x2x3)
- V-JEPA ViT-H (2024): 600M / 85.2 / 74.3 / 87.9 / 97.7 / 84.5 / 87.1 / 80.0
- InternVideo2_s2-1B: 87.0 / 69.7 / 86.4 / 97.0 / 89.4 / 93.8 / 85.8; PE_core G: 82.3 / 55.4 / 76.9 / 90.0 / 88.5 / 95.3 / 87.6; DINOv2 81.1 avg, SSv2 50.7; SigLIP2 81.1 avg, SSv2 49.9.

Table 5 (p. 17, EK100 action anticipation, mean-class recall-at-5, validation; 32 frames @ 8 fps, 256 px; 1 s anticipation). Verb / Noun / Action:
- InAViT 160M: 51.9 / 52.0 / 25.8; Video-LLaMA 7B: 52.9 / 52.0 / 26.0; PlausiVL 8B: 55.6 / 54.2 / 27.6
- V-JEPA 2 ViT-L 300M: 57.8 / 53.8 / 32.7; ViT-H 600M: 59.2 / 54.6 / 36.5; ViT-g 1B: 61.2 / 55.7 / 38.0; ViT-g_384: 63.6 / 57.1 / 39.7
- Text: +12.1 action points over PlausiVL = 44% relative; probe uses encoder + predictor outputs (predictor given mask tokens for the frame 1 s ahead).

Video QA after LLM alignment:
- Controlled (Table 6, p. 20; frozen encoder, Qwen2-7B-Instruct, 18M alignment samples, 128 H100): V-JEPA 2 ViT-g_512 avg 52.3 vs PE ViT-G/14_448 49.1, SigLIP2 ViT-g_384 48.1, DINOv2 ViT-g_518 45.7. Per task V-JEPA 2: PerceptionTest 72.0 (SigLIP2 72.4 is best), MVP 31.1, TempCompass 69.2, TemporalBench 33.3, TVBench 55.9, TOMATO 37.0, MVBench 67.7.
- Scaling (Table 7, end-to-end, Qwen2-7B): ViT-L_256 avg 51.7 -> ViT-g_256 52.3 -> ViT-g_384 54.0 -> ViT-g_512 54.4.
- SOTA (Table 8, p. 21; Llama 3.1 8B Instruct backbone, full 88.5 million alignment samples, PLM recipe, MLP projector, 288 tokens per frame, 512 H100 GPUs, 32 eval frames): V-JEPA 2 ViT-g_384 avg 59.5; PerceptionTest 84.0 (test set, with SFT; others zero-shot); MVP 44.5; TempCompass 76.9; TemporalBench 36.7; TOMATO 40.3; TVBench 60.6; MVBench 73.5. PLM 8B: 56.7 / 82.7 / 39.7 / 72.7 / 28.3 / 33.2 / 63.5 / 77.1 (PLM wins TVBench and MVBench). Gains over PLM: +1.3, +4.8, +4.2, +8.4, +7.1.

Robot (Table 2, p. 14; success out of 10 trials per task per lab; two labs). Reach / Grasp cup / Grasp box / Reach-with-object cup / box / Pick-and-place cup / box:
- Octo Lab 1: 100% / 20% / 0% / 20% / 70% / 20% / 10%; Lab 2: 100% / 10% / 0% / 10% / 70% / 10% / 10%; Avg: 100% / 15% / 0% / 15% / 70% / 15% / 10%
- V-JEPA 2-AC Lab 1: 100% / 70% / 30% / 90% / 80% / 80% / 80%; Lab 2: 100% / 60% / 20% / 60% / 70% / 80% / 50%; Avg: 100% / 65% / 25% / 75% / 75% / 80% / 65%
Table 3 (p. 15, Lab 2 only, planning details). Samples / Iter / Horizon / Time / Reach / Grasp cup / box / P&P cup / box:
- Cosmos: 80 / 10 / 1 / 4 min / 80% / 0% / 20% / 0% / 0%
- V-JEPA 2-AC: 800 / 10 / 1 / 16 sec / 100% / 60% / 20% / 80% / 50%
Single-goal reaching (Fig. 8): end-effector ends within 4 cm of goal, monotonic decrease. Energy landscape (Fig. 9): minimum near (dx, dy) = (0, -0.05) for ground truth (0, -0.1). Camera sensitivity (App. B.4, Fig. 16): inferred action axis rotates with camera angle; mean abs error about 1.6 cm on ~5 cm moves; no calibration used.
Cooldown effect (Table 14, ViT-g): phase 1 (no cooldown) IN1K 83.8 / COIN 89.1 / SSv2 75.1 / K400 85.8 -> cooldown 256: 84.6 / 90.7 / 75.3 / 86.6 -> cooldown 384: 85.1 / 90.2 / 76.5 / 87.3. Eval with 64 vs 16 frames: +9.7 avg (single clip, Fig. 13).

## 9. Limitations stated
- Camera position sensitivity (Sec. 4.3, App. B.4): action axis inferred from an uncalibrated camera; positions were chosen by hand.
- Long-horizon planning: error accumulation in autoregressive rollouts and exponential search space; pick-and-place needs image sub-goals; predictions studied only up to ~16 s (Sec. 7 Future work).
- Goals must be images, not language.
- Model scale stopped at a "modest" 1B.
- EK100: 1 s horizon only, kitchens only, closed vocabulary.
Later fixes: the README (code-only) announces V-JEPA 2.1 (2603.14482, dense predictive loss, deep self-supervision, ViT-G 2B) and the outline ch-8 says it improves robotics; not verified here. Language goals and hierarchy are left open.

## 10. Outline feedback (ch-vjepa2)
- ArchFlow label "L1 vs teacher embeddings (red)" is wrong for 2-AC: the target is E(x_{k+1}) from the same frozen encoder (Eq. 2); there is no EMA teacher in post-training. Use "L1 vs frozen-encoder embedding of the next frame". "Teacher forcing" means ground-truth z_k are fed as inputs. Brown encoder + green predictor + red L1 is otherwise right; backward touches only phi.
- "action + proprio tokens": the paper's word is end-effector state s_k (7D), action a_k is its per-frame delta (7D); each is one token per timestep via its own affine map; patch tokens are 256 per frame. Show block-causal attention per timestep block of (a_k, s_k, 256 patches).
- Visual B horizon slider: all robot results use horizon T=1 (Table 3, App. B.2). Default the slider to 1 and say longer horizons "worked reasonably well" but cost planning time. CEM: 800 samples, 10 iterations, top-10 elites, 16 s per action.
- Visual A stacked bar: numbers 84.2 -> 85.2 -> 86.7 -> 87.5 -> 88.2 (Fig. 3). Caption must say it mixes model sizes (ViT-L for baseline/data, ViT-g for later steps) and that Fig. 5 reports +1.7 for model scaling under a different protocol.
- Headline 77.3 SSv2 is ViT-g_384 under a different eval protocol (384 px, 64x2x3); same-protocol ViT-g is 75.3. 84.0 PerceptionTest is test set with SFT; PLM 8B beats V-JEPA 2 on TVBench and MVBench.
- "Zero-shot on a real arm" means no data from the two labs; pick-and-place still uses two sub-goal images (4/10/4 steps). Also list the "reach with object" task (Table 2 has it).
- Progressive resolution: "from 16 frames at 256 to 64 frames at 256/384/512 during the 12K-step cooldown"; say 8.4x.
- The intuitive-physics claim card (2502.11831) tests V-JEPA (2024, ViT-H on VM2M, with RoPE), not V-JEPA 2, and predates it (Feb vs Jun 2025). Narration must not attribute it to V-JEPA 2.
- Hours: say "over 1 million hours" and "22 million videos"; 2-AC "under 62 hours", "23k trajectories".
