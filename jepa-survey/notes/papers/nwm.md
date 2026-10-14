# Navigation World Models (2412.03572v2)

Lens: architecture. Sources read: abs.html, src/main.tex (shell only: the `sec/*.tex` files are not in the
bundle), src/rebuttal.tex, paper.pdf (17 pages; all equations below are transcribed from the PDF), README.md.

## 1. Identity
- Title: Navigation World Models
- arXiv 2412.03572, version read v2. v1 2024-12-04; v2 2025-04-11 (abs.html).
- Authors: Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, Yann LeCun (FAIR at Meta, NYU, Berkeley AI Research).
- Venue: CVPR 2025 (abs.html comments); README adds "(Oral)" (code-only).
- Repo: github.com/facebookresearch/nwm; weights huggingface.co/facebook/nwm; site amirbar.net/nwm.

## 2. Core problem
Visual navigation policies (GNM, NoMaD) are fixed after training: no new constraints at test time and no way to
spend more compute on a hard case. NWM instead learns a controllable video model that predicts the next
egocentric observation from past observations and a navigation action, then plans by simulating candidate
trajectories and checking whether they reach the goal image (Sec 1, p1-2).

## 3. What prior methods did and why that hurt (per the paper)
- Supervised navigation policies are "hard-coded" after training; constraints like "no left turns" cannot be
  added (Sec 1).
- Diffusion world models for RL (DIAMOND, GameNGen) are trained per environment; NWM trains one model across
  environments and embodiments (Sec 1, 2).
- Standard DiT over all context tokens costs O(m^2 n^2 d) attention; CDiT is O(m n^2 d), linear in context,
  needing 4x fewer FLOPs than DiT (intro) while predicting better (Sec 3.2, Fig 5).

## 4. The one key idea
A Conditional Diffusion Transformer (CDiT) that denoises only the target frame's latent tokens while
cross-attending to past-frame tokens, conditioned on (translation, yaw, time shift) through AdaLN; trained on
egocentric robot video with actions; scaled to 1B parameters; used as a simulator inside CEM/MPC where a
trajectory's score is the perceptual (LPIPS) similarity of its decoded final frame to the goal image.

## 5. Mechanism, step by step
1. Tokenise each frame x_i with a frozen pretrained Stable Diffusion VAE: s_i = enc_theta(x_i) (Eq 1, p3).
   Latents can be decoded back to pixels for scoring and visualisation. Input resolution is not stated in the
   paper; the README changes the NoMaD preprocessing resolution from (160, 120) to (320, 240) (code-only).
2. Action a_tau = (u, phi, k): u in R^2 forward/back and left/right translation, phi yaw change, k a time shift
   in [T_min, T_max], up to +-16 s. Actions over a k-step window are summed (Eq 2). Translation is normalised
   by each agent's average step size; backward motion is filtered out (Sec 4.1).
3. Conditioning vector: each scalar of a -> sine-cosine features -> 2-layer MLP -> R^{d/3}; concatenated to
   psi_a in R^d; same for the time shift psi_k and diffusion timestep psi_t; summed to xi (Eq 3), fed to AdaLN
   for scale/shift of LayerNorm and attention outputs. Unlabeled data (Ego4D): omit the action term.
4. CDiT block (Fig 2, p4), applied N times: self-attention restricted to the target frame's tokens; a
   cross-attention layer where target queries attend to past-frame tokens (keys/values), with a skip; then MLP.
   Complexity O(m n^2 d) for m context frames of n tokens.
5. Diffusion training: noise the target latent s_{tau+1} at a random timestep t (DiT schedule); the network
   F_theta(s^{(t)}_{tau+1} | s_tau, a_tau, t) is trained with L_simple (MSE to the clean target) plus the DiT
   covariance term L_vlb. Default: CDiT-XL, 1B parameters, context 4 frames, batch 1024 x 4 goals = 4096,
   AdamW lr 8e-5, 8 H100 machines x 8 GPUs, "same setting as DiT-*/2" (Sec 4.1, p5). Multiple goals per
   context (1 to 4) are sampled within +-16 s to create natural counterfactuals (Sec 3.1, Table 1).
6. Planning: roll out s ~ F_theta(. | s_0, a) autoregressively, decode s_T and the goal s* to pixels, score
   S = perceptual similarity (LPIPS); energy Eq 4 adds indicator penalties for invalid actions or unsafe states;
   minimise Eq 5 with CEM (App 7, p12). Standalone planning parameterises the whole trajectory as a straight
   line (one u, one phi) mapped to 8 steps at k = 0.25 s (2 s), N = 120 samples, a single CEM iteration, each
   sample scored M = 3 times (the model is stochastic). Ranking mode: sample n in {16, 32} NoMaD trajectories
   of length 8, simulate, keep the lowest energy (Sec 3.3, 4.4).

Which weights get a gradient: the CDiT only, from L_simple + L_vlb on the target-frame latent. The VAE is
frozen; LPIPS is used only at planning time.

## 6. Equations (PDF transcriptions; no .tex available)
Eq 1 (p3), the world model: `s_i = enc_\theta(x_i) \qquad s_{\tau+1} \sim F_\theta(s_{\tau+1} \mid s_\tau, a_\tau)`
- s_tau = (s_tau, ..., s_{tau-m}) the past m VAE latents; F_theta a stochastic map to the next latent.

Eq 2 (p3), action summation over a time-shifted window tau..m = tau + k - 1:
`u_{\tau \to m} = \sum_{t=\tau}^{m} u_t \qquad \phi_{\tau \to m} = \sum_{t=\tau}^{m} \phi_t \bmod 2\pi`

Eq 3 (p4), conditioning: `\xi = \psi_a + \psi_k + \psi_t`
- psi_a action embedding, psi_k time-shift embedding, psi_t diffusion-timestep embedding, each in R^d.

L_simple (unnumbered, p4), "mean-squared between the clean and predicted target":
`\mathcal{L}_{simple} = \mathbb{E}_{s_{\tau+1}, a_\tau, s_\tau, \epsilon, t} \left[ \lVert s_{\tau+1} - F_\theta(s^{(t)}_{\tau+1} \mid s_\tau, a_\tau, t) \rVert_2^2 \right]`
with `s^{(t)}_{\tau+1} = \sqrt{\alpha_t}\, s_{\tau+1} + \sqrt{1-\alpha_t}\, \epsilon`, epsilon ~ N(0, I).

Eq 4 (p5), planning energy:
`E(s_0, a_0, \ldots, a_{T-1}, s_T) = -S(s_T, s^*) + \sum_{\tau=0}^{T-1} I(a_\tau \notin A_{valid}) + \sum_{\tau=0}^{T-1} I(s_\tau \notin S_{safe})`
- S: unnormalised perceptual similarity between decoded s_T and goal s*; I: indicator with a large penalty.

Eq 5 (p5), the planner: `\arg\min_{a_0, \ldots, a_{T-1}} \mathbb{E}_s [E(s_0, a_0, \ldots, a_{T-1}, s_T)]`
- expectation over the stochastic rollout; optimised with the Cross-Entropy Method.

## 7. Config / setup table
| Item | Value | Loc |
|---|---|---|
| Model | CDiT-XL, 1B params, context 4 frames, DiT-*/2 settings | Sec 4.1, p5 |
| Tokeniser | Stable Diffusion VAE (frozen) | Sec 3.1, 4.1 |
| Conditioning | (u in R^2, phi, k) + diffusion t via AdaLN | Sec 3.2, Eq 3 |
| Time shift | up to +-16 s; 4 goals per context | Sec 3.1, Table 1 |
| Batch / lr | 1024 x 4 goals = 4096; AdamW 8e-5 | Sec 4.1 |
| Hardware | 8 H100 machines x 8 GPUs | Sec 4.1 |
| Labelled data | SCAND 8.7 h; TartanDrive 5 h; RECON 40 h; HuRoN over 75 h | App 8.1, p12-13 |
| Unlabeled data | Ego4D: 1619 videos, over 908 h (of 3,670 h total), time shift only | App 8.1, p13 |
| Held-out env | GO Stanford, ~25 h, eval only | App 8.1 |
| Eval sampling | 5 samples per model for mean and std | Sec 4.1 |
| CEM (paper) | N = 120 samples, 1 iteration, M = 3 evaluations per sample, 8 steps at k = 0.25 s, straight-line (u, phi) parameterisation; init mu_dy = mu_phi = 0, sigma^2_dy = sigma^2_phi = 0.1; (mu_dx, sigma^2_dx) = (-0.1, 0.02) RECON, (0.5, 0.07) TartanDrive, (-0.25, 0.04) SCAND, (-0.33, 0.03) HuRoN | App 7, App 8.2, p12, p14 |
| CEM (code-only) | `--num_samples 120 --topk 5 --opt_steps 1 --num_repeat_eval 3 --rollout_stride 1` | README "Trajectory Evaluation - Planning" |
| Ranking | NoMaD samples n in {16, 32}, length 8, ranked by LPIPS of final frame | Sec 4.4 |
| Metrics | ATE, RPE (trajectories); LPIPS, DreamSim, PSNR (prediction); FID, FVD (synthesis) | Sec 4.1, App 8.1 |
| Runtime | 30.3 s per 4 s trajectory on RTX 6000 Ada; 14.7 with time skip; 0.4 with distillation (250 -> 6 denoising steps) | Table 8, p14 |

## 8. Results worth remembering
Table 2 (p7), goal-conditioned navigation on RECON, 2 s trajectories (ATE / RPE, lower is better):
GNM 1.87 +- 0.00 / 0.73 +- 0.00; NoMaD 1.93 +- 0.04 / 0.52 +- 0.00; NWM + NoMaD (x16) 1.83 +- 0.03 / 0.50 +- 0.01;
NWM + NoMaD (x32) 1.78 +- 0.03 / 0.48 +- 0.01; NWM (planning) 1.13 +- 0.02 / 0.35 +- 0.01.
Table 7 (p14), all in-domain datasets, slightly different RECON numbers for the same rows (NoMaD 1.95 +- 0.05;
x16 1.88 +- 0.03; x32 1.79 +- 0.02; NWM only 1.13 +- 0.02); do not merge with Table 2. Other datasets, NWM
only vs NoMaD ATE: SCAND 1.28 vs 2.24; TartanDrive 5.63 vs 6.32; HuRoN 4.12 vs 3.73 (NWM worse on HuRoN).
Table 1 (p6), 4 s prediction on RECON: 4 goals LPIPS 0.296 vs 1 goal 0.312; context 4 0.296 vs 1 0.304;
time only 0.760, action only 0.318, action + time 0.295.
Fig 5 (p6): CDiT beats DiT per FLOP up to 1B; CDiT-L vs DiT-XL is 4x faster and better (Sec 4.2 text).
Fig 6 (p6) and Table 6 (p13), FVD of 16 s videos at 4 FPS: RECON NWM 200.969 +- 5.629 vs DIAMOND 762.734 +- 3.361.
Table 3 (p7): planning under action constraints meets all constraints with small shifts in final pose
(forward-first: +0.36 rel. delta-u, +0.61 rel. delta-phi).
Table 4 (p8), generalisation claim: adding unlabeled Ego4D improves the unknown GO Stanford environment
(LPIPS 0.658 -> 0.652; DreamSim 0.478 -> 0.464; PSNR 11.031 -> 11.083) but hurts the known RECON
(LPIPS 0.295 -> 0.368). Table 9 (p14): test-time adaptation (2k fine-tune steps) gives 0.650.
Qualitative claim (Fig 1c, Fig 8): from a single image NWM "imagines" traversals of unfamiliar environments.

## 9. Limitations the paper states
(Sec 5, p8) Out-of-distribution inputs drift toward training data (mode collapse, Fig 10); temporal dynamics
such as pedestrians are not simulated well; actions are 3 DoF only. Inference is slow (30.3 s per trajectory,
Table 8); distillation and time skip bring it to 0.4 s. No later paper in this extraction's sources fixes these.

## 10. Outline feedback
- Ch 10 lists NWM with DINO-WM and PLDM as three "designs" of latent planning. NWM is not a JEPA: it is a
  latent-diffusion video model, and the planning score is LPIPS in pixel space after VAE decoding, not a
  distance in embedding space. The chapter should say so when it introduces the "score by distance to goal
  embedding" readout, or present NWM as the generative contrast (as DINO-WM does with AVDC).
- "plan by sampling trajectories": true, but the CEM runs a single iteration over a straight-line endpoint
  (N = 120, 3 evaluations each) and the main use is ranking NoMaD proposals; it is not an iterated latent CEM.
- "scale": 1B-parameter CDiT, 64 H100 GPUs (8 machines x 8), ~129 h labelled robot video (8.7 + 5 + 40 + 75)
  plus 908 h unlabeled Ego4D for the generalisation experiment. (The 129 h sum is our arithmetic; the paper
  only lists per-dataset hours.)
- Headline metric is ATE/RPE (lower is better), not a success rate; the claim card wording must match
  (e.g. "ATE 1.13 vs NoMaD 1.93 on RECON").
- For the ch-lewm labels: NWM is frozen SD-VAE + trained 1B CDiT, i.e. "frozen features" at the tokeniser.
