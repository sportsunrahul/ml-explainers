# Intuitive physics from video SSL (2502.11831v1)

## 1. Identity
- Title: Intuitive physics understanding emerges from self-supervised pretraining on natural videos
- arXiv: 2502.11831v1 (version read). Published 2025-02-17 (abs.html: "Submitted on 17 Feb 2025").
- Authors: Quentin Garrido (first) ... Yann LeCun (last); 8 authors; FAIR at Meta, Univ Gustave Eiffel, EHESS.
- Venue: none stated (fairmeta.cls). Repo: github.com/facebookresearch/jepa-intuitive-physics (code CC-BY-NC; raw surprises and performance.csv per model/dataset are in `data_intphys.tar.gz`, not in our sources).
- PDF: 24 pages. Main text has unnumbered sections; appendix equations are S1-S4, tables S1-S5, figures S1-S11.
- Model tested: V-JEPA (2024, 2404.08471), not V-JEPA 2. Flagship for Fig. 1 and the human comparison: ViT-H pretrained on VideoMix2M. Per-property stats and all ablations: ViT-L pretrained on HowTo100M (n=5 seeds). Both use RoPE instead of V-JEPA's sincos (Table S1 caption).

## 2. Core problem
Do deep networks acquire intuitive physics (object permanence, continuity, shape constancy, gravity, support, solidity, inertia, collision) from a generic learning objective on natural video, with no hard-coded "core knowledge"? And does it matter whether prediction happens in pixel space, in text, or in a learned representation space?

## 3. What prior methods did and why that hurt (per the paper)
- Structured models hand-code objects and 3D relations (a built-in "game engine"); they implement the core-knowledge hypothesis rather than test it.
- Pixel-based generative models reconstruct future frames; on these benchmarks they score near chance (confirmed here with VideoMAEv2).
- Multimodal LLMs reason in text and have no video-prediction objective; they also score near chance (Qwen2-VL, Gemini 1.5 pro).

## 4. The one key idea
A model trained to predict masked video in representation space can be read out zero-shot as a "surprise" meter: run it causally, measure the L1 error between predicted and actual future representations, and the implausible video of a matched pair is the one with the higher surprise (violation of expectation).

## 5. Mechanism, step by step
1. Pretrain V-JEPA (Eq. S1): context encoder f_theta on a block-masked clip, predictor p_phi predicts the EMA target encoder's features of the masked part; L1 loss. 16 frames at 5.33 fps (max 3 s), 224x224, patches 16x16x2, predictor 12 blocks of width 384, batch 3072, 90000 iterations (schedule stretched to 112500), AdamW, A100 80G. Masking: Block (8 blocks at scale 0.15 + 2 at 0.7, aspect U(0.75,1.5), spanning all frames); Causal Block (Block + last 4 frames fully masked); Random (90% of patches).
2. At test time, no adaptation. Slide a window of C+M frames over the video with stride s=2 (start frames 1, 3, 5, ...). Context encoder sees the first C frames; the predictor predicts the representations of the M future frames; the target encoder g_psi encodes the full window and keeps the future frames. Surprise at t is the L1 distance between the two (Eq. S2).
3. Aggregate over the video: AvgSurprise (mean over start frames) or MaxSurprise (Eq. S3). Pairwise classification uses Avg; single-video classification uses Max.
4. Pairwise decision: the video of a pair with the higher surprise is called impossible. Accuracy = fraction of pairs right; chance = 50%. Relative surprise = surprise(impossible) - surprise(possible).
5. Context length C is chosen per property to maximise accuracy (C+M fixed at 16 or 32; C in {2,4,6,8,10} x (C+M)/16); on IntPhys the min surprise over all C per start frame removes the sweep. Frame skip sets the fps (Table S3).
6. Horizon (our reading): with window 16 and the fixed-context setting of C=2, the model predicts 14 frames ahead; at IntPhys's 7.5 fps that is about 1.9 s; at GRASP's 5 fps about 2.8 s; InfLevel uses window 32 at 6 fps.
7. Control for low-level bias: 20 untrained (randomly initialised) networks of the same architecture; Welch's one-tailed t-test and Hedge's g effect size of V-JEPA (n=5) over untrained (Fig. 2A).
8. Pixel baseline: VideoMAEv2-g, same protocol, target = normalised pixels of 16x16x2 patches. MLLM baselines: both videos in one prompt, "Exactly one of the two videos has an event which breaks the laws of physics ... which one is it?", order shuffled, temperature 0, refusals counted as errors (Gemini refused 10-15% of the time).

## 6. Equations (verbatim from src/paper.tex)

Eq. S1 (Sec. A.1, p. 13), V-JEPA training objective, "predict the representations of bar V_C from V_C":
```
 \|p_\phi\left(f_\theta\left(V_C\right)\right) - f_{\theta^{EMA}}(\bar{V_C}) \|_1 .
```
Gloss: V = video; V_C = corrupted (block-masked) video; bar V_C = the masked complement; f_theta = context encoder; f_{theta^EMA} = target encoder with EMA weights theta^EMA_{t+1} = (1-alpha) theta_t + alpha theta^EMA_t; p_phi = predictor; L1 over the masked tokens. "If we instead use the first C frames to predict the rest of the video, this objective turns into a measure of error for the prediction of the future."

Eq. S2 (Sec. A.7, p. 16), surprise at time t:
```
    S_t = \| p_\phi\left(f_\theta\left(V_{t:t+C}\right) \right) - g_\psi\left(V_{t:t+C+M}\right) \|_1 .
```
Gloss: V_{t:t+C} = the C context frames starting at t; f_theta = context encoder; p_phi = predictor producing M future frames' representations; g_psi = target encoder, applied to the C+M window, keeping the M future frames (for V-JEPA: the EMA encoder's features; for VideoMAEv2: normalised pixels); S_t = L1 distance = surprise.

Eq. S3 (Sec. A.7, p. 16), per-video aggregation:
```
    \texttt{AvgSurprise} = \frac{1}{T} \sum_{t \in \{1,1+s,...,T-(C+M)\}} S_t \quad\textbf{or}\quad     \texttt{MaxSurprise} = \max_{t \in \{1,1+s,...,T-(C+M)\}}S_t .
```
Gloss: T = number of frames in the video; s = stride (2); Avg is best for pairs; Max is best for single videos ("eliminating the surprise contribution coming from the complexity of the scene").

Eq. S4 (Sec. A.7, p. 18), Qwen2-VL normalised answer probability:
```
P = \frac{P("1")}{P("1") + P("2")} \quad or \quad \frac{P("2")}{P("1") + P("2")}
```
Gloss: probability mass on the answer tokens "1"/"2"; Fig. S2 shows values near 0.5 for possible and impossible videos alike.

## 7. Measurement setup
| Item | Value | Loc |
|---|---|---|
| Benchmarks | IntPhys dev set (synthetic, diverse scenes, ~360 videos = 120 per property, 3 properties: object permanence, continuity, shape constancy); GRASP (synthetic, ~4000 videos, 10 properties); InfLevel-lab (real videos, ~4000, 3 properties: object permanence (renamed from continuity), gravity, solidity; gravity and solidity need a contextualising prefix video) | Table S2, Sec. A.4 |
| IntPhys private test set | 3600 videos per property; human data from Riochet et al. 2022 (MTurk) | Sec. B.3 |
| Metric | pairwise classification accuracy (chance 50%); on the test set: pairwise error rate and single-video 1-AUROC | Fig. 1, Tables S4-S5 |
| Models | V-JEPA ViT-H (VM2M) flagship; ViT-L (HowTo100M) n=5 for stats/ablations; Fig. 3C sizes Base 115M, Large 405M, Huge 842M | Sec. "Per property", Fig. 3 |
| Baselines | VideoMAEv2-g (pixel prediction); Qwen2-VL-72B (open MLLM; main text twice says "Qwen2-VL-7B", legend, Table S3, Fig. S2/S10 say 72B); Gemini 1.5 pro (closed); untrained networks n=20 | Fig. 1A legend, Sec. A.6 |
| Eval hyperparameters | IntPhys: frame skip 2 (7.5 fps), window 16, stride 2; GRASP: skip 10 (5 fps), window 16; InfLevel-lab V-JEPA: skip 5 (6 fps), window 32; VideoMAEv2 on InfLevel: skip 10 (3 fps), window 16; MLLMs see all frames at 3 / 5 / 1.5 fps (Qwen) and 7.5 / 5 / 1 fps (Gemini) | Table S3 |
| Context choice | best C per property (sweep); fixed C=2 across everything costs little for V-JEPA (Fig. S3) | Sec. A.8, Sec. B.1 |
| Held fixed in ablations | compute = 30 years of video seen (text; Sec. A.3 says 90000 iterations = "around 26 years"); HowTo100M subsets 0.1% = 128h, 1% = 1289h, 10% = 537d, 100% = 14.7y (Fig. 3B axis) | Sec. "Keys", Fig. 3 |
| What the data confirm | none of the benchmarks are seen in training; encoders frozen; no task-specific training | Sec. A.4 |

## 8. Results worth remembering
- Headline (Fig. 1A, p. 2; text "Representation prediction learns..."): V-JEPA(-H) pairwise accuracy 98% (95% CI [95%,99%]) on IntPhys, 66% ([64%,68%]) on GRASP, 62% ([60%,63%]) on InfLevel-lab. V-JEPA is "the only method that achieves significantly higher performance than untrained networks across all datasets".
- Baselines: no numbers are printed anywhere in the paper; the text says VideoMAEv2, Qwen2-VL and Gemini 1.5 pro are "only marginally above" randomly initialised models. Our reading of the Fig. 1A bars: untrained and all three baselines sit roughly between 47 and 64 on IntPhys and GRASP and near 50-55 on InfLevel-lab, with CIs overlapping chance. Exact values exist only in the repo's performance.csv (not in sources). Do not print specific baseline numbers.
- Per property, ViT-L on HowTo100M vs untrained (Fig. 2A text, p. 5): IntPhys object permanence 85.7 vs 51.4 (g=9.0), continuity 86.3 vs 51.2 (g=11.0), shape constancy 83.7 vs 51.7 (g=8.1); GRASP object permanence 70.7 vs 54.1, continuity 65.0 vs 55.0, support 98.1 vs 58.4, gravity 74.9 vs 55.3, inertia 62.0 vs 54.3; InfLevel object permanence 72.1 vs 52.5 (g=5.4). Not significant (p > 0.05): GRASP colour constancy, solidity, collision; InfLevel gravity, solidity.
- IntPhys private test set, pairwise error (Table S4, p. 22), object permanence / shape constancy / continuity "All" columns: V-JEPA-H Avg 0.28% / 0.0% / 0.09%; V-JEPA-H Max 4.4% / 4.4% / 12.87%; V-JEPA-L Avg 1.4% / 3.1% / 2.0%; Riochet 2020 12.0% / 21.0% / 41.0%; Human 12.5% / 14.5% / 30.0%.
- Single-video 1-AUROC (Table S5): V-JEPA-H Max 19.2% / 21.9% / 29.67%; Human 24.0% / 26.0% / 38.0%; V-JEPA-L Max 40.0% / 41.8% / 41.6% (scale matters for the harder task).
- Ablations on IntPhys (Fig. 3, p. 6; ViT-L, bars not labelled numerically): masking: Random costs "only around 5 points" vs 20 on classification tasks; Causal Block is worse than Block. Data: SSv2-only near chance; K710 above chance; HowTo100M best; 128h of unique video keeps "over 70% on all considered properties"; abstract phrases 128h as "one week of unique video". Size: 115M model "over 85%".
- Relative surprise > 0 for every IntPhys property (one-sample t-test; Avg surprise means 7.8e-03, 7.1e-03, 8.5e-03; Sec. A.7).
- Fixed context C=2 for all properties keeps V-JEPA non-trivial (Fig. S3); GRASP is the most context-sensitive (Fig. S4).

## 9. Limitations stated (Discussion, Sec. B.2)
- Near chance on interactions between objects (solidity, collision) and on InfLevel gravity/solidity, which need a contextualising event the model cannot remember.
- Short memory: clips of 3-4 s; performance depends on context length; frame-rate constraints.
- Predicts only as an observer; cannot condition on actions. V-JEPA 2-AC (2506.09985) adds action conditioning; hierarchical JEPA for interactions is left open.
- Benchmarks have low-level biases (untrained networks above chance on GRASP); relabelled InfLevel raises untrained networks too.
- Qwen2-VL / Gemini may be handicapped by the text-only protocol; the paper says the result does not mean LLMs cannot learn intuitive physics.

What would falsify the claim ("prediction in a learned representation space is sufficient; core knowledge is not needed"): (a) V-JEPA surprise not significantly above untrained networks of the same architecture (their Welch tests and n=20 control); (b) relative surprise (impossible minus possible) not greater than zero; (c) a pixel-space predictor with the same protocol matching V-JEPA, which would locate the effect in prediction per se rather than in the representation space; (d) accuracy collapsing to chance under a fixed context (Fig. S3) or on the held-out private test set (Table S4). The paper's negative cases (solidity, collision, InfLevel gravity) are where the claim already does not hold.

## 10. Outline feedback (ch-vjepa2 claim card)
- "surprise = prediction error measured how": L1 between the predictor's output and the target encoder's features of the actual future frames (Eq. S2), per start frame; averaged over the video for pairs, max for single videos (Eq. S3). Decision = higher surprise is the impossible video.
- "over which horizon": M = (C+M) - C frames; window 16 or 32; stride 2; best C per property; with C=2 and window 16 that is 14 predicted frames (about 1.9 s at 7.5 fps on IntPhys). Our reading for the seconds.
- "headline accuracies vs pixel-prediction and VLM baselines (table rows)": there is no table; baseline bars are unlabelled. Show V-JEPA bars with the printed CIs and draw the baselines as a qualitative "near chance" band, or ask the user to fetch performance.csv from the repo.
- The model is V-JEPA (2024) ViT-H on VM2M with RoPE, not V-JEPA 2. Say so in the card; the paper predates V-JEPA 2 (Feb 2025).
- Name the MLLM baseline "Qwen2-VL-72B" (legend and tables); the main text's "7B" is inconsistent.
- Per-property result worth a sentence: solidity and collision are not above chance; the claim is about object-intrinsic properties.
