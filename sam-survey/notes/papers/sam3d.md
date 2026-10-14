# SAM 3D: 3Dfy Anything in Images

## 1. Identity
- Title: "SAM 3D: 3Dfy Anything in Images". arXiv 2511.16624, read v2 (LaTeX source + PDF of v2).
- Dates (abs.html): v1 submitted 2025-11-20; v2 (this version) revised 2026-06-02. The outline gives only the v1 date.
- Authors: "SAM 3D Team", Xingyu Chen (first named, core contributors alphabetical, equal contribution) ... Jitendra Malik (last). Project leads: Piotr Dollar, Georgia Gkioxari, Matt Feiszli, Jitendra Malik. Meta Superintelligence Labs (README).
- Venue: none stated.
- Repo: github.com/facebookresearch/sam-3d-objects. "SAM 3D Objects is one part of SAM 3D, a pair of models for object and human mesh reconstruction" (README); the other is SAM 3D Body (separate repo, not covered by this paper). README: checkpoints, demo, paper out 11/19/2025; encoder weights 06/01/2026; SA-3DAO set + HF leaderboard 06/02/2026 (code-only).

## 2. Core problem
Reconstruct full 3D shape, texture and layout (pose in camera frame) of any object from one natural image plus an object mask, including occluded, small and cluttered objects. The blocker is data: natural images paired with 3D ground truth do not exist at scale, and generalist annotators cannot build 3D meshes (Sec. 1, 3.2.1).

## 3. What prior methods did and why that hurt
- Single-image 3D generators (Trellis, Hunyuan3D) "are trained on isolated objects and struggle with objects in natural scenes, where they may be distant or heavily occluded" (Sec. 1).
- Pose/layout was done by pipelines (3D generator + MegaPose/FoundationPose render-and-compare) or by joint scene generators (MIDI) that are weak on real data (Table 3).
- Real-world benchmarks skew to indoor furniture or synthetic isolated renders (App. D).

## 4. The one key idea
Humans cannot generate 3D shapes, but they can select the best of N model proposals and align it to a point cloud. Use that as a model-in-the-loop (MITL) data engine, and train like an LLM: synthetic pretraining -> semi-synthetic mid-training -> real-world post-training (SFT + DPO), repeated as a flywheel (Sec. 1, 3).

## 5. Mechanism, step by step
1. Input: image I and binary mask M of one object (from SAM-family or any source; "SAM 3D is agnostic to the source of mask", App. E.9). Optional coarse scene pointmap P (LiDAR or monocular depth, e.g. MoGe) (Sec. 2.2).
2. Encoding: DINOv2 encodes two image/mask pairs -> 4 sets of conditioning tokens: cropped object image + cropped binary mask; full image + full-image binary mask (Sec. 2.2). DINOv2 size and input resolution: not stated.
3. Geometry model (1.2B params): flow transformer with Mixture-of-Transformers (MoT). Models p(O, R, t, s | I, M). O = 64^3 coarse voxel shape, denoised in a latent of 16^3 x 8 (Trellis structured latent) -> 4096 shape tokens; R = 6D rotation, t in R^3, s in R^3, denoised directly in parameter space -> 1 layout token. Modality-specific input/output projections to width 1024. Two transformers: one for shape tokens, one for layout; they share information only through joint self-attention under a structured attention mask (Fig. 2, App. C.1). The mask lets one modality be trained or frozen alone (e.g. layout-only finetune).
4. Texture & Refinement model (600M params): sparse latent flow transformer over the active voxels of O; refines geometry and synthesizes texture; models p(S, T | I, M, O) (Sec. 2.2).
5. 3D decoders: two VAE decoders, mesh D_m and 3D Gaussian splats D_g, sharing one VAE encoder / structured latent (SLAT). Improved "Depth-VAE" back-projects DINOv2 features only to voxels visible in each view (App. C.6). Up to 32 splats per occupied voxel (App. F).
6. Coordinate frame: shape, rotation, translation, scale "in camera coordinates" (Sec. 2.1); mid-training predicts translation and scale "in normalized camera coordinates" (Sec. 3.1.2).
7. Inference: 25 flow-matching steps (CFG on first half -> NFE = 1.5x steps); shortcut distillation of the Geometry model to 4 steps (NFE 25 -> 4) for "sub-second shape and layout" (Sec. 3.2.2, App. E.8). Texture model not distilled for release.
8. Optional test-time render-and-compare refinement of layout (App. E.3, Table 9).

Training (which weights get gradient from what):
- Pretraining: Iso-3DO, 2.7M meshes (Objaverse-XL + licensed) rendered from 24 views, object-centric crop input, modalities S, R; 2.5T tokens; loss Eq. 1 (Sec. 3.1.1, Table 1, Table 5).
- Mid-training: RP-3DO render-paste, 61M samples / 2.8M unique meshes. Flying Occlusions (FO, 55.1M samples, full image, S,R) then Object Swap-Random (OS-R, 5.95M samples) + ProcThor with pointmap, adds t, s. Skills: mask-following, occlusion robustness, layout. 2.7T tokens (Sec. 3.1.2; Table 5 lists 2.4T + 0.3T).
- Post-training (Sec. 3.2): iterate collect -> improve. SFT first on noisy MITL-3DO, then on artist Art-3DO; then DPO on Stage-2 D+/D- pairs (shape only for the Geometry model; layout not aligned, Table 1 "Alignment: S"); then shortcut distillation. Final post-training iteration 0.5T tokens; final SFT keeps 500K samples above alpha_K.
- Texture & Refinement trained with an analogous recipe (Trellis500K / Iso-3DO-500K pretraining, RP-3DO FO+OS-A mid-training, MITL SFT, DPO) (App. C.5, Table 6).

Data engine (Sec. 3.2.1, App. A):
- Stage 1, choose targets (I, M): images from SA-1B, MetaCLIP, SA-V, Ego4D, Ego-Exo4D, AEA, AEO, Nymeria, Food, BDD100k; masks from pre-existing annotations (SA-1B cited) and human labelers; a 3D-oriented taxonomy merged from LVIS's 1,200 categories; curriculum simple -> complex -> deformable. 850,000 object instances from 360,000 images. 10 s per object, "We utilize SAM as a tool to assist in segmentation."
- Stage 2, rank and select (S, T): candidates from retrieval, text-to-3D, image-to-3D (incl. SAM 3D checkpoints); annotators do sequential pairwise comparisons over N candidates (main text N=8; App. A.5 says 6-10), then rate against a bar alpha. Below-bar picks become DPO negatives or Art-3DO candidates. 80 s per object. Annotators "cannot edit".
- Stage 2.5, artists (Art-3DO): a small percentage of hardest failures (complex geometry, occlusion, transparency, small objects) meshed by professional 3D artists.
- Stage 3, align to 2.5D scene (R, t, s): annotators rotate/translate/scale the mesh against a point cloud from an off-the-shelf depth estimator. 150 s per object.
- Cold start: early candidates mostly from an ensemble of existing models; the best SAM 3D model eventually produces "about 80%" of annotated data.
- Totals: "almost 1 million images with ~3.14 million untextured meshes and ~100K textured meshes" (Sec. 3.2.1); lifetime yield 3.14M trainable shapes, 1.23M layout samples, ~100K trainable textures, over 7M pairwise preferences (App. A.5).
- Reward-model best-of-N: 50 seeds, reward model tournament, winner to humans; recovers discarded hard inputs: yield 0% -> 86.8%; food share 4% -> 36% (9x) (App. A.7.1).

## 6. Equations
Custom macros: none needed beyond standard (\method = "SAM 3D").

Eq. 1 (label eq:fm, App. C.2), multi-modal conditional rectified flow matching:
```
\mathcal{L}_{\text{CFM}} = \sum_{m \in \mathcal{M}} \lambda_m \underset{\raisebox{-1.3ex}{$\scriptstyle \tau, \mathbf{x}_0^m$}}{\text{\Large$\mathbb{E}$}} \left[ \|\mathbf{v}^m - \mathbf{v}_\theta^m(\mathbf{x}_\tau^m, c, \tau)\|^2 \right]
```
with `\mathbf{x}_\tau^m = \tau\mathbf{x}_1^m + (1-\tau)\mathbf{x}_0^m` and `\mathbf{v}^m = \dot{\mathbf{x}}_\tau^m =(\mathbf{x}_1^m{-}\mathbf{x}_0^m)`.
- Paper: "the Geometry model optimizes the following multi-modal flow matching objective".
- M = {S, R, t, s} modalities; lambda_m per-modality weight (Table 5: S=1.0, R=0.1, t=1.0, s=0.1); c = (I, M) condition; tau in [0,1] time; x_0^m Gaussian noise; x_1^m clean ground truth; x_tau^m straight-line mix; v^m = x_1 - x_0 target velocity; v_theta^m network-predicted velocity.
- Paper notes flow matching alone suffices, "without explicitly enforcing geometric constraints through loss objectives".

Eq. 2-3 (label eq:dpo, App. C.3), Diffusion-DPO adapted to flow matching:
```
\mathcal{L}_\mathrm{DPO} = - \mathbb{E}_{I \sim \mathcal{I}, (x_0^w,x_0^l) \sim \mathcal{X}_I^2, \tau \sim \mathcal{U}(0,T), x_\tau^w \sim q(x_\tau^w\mid x_0^w), x_\tau^l \sim q(x_\tau^l\mid x_0^l)} \big[\operatorname{\log \sigma}\big(-\beta T w(\tau) \cdot \Delta\big) \big]
\Delta = \|\mathbf{v}^w - \mathbf{v}_\theta(x_\tau^w,c,\tau)\|_2^2 - \|\mathbf{v}^w - \mathbf{v}_{\mathrm{ref}}(x_\tau^w,c,\tau)\|_2^2 - \big( \|\mathbf{v}^l - \mathbf{v}_\theta(x_\tau^l,c,\tau)\|_2^2 - \|\mathbf{v}^l - \mathbf{v}_{\mathrm{ref}}(x_\tau^l,c,\tau)\|_2^2 \big)
```
(subscript array flattened onto one line; source uses an `array` under \mathbb{E}.)
- x_0^w / x_0^l: preferred / rejected 3D outputs for the same c. NOTE: here x_0 is the clean sample (Diffusion-DPO convention), the opposite of Eq. 1 where x_0 is noise. v_ref: frozen reference (SFT model, Alg. 1); beta, w(tau): temperature and time weight (values not stated). Delta < 0 when the trained model fits the winner better (relative to ref) than the loser.
- Negatives from non-SAM-3D sources (retrieval, multi-view texture) are removed as out of distribution.
- PDF quirk: App. C.5 says "same training objective described in Equation (4)", but (4) in the PDF is the shortcut loss; the DPO equation prints as (2)-(3).

Eq. 4 (App. C.4), shortcut distillation (Frans et al.):
```
\mathcal{L}_S(\theta) = \mathbb{E}_{x_0\sim\mathcal{N}(0,I), x_1\sim p(x), \tau,d\sim p(\tau,d)} \Big[ \left\| \mathbf{v} - \mathbf{v}_\theta(x_\tau, c, \tau, d{=}0) \right\|^2 + \left\| \mathbf{v}_{\mathrm{consistency}} - \mathbf{v}_\theta(x_\tau, c, \tau, 2d) \right\|^2 \Big]
```
- d = step size (d=0 is plain flow matching); v_consistency = stop-grad average velocity of two CFG-guided steps of size d (Alg. 2). CFG distilled in (w_CFG=2 Geometry, 1 Texture). ~4K finetune iterations, 75% flow matching / 25% shortcut. Both modes kept in one model.

## 7. Config / setup
| Item | Value | Loc |
|---|---|---|
| Image encoder | DINOv2 (size, resolution not stated) | Sec. 2.2 |
| Conditioning | crop image + crop mask, full image + full mask (4 token sets); optional pointmap | Sec. 2.2 |
| Geometry model | 1.2B flow transformer, MoT (2 transformers) | Sec. 2.2 |
| Shape O | 64^3 voxels, latent 16^3 x 8, 4096 tokens | App. C.1 |
| Layout | R 6D, t R^3, s R^3; 1 token | Sec. 2.2, App. C.1 |
| Token width | 1024 | App. C.1 |
| Texture & Refinement | 600M sparse latent flow transformer | Sec. 2.2 |
| Decoders | mesh + Gaussian splats VAE decoders, shared encoder; up to 32 splats/voxel | Sec. 2.2, App. F |
| Encoder trainable/frozen | not stated; README ships separate encoder weights (code-only) | Sec. 2.2; README |
| Decoders training | VAE decoders "separately-trained", share one encoder | Sec. 2.2 |
| DPO reference | frozen SFT checkpoint pi_k^SFT | Alg. 1 |
| Optimizer | AdamW, no weight decay | App. C.2 |
| Geometry LR | 1e-4 pre/mid, 1e-5 SFT, 2.5e-6 alignment | Table 5 |
| Geometry batch | 6 per GPU | App. C.7 |
| Geometry compute | pretrain 512 A100, 200 epochs; FO mid 320 A100 50 ep + 128 A100 50 ep; OS-R 256 A100 12 ep; SFT 128 H200 100 ep; DPO 128 A100 1 ep | App. C.7 |
| Texture compute | pre 256 A100 245 ep bs4; mid 256 A100 80 ep; SFT 192 A100 89 ep; DPO 128 A100 2 ep bs3 | App. C.7 |
| Inference | 25 steps (NFE 1.5x) -> 4 shortcut steps; 1-step 38x, 4-step 10x faster | App. E.8 |

Table 5 (Geometry) per stage: Pretraining Iso-3DO 2.7M meshes / 64.8M images / 64.8M samples / 2.5T tokens; Mid FO 2.87M / 11.17M / 55.1M / 2.4T; Mid OS-R+ProcThor 2.38M / 1.20M / 5.95M / 0.3T; SFT MITL-3DO+Art-3DO 0.6M / 0.5M / 0.6M / 0.9T; Alignment 88K / 31K / 44K / -.
Table 6 (Texture): Pre Trellis500K 350K / 9M / 10M / 1.1T; Mid RP-3DO (FO,OS-A) 800K / 2.4M / 2.4M / 1T; SFT MITL ~100K / 100K / 100K / 115B; Alignment 146K / 73K / 73K / -.
Differs: main text says the final post-training iteration uses 0.5T tokens; Table 5 SFT row says 0.9T (our reading: final iteration vs cumulative across iterations).

## 8. Results worth remembering
- Abstract: "at least a 5:1 win rate in human preference tests on real-world objects and scenes". Sec. 4.1: objects "5:1 head-to-head win rate"; scenes "users prefer scene reconstructions from SAM 3D by 6:1 over prior SOTA".
- Fig. 8 (preference) prints NO numbers despite the caption ("Numbers indicate human preference rates"). Bars only: Full Scene (Scene Pref Set, LVIS, ADT) vs HY3D-2.0/FP, Trellis/MP, MIDI; Objects (SA-3DAO, Object Pref Set, LVIS, ISO3D) vs HY3D-2.0/FP, Trellis/MP. SAM 3D share visibly ~70-90%; exact per-benchmark values are not given. Do not print per-benchmark numbers.
- Table 2 (shape, SA-3DAO F1@0.01 / vIoU / Chamfer / EMD; ISO3D ULIP / Uni3D): SAM 3D 0.2344 / 0.2311 / 0.0400 / 0.1211; 0.1488 / 0.3707. Best baseline on SA-3DAO F1: Hi3DGen 0.1629; Trellis 0.1475 / 0.1392 / 0.0902 / 0.2131. ISO3D ULIP best is TripoSG 0.1529 (SAM 3D not best there).
- Table 3 (layout, joint generation): SAM 3D SA-3DAO 3D IoU 0.4254, ICP-Rot 20.7667, ADD-S 0.2661, ADD-S@0.1 0.7232; ADT 0.4970 / 15.2515 / 0.0765 / 0.7673. MIDI on ADT ADD-S@0.1 0.0175. Text: "ADD-S @ 0.1 2% -> 77%" (MIDI -> SAM 3D on ADT, our reading of which rows). Best pipeline on SA-3DAO ADD-S@0.1: HY3D-2.0 + FoundationPose 0.5396.
- Table 4 (cascade, SA-3DAO F1@0.01): Pre-training 0.1349 -> +Mid 0.1705 -> +SFT MITL 0.2027 -> +DPO MITL 0.2156 -> +SFT Art 0.2331 -> +DPO Art 0.2344. Texture WR vs row above: 60.7, 66.9, 66.4.
- Table 7 knockout (F1): full 0.2344; w/o MITL-3DO 0.2211; w/o Art-3DO 0.2027; w/o DPO on MITL 0.2156.
- Table 8 texture WR (SAM 3D shape given to all): vs Trellis 81.1 (ISO3D) / 87.0 (Pref) / 86.2 (SA-3DAO) / 89.1 (LVIS); vs Unitex 83.3/84.7/84.5/88.3. Note: Trellis and Hunyuan3D-2.1 rows are identical in three columns (87.0, 86.2, 89.1); possible copy error in the source.
- Fig. 10: Elo; "a 400 point Elo difference corresponds to 10:1 odds"; checkpoints ~3 weeks apart; near-linear Elo scaling (no numbers extracted).
- SA-3DAO: 1,000 untextured artist-made meshes paired with natural images; artist time "up to 5 minutes" to "over 5 hours" per mesh; median 4751 vertices (App. D.1).
- Pref sets: SA-1B 1,000, MetaCLIP 1,000 (occluded), LVIS 1,000 (10-30 objects/scene), ADT ~40 frames (App. D.2).
- Rotation (Table 10): ICP-Rot quaternion 17.9585 -> 6D 15.5399 -> normalized 6D 14.5946.
- Pointmap vs none: shape preference 48% each on LVIS (App. E.5).
- Test-time layout optimization on ADT (Table 9): 3D IoU 0.4837 -> 0.5258; accepted for 554 of 1027 instances.
- Reward models: VLM 68.9% agreement with humans, DPO ~65%, two humans <75% (App. E.7).

## 9. Limitations (App. F, quoted)
- "The geometry model uses a coarse shape resolution of O in R^{64^3}; we trained multiple 3D Gaussian splat decoders, with up to 32 splats per occupied voxel." -> distortions on complex shapes, thin structures, hands/faces of whole bodies.
- "SAM 3D predicts objects one at a time, and isn't trained to reason about physical interactions, such as contact, physical stability, interpenetration, or co-alignment".
- "texture predictions are made without knowledge of the predicted object's pose" -> wrong texture orientation on rotationally symmetric objects.
- Mask errors compound from the segmenter, but "fair robustness" to over/under-segmentation (Fig. 20).
- No later paper in the given sources fixes these.

Connection to SAM: "SAM 3D assumes a segmentation mask as input ... the ready availability of segmentation models like the SAM series [SAM, SAM 2, SAM 3] offer a fast, promptable solution" (App. E.9). SAM is not part of the model; the mask is an external input. SAM assists Stage 1 annotation; SA-1B supplies images, masks and a preference set. The data engine is framed against SAM's: "most people cannot create meshes directly ... This is different from the segmentation masks collected in SAM" (Sec. 3.2.1).

## 10. Outline feedback
- Add v2 date 2026-06-02 (outline lists only v1).
- "a SAM mask": correct reading is "a mask, e.g. from SAM"; the model is mask-source agnostic. Say so.
- "coarse voxels + layout": exact = 64^3 voxels (latent 16^3x8, 4096 tokens) + one layout token (6D R, t, s), 1.2B MoT; texture model 600M; outputs mesh or Gaussian splats.
- "preference optimization" = DPO (Diffusion-DPO adapted to flow matching), applied to shape (and texture), not layout.
- Visual: rotation shown should be 6D (two columns of a rotation matrix), scale is 3D (anisotropic s in R^3), not a single scalar.
- Do not show per-benchmark preference percentages: Fig. 8 prints none. Use "5:1" (objects) and "6:1" (scenes).
- Data engine is a natural parallel to SAM's ch-engine: assisted stages, then model produces ~80% of data; consider one sentence linking back.
