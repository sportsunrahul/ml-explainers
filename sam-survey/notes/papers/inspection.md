# Inspection and failure-analysis papers (chapter ch-inspection)

Scope: eight papers for ch-inspection. Lens: analysis (claim, measurement setup, result, what would falsify it).
Each paper keeps the brief's 10 headings; qualitative papers merge headings where a section would be empty.
Numbers are as printed. Delta columns in Not Always Perfect and Concealed are absolute metric differences printed with a "%" sign.
"our reading" marks my inference. All dates come from abs.html.

Quick map (which paper measures what):

```
paper            data                         SAM prompt            metric            SAM alone measured?
---------------  ---------------------------  --------------------  ----------------  --------------------------
Industrial SAM   aircraft mfg/MRO photos      points, grid, box,    none (qualit.)    qualitative only
                 (own, unreleased)            text (GroundedSAM)
DefectSAM        IR thermography, CFRP        expert boxes          F2-score, IoU     yes (Tab. 2, 3)
                 (own, 218 sequences)
SAA / SAA+       VisA, MVTec-AD, KSDD2, MTD   text -> GDINO boxes   max-F1 pixel /    no (SAA = GDINO+SAM)
                                              -> SAM                region
KairosAD         MVTec-AD, VisA               none (encoder only)   image AUROC       no (MobileSAM encoder
                                                                                      + MLP, supervised)
Boxes2Pixels     DTU wind-turbine (YOLO       GT boxes -> SAM       mIoU, IoU_bin,    no (SAM is the teacher;
                 boxes), manual test masks    pseudo-masks          F1, recall        teacher IoU not given)
Not Always Perf. 8 benchmarks incl. CDS2K     everything + oracle   MAE               yes, oracle upper bound
Concealed        CAMO, COD10K, NC4K (+qual.   everything + oracle   S, E, F, MAE      yes, oracle upper bound
                 KolektorSDD, MagneticTile,
                 MVTec)
SAM-Adapter      COD10K/CAMO/CHAMELEON,       whole-image box       S, E, F, MAE,     yes (weak protocol)
                 ISTD, Kvasir                                       BER, Dice/IoU
```

---

## 1. Industrial Segment Anything (industrialsam)

1. **Identity.** "Industrial Segment Anything -- a Case Study in Aircraft Manufacturing, Intralogistics, Maintenance, Repair, and Overhaul". arXiv 2307.12674v1, submitted 2023-07-24 (only version). Authors: Keno Moenck (first) ... Daniel Schoepflin (last in the abs.html list; 12 authors). No venue stated. No repo.
2. **Core problem.** Aircraft production has few datasets for non-everyday objects. The paper asks where SAM's zero-shot masks help across manufacturing, intralogistics and MRO use cases.
3. **Prior methods.** Supervised models (e.g. U-Net) need task-specific training data and high-quality annotations that do not exist for borescope or WLI data (Sec. 6.1.2).
4. **Key idea.** SAM's masks have no semantics. Domain knowledge must come from a human prompt (human-in-the-loop) or from pipeline logic (prompt placement plus mask selection).
5. **Mechanism (prompt taxonomy, Sec. 2).** P.1 = automatic grid of query points. P.2 = single foreground point. P.3 = foreground and background points. P.4 = box. P.5 = text via GroundedSAM. NMS removes overlapping masks. The default is ViT-H with the default automatic-mask-generator parameters.
6. **Equations.** None.
7. **Measurement setup.** **Qualitative case study.** No benchmark, no metric, no timing study. Figures show SAM outputs on the authors' own images.
8. **Concrete findings (all qualitative).**
   - Cabin QA surface defects with one P.2 click at the defect centre (Fig. 6): the dent mask is accurate; the dirt/contamination mask is wrong, because "major areas without dirt were also added"; for the indentation, the first two mask levels cover the defect. All three defects are small and low-contrast against the part (Sec. 4.2).
   - Borescope engine images, five defect classes (corrosion, TBC missing, crack, dent, missing material): "only 1-3 cursor inputs by the user are required to create the segmentation map with pixel accuracy" (Sec. 6.1.1, Fig. 12, P.3 prompts).
   - WLI surface data: SAM separates noise from surface, selects anomalies and segments a scratch "with just a few clicks" (Fig. 13).
   - Foam inlay with metal chips (automatic mode): default parameters do worse on the chips than a tuned set of 64 points per side, 0.7 pIoU, 0.2 NMS threshold, 0.5 stability score threshold and 250 minimum mask region area (Fig. 8 caption).
   - Close-ups (WLI, point-cloud projections): the highest-predicted-IoU mask is meaningful and the other two are "mostly cluttered" (Sec. 7).
   - Counting with the method of Ma et al. is "not sufficient enough for a real-world application" in their domain (Sec. 7.1).
   - Labeling time: the paper says point prompting is "possibly leading to a significant time advantage" over polygons. It gives **no number** and says "robustness has yet to be proven by broader studies" (Sec. 4.2).
9. **Limitations stated.** No quantification. Domain robustness "definitely needs further investigation ... based on domain-related benchmark datasets" (Sec. 7). Mask-level choice "requires domain and context knowledge". Usability with shop-floor staff is not evaluated.
10. **Outline feedback.** This is point-prompt labeling evidence, and it has no numbers. Do not group it with Boxes2Pixels as "box prompts". Use it for the human-in-the-loop argument and the dirt failure.

**What would falsify it:** a timed labeling study where SAM clicks are not faster than polygons at equal mask quality. No such study exists in these sources.

---

## 2. Segment Anything in Defect Detection / DefectSAM (samdefect)

1. **Identity.** "Segment Anything in Defect Detection". arXiv 2311.10245v1, submitted 2023-11-17 (only version). Authors: Bozhen Hu (first) ... Stan Z. Li (last). CVPR-template preprint; no venue stated.
2. **Core problem.** Defect segmentation in infrared thermal images from optical pulsed thermography (OPT and portable POPT), mostly on CFRP specimens. The images have low resolution, high noise and uneven heating (Abstract, Sec. 1). **This is infrared non-destructive testing, not RGB surface inspection.**
3. **Prior methods.** Supervised and semi-supervised models overfit thermal images, because the images are similar across the whole dataset (Sec. 1).
4. **Key idea.** Fine-tune SAM on a curated thermal defect dataset and prompt it with expert boxes.
5. **Mechanism.** (1) Remove the first L0 and last Lf frames of each thermal sequence and sample every L_I-th frame. (2) Subtract the mean of the last Lf frames (the residual heat). (3) Resize to 1024x1024. (4) Fine-tune the SAM image encoder and mask decoder ("mask encoder" in the text). Keep the prompt encoder weights at their original values. (5) Use box prompts (Fig. 5).
6. **Equations** (sec/3_Method.tex):
   - Frame count: `N_f(f_i)=B\frac{f_i-L_0-L_f}{L_I}` -- B sequences, f_i frames each.
   - Residual-heat removal: `x_{ij}=D_{ij}- \frac{{\textstyle \sum_{k=f_i-L_f+1}^{f_i}}D_{ik}}{L_f}`.
   - Loss: `L = L_{\operatorname{BCE}}+\beta L_{\operatorname{Dice}}` with beta = 1.
   - F-score: `F = \left(\gamma^2+1\right)(P \times R) /\left(\left(\gamma^2 \times P\right)+R\right)` with **gamma = 2** (recall weighted, Sec. 3.4).
7. **Measurement setup.** B = 218 volumetric sequences. L0 = 15, Lf = 15, L_I = 5. More than 6000 instances. Split 80/10/10 at random; five-fold cross-validation. AdamW, LR 1e-4, weight decay 0.01, batch 4, one A100 (Sec. 4.1). Baselines: TernausNet, H-DenseUNet, RVOS, UNet++, 3D-UNet++, ASGNet, Xue et al., SAM, MedSAM. Several baseline numbers are copied from earlier papers (Table 2, * and dagger marks). The SAM and MedSAM variants are not stated.
   - **our reading (leakage risk):** "the segmentation ground truths are the same for each D_i", and the split is random over instances. Frames from one sequence can therefore appear in both train and test.
8. **Results.**
   - Table 2 (F-score, %, gamma = 2), Flat-type / R-type columns: SAM 98.24 / 68.02. MedSAM 100.0 / 100.0. DefectSAM 100.0 / 100.0. 3D-UNet++ 93.33 / 88.70.
   - Table 3 (IoU), Flat / R: SAM 49.32 / 40.44. MedSAM 66.56 / 65.75. DefectSAM 78.58 / 81.98. 3D-UNet++ 52.91 / 48.07.
   - The "Improved (%)" row reads 18.05 / 24.68. These are **relative gains over MedSAM**, not over SAM: (78.58-66.56)/66.56 is about 18.06% and (81.98-65.75)/65.75 is about 24.68% (our arithmetic).
   - Table 1 (qualitative): SAM misses defects in image No. (6), which is R-type. MedSAM makes rectangular defects round.
   - Key point: zero-shot box-prompted SAM finds almost all defects (F2 98.24 on flat parts), but its mask extent is poor (IoU 49.32). Fine-tuning fixes the extent.
9. **Limitations.** Defect depth, their problem (d), is left to future work. Labels are uncertain because of thermal diffusion (Fig. 6).
10. **Outline feedback.** Present it as the "fine-tune SAM on your own modality" data point, and label it thermography. Do not cite the F = 100 rows without the gamma = 2 and box-prompt caveats.

**What would falsify it:** a sequence-level split where DefectSAM's IoU gain over MedSAM disappears.

---

## 3. Segment Any Anomaly without Training via Hybrid Prompt Regularization (saa)

1. **Identity.** arXiv 2305.10724v1, submitted 2023-05-18. Authors: Yunkang Cao (first, equal contribution) ... Weiming Shen (last). abs.html journal-ref: IEEE Transactions on Cybernetics, 2025. Repo: github.com/caoyunkang/Segment-Any-Anomaly.
2. **Core problem.** Zero-shot anomaly segmentation (ZSAS): the model sees no normal and no anomalous training images of the target category (Sec. 3.1).
3. **Prior methods.** Unsupervised methods (reconstruction, one-class, memory bank) train one model per category. WinCLIP segments by text-visual similarity.
4. **Key idea.** Cascade GroundingDINO (text -> boxes) and SAM (boxes -> masks). The plain cascade "SAA" with the prompt "anomaly" gives many false alarms: in Fig. 1 it marks **every** candle wick, while only the overlong wick is the anomaly. The paper calls this "language ambiguity". SAA+ regularizes the cascade with hybrid prompts and uses no training.
5. **Mechanism.**
   1. Generator: GroundingDINO(I, T) -> boxes R^B and scores S (Eq. 1).
   2. Refiner: SAM(I, R^B) -> masks R (Eq. 2).
   3. Language prompts P^L: class-agnostic T_a ("anomaly", "defect") plus class-specific T_s from expert knowledge ("black hole", "white bubble").
   4. Property prompts P^P: drop candidates whose IoU with the detected object is below theta_IoU, or whose area does not fit theta_area times the object area (Eq. 4).
   5. Saliency prompt P^S: a kNN feature-distance saliency map (WideResNet50, N = 400), averaged inside each mask and exponentiated, multiplies each score (Eqs. 5-7).
   6. Confidence prompt P^C: keep the top K = 5 regions and fuse them into an anomaly map (Eqs. 8-9).
6. **Equations** (arxiv.tex):
   - `\mathbf{s}_{ij} := \frac{1}{N}\sum\limits_{\mathbf{f}\in N_p(\mathbf{f}_{ij})}(1-  \langle \mathbf{f}_{ij},\mathbf{f}  \rangle)` (eq:saliency-map): the mean cosine distance from a pixel feature to its N nearest neighbours.
   - `\mathcal{P}^{S} := \left\{ \exp(\frac{\sum_{i j}\mathbf{r}_{i j}\mathbf{s}_{i j}}{\sum_{i j}\mathbf{r}_{i j}}) \quad | \quad \mathbf{r} \in \mathcal{R}^P \right\}` (eq:score_saliency).
   - `\mathcal{S}^{S} := \left\{ p \cdot s  \quad | \quad   p \in \mathcal{P}^{S},   s \in \mathcal{S}^P \right\}` (eq:rescore).
   - `\mathcal{R}^C, \mathcal{S}^C := \mathrm{Top}_K(\mathcal{R}^P,\mathcal{S}^S)` (eq:confidence-prompts).
   - `\mathbf{A}_{ij} := \frac{ \sum_{\mathbf{r}^C \in \mathcal{R}^C} \mathbf{r}^C_{i j} \cdot s^C}{ \sum_{\mathbf{r}^C \in \mathcal{R}^C} \mathbf{r}^C_{i j}}` (eq:fusion).
7. **Measurement setup.** Data: VisA, MVTec-AD, KSDD2 and MTD, with subsets grouped into texture and object. Metrics: max-F1-pixel (F_p), and max-F1-region (F_r), which this paper introduces: a region counts as positive if its overlap exceeds 0.6. Input 400x400. Baselines: WinCLIP (official numbers, VisA and MVTec only), ClipSeg, UTAD, and SAA. All methods are training-free, so variance is +-0.00. **No row measures SAM by itself.** The class-specific prompt lists and theta values are said to be "in the supplementary material", but v1 has no appendix (LaTeX and PDF checked).
8. **Results.**
   - Table 1 (the caption says "Qualitative comparisons", but the table is quantitative). F_p for SAA -> SAA+: VisA 12.76 -> 27.07. MVTec-AD 23.44 -> 39.40. KSDD2 8.79 -> 59.19. MTD 14.78 -> 35.40. Texture 20.94 -> 53.79. Object 17.35 -> 28.82. Total 18.22 -> 34.85. WinCLIP F_p: VisA 14.82, MVTec-AD 31.65.
   - Table 1, F_r for SAA -> SAA+: VisA 4.83 -> 14.46. MVTec-AD 32.49 -> 49.67. KSDD2 16.40 -> 39.34. MTD 10.63 -> 30.27. Total 19.74 -> 34.07.
   - Table 2 ablation (Total F_p): without P^L 30.95, without P^P 21.50, without P^S 30.96, without P^C 34.13, full 34.85. **The largest single effect is the property filter on textures: F_p 21.83 -> 53.79.** The language prompt adds +3.90 F_p and +4.90 F_r. The confidence prompt adds +0.72 F_p.
   - Fig. 5: performance peaks near K = 5.
   - Even SAA+ reaches only F_p 28.82 on object categories and 27.07 on VisA.
9. **Limitations stated.** They did not test larger foundation models.
10. **Outline feedback.** Fine as planned. Make the "segments objects, not anomalies" point with Fig. 1 (wick). Say that the main gain comes from rule-based filters (location, area, top-K) and image saliency, not from SAM.

**What would falsify it:** SAA+ hybrid prompts that do not beat SAA on a new dataset without per-dataset prompt tuning. The expert prompts are not published in v1, so this cannot be checked from the paper.

---

## 4. KairosAD (kairosad)

1. **Identity.** "KairosAD: A SAM-Based Model for Industrial Anomaly Detection on Embedded Devices". arXiv 2505.24334v1, submitted 2025-05-30. Authors: Uzair Khan (first), Franco Fummi, Luigi Capogrosso (last). ICIAP 2025. Repo: github.com/intelligolabs/KairosAD.
2. **Core problem.** SMEs need anomaly detection on low-cost embedded devices. SOTA models are too large for them.
3. **Prior methods.** PatchCore, SoftPatch and similar memory-bank methods need large memory and compute. STLM uses two branches.
4. **Key idea.** Use the frozen MobileSAM image encoder "as it is" and add a small MLP that outputs an image-level anomaly score.
5. **Mechanism.** (1) SFE: MobileSAM encoder f_theta(I) -> E, where the paper states d = 3. (2) ASP: N_l linear+ReLU layers -> score s in [0,1]. (3) Loss: weighted BCE, where w_i is the ratio of negatives to positives.
6. **Equations** (srcs/3_method.tex):
   - `E=f_{\theta{}}(I)\;,`
   - `s = g_{\phi}(E)\;,`
   - `\mathcal{L}_{\text{WBCE}} = - \frac{1}{N} \sum_{i=1}^{N} w_i \left[ y_{i} \log \sigma(s_{i}) + (1 - y_{i}) \log (1 - \sigma(s_{i})) \right]\;,`
7. **Measurement setup.** **Image-level classification (I-AUROC), not segmentation. Supervised:** the labels y_i include anomalous samples. Most compared methods (PatchCore, RD4AD, SimpleNet, ...) train on normal images only, so the comparison is not like-for-like (our reading). Competitor numbers are copied from STLM (MVTec) and GLAD (VisA). Training: MVTec 35 epochs with N_l = 2; VisA 50 epochs with N_l = 3; LR 1e-2, Adam, RTX 4090. **The hardware for the 5 ms latency is not stated.** Copied competitor latencies come from other papers' hardware.
8. **Results.**
   - Table 1 (MVTec-AD), Average I-AUROC: KairosAD 99.10. SimpleNet 99.57. RD++ 99.44. PatchCore 99.20. STLM 99.05. Latency (ms): KairosAD 5, STLM 20, SimpleNet 39, PatchCore 180. Params (M): KairosAD 11.53, STLM 16.56, SimpleNet 52.88, PatchCore 186.55.
   - Weak classes in Table 1: Metal nut 92.43 and Wood 96.49.
   - Table 2 (VisA), Average: KairosAD 92.3, which ranks fourth (RD4AD 96.9, SimpleNet 96.2, DRAEM 92.4). Capsules 69.7 is "a key limitation". Params are 10.55 M here, against 11.53 M in Table 1.
   - Embedded (Sec. 4.4): Jetson NX 211 ms and Jetson AGX 218 ms. The paper gives **no FPS**. The model was installed on the ICE Lab production line, with only qualitative results (Fig. 3).
   - Energy: 0.0465 kWh and 0.0154 kg CO2eq for training plus testing; about 270 W average GPU use in training (Sec. 1).
   - Internal inconsistencies:
     - The abstract's "78% fewer parameters and 4x faster ... compared to the leading SOTA" uses two comparators: SimpleNet for parameters (52.88 -> 11.53) and STLM for speed (20 -> 5 ms).
     - Fig. 1 says "35% fewer parameters" than STLM, but 16.56 -> 11.53 is about 30% (our arithmetic).
     - Hazelnut is 99.17 in the top half of Table 1 and 100 in the bottom half.
     - The VisA latency row gives PatchCore 37 and RD4AD 180, while the MVTec row gives 180 and 28.
9. **Limitations stated.** The VisA Capsules class. It does image-level (semantic) detection only, not fine-grained "sensory" anomalies such as minor scratches (Sec. 1).
10. **Outline feedback.** The outline says "edge deployment (KairosAD, efficient encoders)". Make it clear that KairosAD uses no SAM masks or prompts. It reuses a distilled SAM encoder as a feature extractor for OK/NG classification. Jetson latency is about 211-218 ms per image.

**What would falsify it:** an unsupervised baseline given the same anomalous labels, or measured on the same hardware, that removes KairosAD's speed or accuracy advantage.

---

## 5. Boxes2Pixels (boxes2pixels)

1. **Identity.** "Boxes2Pixels: Learning Defect Segmentation from Noisy SAM Masks". arXiv 2604.11162v1, submitted 2026-04-13. Authors: Camile Lendering (first) ... Egor Bondarev (last). AI4RWC Workshop at CVPR 2026. Repo: github.com/CLendering/Boxes2Pixels.
2. **Core problem.** Pixel-level defect masks are rarely available. A common workaround converts boxes into masks with SAM, but SAM pseudo-masks on industrial surfaces are "systematically noisy" (Abstract).
3. **Prior methods.** U-Net and SegFormer trained on SAM pseudo-masks "overfit the systematic errors" and inherit the teacher's failure modes (Sec. 1).
4. **Key idea.** Treat SAM as a noisy teacher, offline only. Train a compact student on frozen DINOv2 features with a binary localization head and a one-sided self-correction that relabels confident background pixels as defects.
5. **Mechanism.**
   1. For each GT box, SAM produces one mask from the box alone (no points). The masks are rasterized into a (K+1)-class label image (Dirt and Damage -> classes 1 and 2).
   2. Student: DINOv2 ViT-S/14, frozen except BitFit (biases and norms). Layers {1,2,4,7} feed a top-down residual fusion, then PixelShuffle to H/4.
   3. A CNN detail branch (two stride-2 conv blocks) runs at H/4. A concat + 3x3 mixer feeds a binary head (2 channels) and a fine head (K+1 channels).
   4. Losses: asymmetric Dice on the binary head (beta = 0.4) and class-weighted CE on self-corrected targets (tau = 0.9, after a warm-up).
   5. Training: input 518x518, AdamW, LR 5e-4, cosine schedule, weight decay 1e-2, gradient clipping at 1.0, EMA 0.999, lambda_bin = lambda_fine = 0.5.
6. **Equations** (sec/3_method.tex):
   - `\mathcal{L}_{\text{bin}} = 1 - \frac{\langle p, g \rangle + \epsilon} {\langle p, g \rangle + \beta \langle p, 1-g \rangle + \langle 1-p, g \rangle + \epsilon},` (eq:asym_dice_clean). beta < 1 down-weights false positives and biases training toward recall.
   - `\tilde{y}_i^{\text{corr}} = \begin{cases} \arg\max_{c>0} p_{i,c}, & \text{if } \tilde{y}_i = 0 \ \land\ \max_{c>0} p_{i,c} > \tau, \\ \tilde{y}_i, & \text{otherwise}, \end{cases}` (eq:self_correction). Only background labels can change.
   - `\mathcal{L}_{\text{total}} = \lambda_{\text{bin}} \mathcal{L}_{\text{bin}} + \lambda_{\text{fine}} \mathcal{L}_{\text{fine}},` (eq:total_loss).
7. **Measurement setup.** Data: DTU drone images of wind turbines in the YOLO-box release (Foster et al.), about 13,000 RGB images of 586x371, with two classes (damage, surface dirt). Fewer than 25% of images contain defects, and defects usually cover less than 5% of the image. All models train only on SAM pseudo-masks from boxes. **Test:** a held-out split with manual pixel masks (size not stated). **Validation and ablation (Table 2):** measured against SAM pseudo-masks, so they show agreement with the teacher. The SAM variant is not stated. **SAM's pseudo-mask quality against the manual GT is never reported**, so "noisy teacher" rests on qualitative evidence (Figs. 1 and 3). Hardware for timing: NVIDIA H100.
8. **Results.**
   - Table 1 (test, manual GT), mIoU / mIoU_anom / F1_anom / IoU_bin: U-Net 0.7057 / 0.5629 / 0.7036 / 0.5427. DeepLabV3-B2 0.6867 / 0.5342 / 0.6955 / 0.5331. SegFormer-B2 0.7231 / 0.5881 / 0.6939 / 0.5312. **Boxes2Pixels 0.7661 / 0.6523 / 0.7674 / 0.6226.** Text: +0.0642 mIoU_anom and +0.0914 IoU_bin over SegFormer-B2.
   - Table 3 (test), without -> with self-correction: mIoU_anom 0.5826 -> 0.6523 (+0.0697). F1_anom 0.6889 -> 0.7674 (+0.0785). IoU_bin 0.5255 -> 0.6226 (+0.0971). **Recall_bin 0.6195 -> 0.8051 (+0.1856).**
   - **Inconsistency:** the abstract says "+6.97 anomaly mIoU and +9.71 binary IoU over the strongest baseline". Those deltas are Table 3's self-correction ablation. Table 1 and its text give +0.0642 / +0.0914 over SegFormer-B2.
   - Table 2 (validation vs pseudo-labels), mIoU_anom: full 0.5093. Without the binary head 0.3851 (the largest drop). Without the detail branch 0.5013. Without self-correction 0.5049.
   - Table 4 (per-class IoU): Damage 71.65 vs SegFormer-B2 71.48 (about equal). Dirt 58.81 vs 46.13.
   - Table 5 (H100): Boxes2Pixels has 27.6 M total and 5.6 M trainable parameters, 43.37 GFLOPs, 6.20 ms and 161.4 FPS. SegFormer-B2: 27.3 M, 58.97 GFLOPs, 13.33 ms, 75.0 FPS. DeepLabV3-B2: 20.5 M, 18.92 GFLOPs, 7.99 ms, 125.1 FPS. The paper calls 5.6 vs 27.3 trainable "80% fewer".
   - Named SAM failure modes on industrial imagery (Sec. 3.2, 5.7): **false positives from shadows, stains and background texture; false negatives on thin cracks and low-contrast defects inside the box**; SAM masks are "often conservative". Boxes can also miss subtle damage entirely.
9. **Limitations stated.** One domain (wind turbines). Future work: other domains and uncertainty-aware self-correction.
10. **Outline feedback.** Boxes2Pixels is the only measured box -> SAM -> train pipeline in this set. Put it at the centre of the label-acceleration claim. Note that the gain comes from training a student that does not trust SAM, not from SAM masks themselves.

**What would falsify it:** a manual-GT comparison where SAM pseudo-masks used directly score at least as well as the student. That comparison is absent.

---

## 6. Segment Anything Is Not Always Perfect (notperfect)

1. **Identity.** "Segment Anything Is Not Always Perfect: An Investigation of SAM on Different Real-world Applications". arXiv 2304.05750, **v3 read**. v1 2023-04-12, v2 2023-04-13, v3 2023-05-22. Authors: Wei Ji (first) ... Li Cheng (last). Comments: CVPRW Oral. Journal-ref: Machine Intelligence Research, 2024. Repo: github.com/LiuTingWed/SAM-Not-Perfect.
2. **Core problem.** How does SAM do across natural images, agriculture, manufacturing, remote sensing and healthcare?
3. **Prior methods.** Not applicable (an evaluation paper).
4. **Key idea.** Probe SAM with click (SAM^1), box (SAM^2) and everything (SAM^3) modes. Then compare everything-mode SAM with dedicated SOTA on 8 benchmarks.
5. **Mechanism (quantitative protocol, Sec. 3.2).** Run SAM everything mode to get N masks. **Pick the mask with the highest IoU against the ground truth (oracle selection, following Tang et al.).** Compute MAE.
6. **Equations.** None. Metric: MAE.
7. **Measurement setup.** Benchmarks: DUTS, COME15K-Diff (low contrast), VT1000 (low light), DIS-TE4 (fine detail), COD10K (camouflage), SBU (shadow), **CDS2K (concealed industrial defects: 2,492 samples, 1,330 positive and 1,162 negative)**, ColonDB. Backbones: ViT-B, ViT-L, ViT-H. Qualitative figures use the online demo with human prompts; the prompts are not claimed to be optimal (footnote).
8. **Results.**
   - Table 1(g), CDS2K MAE (lower is better): SAM ViT-B 0.372, ViT-L 0.281, ViT-H 0.265. DGNet (best SOTA) 0.089, SINetV2 0.102, HitNet 0.118. The text calls ViT-H "17.6% worse": this is 0.265 - 0.089 = 0.176 MAE points.
   - Table 1(f), SBU shadow: ViT-H 0.183 vs MTMT 0.029. Table 1(e), COD10K: ViT-H 0.054 vs ZoomNet 0.029. Table 1(d), DIS-TE4: ViT-L and ViT-H both 0.166 vs IS-Net 0.072. Table 1(a), DUTS: ViT-H 0.046 vs EDNet 0.035.
   - **CDS2K (industrial) has the largest gap of all 8 benchmarks.**
   - Encoder scale helps a lot from B to L and little from L to H. For example, DIS-TE4 does not change from L to H.
   - Qualitative manufacturing: the MVTec AD figure (Fig. 8) "shows that it has powerful recognition ability". For surface defects on wood (Fig. 9, He et al. 2019 data), click mode is "acceptable" and everything mode "does not generate any results on these cases".
   - Named failures: low contrast (camouflage, transparent objects), shadows even with many clicks, fine detail, small or irregular objects, and professional data in box and everything modes (Sec. 1 bullets).
9. **Limitations.** The oracle selection makes SAM's numbers an upper bound for unprompted use. MAE only. The paper's Sec. 4 suggests application-specific SAM and data, multi-modal input (depth, thermal), and SAM pseudo-labels for semi-supervised segmentation.
10. **Outline feedback.** For industry, use the CDS2K row with the oracle caveat, plus Fig. 9: "everything mode gives nothing on wood defects".

**What would falsify it:** a non-oracle SAM protocol that closes the CDS2K gap. That would contradict the "limited understanding of professional data" claim.

---

## 7. SAM-Adapter (samadapter) -- failure evidence and adapter gain only

1. **Identity.** "SAM Fails to Segment Anything? -- SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, Medical Image Segmentation, and More". arXiv 2304.09148, **v3 read**. v1 2023-04-18, v3 2023-05-02. Authors: Tianrun Chen (first) ... Ying Zang (last in the abs.html list).
2. **Setup.** The SAM ViT-H image encoder is **frozen**. The SAM mask decoder starts from SAM weights and **is tuned**. The decoder gets no prompts. Each transformer layer receives an adapter prompt P^i = MLP_up(GELU(MLP_tune^i(F_i))), where F_i = F_hfc + F_pe (high-frequency components plus patch embedding, following EVP). Training: AdamW, LR 2e-4, cosine; 20 epochs for COD, 90 for shadow, 120 for polyp; four A100s.
   - Equation (get_context): `P^i = {\rm MLP}_{up}\left({\rm GELU}\left({\rm MLP}_{tune}^i\left(F_i\right)\right)\right)`.
3. **SAM baseline protocol.** The tables do not say which SAM prompt gives the SAM row. The figure captions describe two modes: "SAM online" (everything mode) and a whole-image box with no points (Figs. 3-5). **This is weak prompting (our reading).**
4. **Results.**
   - Table 1 (camouflage), S_alpha, SAM -> SAM-Adapter: CHAMELEON 0.727 -> 0.896. CAMO 0.684 -> 0.847. COD10K 0.783 -> 0.883.
   - Table 1, MAE, SAM -> SAM-Adapter: CHAMELEON 0.081 -> 0.033. CAMO 0.132 -> 0.070. COD10K 0.050 -> 0.025.
   - Table 2 (ISTD shadow), BER: SAM 40.51 -> SAM-Adapter 1.43 (FDRNet 1.55).
   - Table 3 (Kvasir polyp), mDice / mIoU: SAM 0.778 / 0.707 -> SAM-Adapter 0.850 / 0.776.
   - The text's "+17.9% in S_alpha" matches no cell. The table deltas are +0.169 (CHAMELEON), +0.163 (CAMO) and +0.100 (COD10K).
5. **Relevance limits.** **No industrial dataset is measured.** The evidence transfers to inspection only by analogy: low contrast, camouflage and shadows. It shows that a frozen encoder with adapters and a tuned decoder recovers these failures when task labels exist.
6. **Outline feedback.** Fine in ch-inspection as one line: "SAM fails on shadows (BER 40.51); adapters with a tuned decoder fix it (1.43)". Mark it non-industrial.

**What would falsify it:** a strongly prompted SAM (boxes or points) that closes most of the gap without adapters. The paper does not test this.

---

## 8. SAM Struggles in Concealed Scenes (concealed)

1. **Identity.** "SAM Struggles in Concealed Scenes -- Empirical Study on 'Segment Anything'". arXiv 2304.06022, **v4 read**. v1 2023-04-12, v4 2023-09-30. Authors: Ge-Peng Ji (first) ... Luc Van Gool (last). Sci. China Inf. Sci. 66, 226101 (2023).
2. **Claim.** "SAM looks unskilled in concealed scenes": camouflaged animals, industrial defects and medical lesions, in unprompted settings (Abstract).
3. **Measurement setup.** Everything mode, then **oracle selection of the mask with the best IoU against GT** (Sec. 2.1). COS benchmarks: CAMO (250), COD10K (2,026), NC4K (4,121). Baselines: CamoFormer-P/-S and HitNet. Metrics: S_alpha, weighted F, MAE, E (ad/mn/mx) and F (ad/mn/mx). **The industrial part is qualitative only**: online demo, KolektorSDD, MagneticTile and MVTec AD (Fig. 2).
4. **Results.**
   - Table 1, COD10K, SAM ViT-B / L / H: S_alpha 0.585 / 0.751 / 0.781. F_beta^w 0.353 / 0.655 / 0.700. MAE 0.108 / 0.065 / 0.054.
   - Table 1, COD10K best SOTA: S_alpha 0.871 (HitNet), MAE 0.023.
   - Table 1, CAMO, ViT-H: S_alpha 0.677, F_beta^w 0.594, E^mx 0.682. CamoFormer-S: 0.876 / 0.832 / 0.938.
   - Table 1, NC4K, ViT-H: S_alpha 0.763, E^mx 0.777. CamoFormer-P: 0.892 / 0.946.
   - Text: the E^mx gap is 13.8% on COD10K, 25.6% on CAMO and 16.9% on NC4K.
   - Inconsistencies:
     - The text says F_beta^w rises "from 0.353 to 0.655 on CAMO" and then to 0.700. Those are the **COD10K** cells; CAMO is 0.238 / 0.534 / 0.594.
     - The COD10K E^mx Delta of 13.8% is computed against 0.938 (HitNet), while the text names CamoFormer-S (0.941).
   - Industrial qualitative findings (Fig. 2, Sec. 2.4b): on close-up product shots, SAM "segment[s] the main part of the object, such as the screw ... and bottle", not the defect. It cannot separate surface cracks or a poke in leather from the textured background. The paper's reason: SAM is trained on natural objects with standard sizes and high contrast.
   - Discussion: SAM often splits an occluded object into several masks. Granularity and uncertainty are the bottlenecks.
5. **Limitations.** Oracle mask choice, which gives an upper bound. Only COS is quantitative. Demo-based visuals.
6. **Outline feedback.** This paper gives the clearest primary statement of "SAM segments objects, not anomalies" (Fig. 2). Pair it with SAA's wick figure.

**What would falsify it:** SAM-generated masks that match COS SOTA with oracle selection. Table 1 shows they do not.

---

## Cross-paper outline feedback (ch-inspection)

- Split "label acceleration" into two claims. (a) Point-click labeling (Industrial SAM) is **qualitative, with no time numbers**. (b) Box -> SAM pseudo-mask -> student training (Boxes2Pixels) is **measured**, with recall +0.1856 from self-correction.
- No paper in this set reports labeling time saved, in seconds or as a percentage. Do not put such a number on the page.
- DefectSAM is IR thermography on CFRP. KairosAD is supervised image-level OK/NG and uses no masks. SAM-Adapter has no industrial measurement. Label each one this way on its card.
- The two quantitative failure studies (Not Always Perfect, Concealed) both use oracle mask selection. State that their SAM numbers are best-case.
- SAA: SAM is only the refiner, and the measured gains come from expert rules and saliency.

## Practical recipe evidence (our reading; each bullet cites its support)

- **Do not use SAM everything mode to find defects.** On wood defects it "does not generate any results" (Not Always Perfect, Fig. 9). With oracle mask choice it still has the largest gap on CDS2K: MAE 0.265 vs 0.089 for DGNet (Not Always Perfect, Table 1g). On close-up products it masks the screw or bottle, not the defect (Concealed, Fig. 2).
- **Use SAM as a mask tool inside a human-in-the-loop labeler, with point clicks.** Five borescope defect classes needed only 1-3 clicks each (Industrial SAM, Fig. 12). The paper measures no time saving (Sec. 4.2).
- **Add negative clicks and check the result for diffuse defects (dirt, stains, contamination).** One click over-segmented dirt (Industrial SAM, Fig. 6b). SAM false positives come from shadows, stains and texture (Boxes2Pixels, Sec. 3.2).
- **Expect SAM to miss thin cracks and low-contrast defects even inside a box.** Do not treat SAM masks as ground truth. See Boxes2Pixels Sec. 3.2 and Fig. 3, and DefectSAM Table 3: zero-shot box-prompted SAM IoU is 49.32 flat / 40.44 R-type, while F2 is 98.24 on flat parts (Table 2). SAM finds the defect, but the extent is wrong.
- **To turn existing detection boxes into a segmentation model, train a student on SAM pseudo-masks with noise-aware losses.** Boxes2Pixels reaches mIoU_anom 0.6523 vs 0.5881 for SegFormer-B2 (Table 1). One-sided self-correction raises binary recall from 0.6195 to 0.8051 (Table 3). Keep a small manually masked test set, because validation against pseudo-masks hides the gain: self-correction moves validation mIoU_anom only from 0.5049 to 0.5093 (Table 2).
- **Add a separate binary defect-vs-background head when defects are sparse.** Removing it drops validation mIoU_anom from 0.5093 to 0.3851 (Boxes2Pixels, Table 2).
- **When you have task labels, fine-tune the decoder (and the encoder if you can) or add adapters, instead of using zero-shot SAM.** DefectSAM IoU went from 49.32 (SAM) to 78.58 on flat parts (DefectSAM, Table 3). SAM-Adapter cut shadow BER from 40.51 to 1.43 and raised COD10K S_alpha from 0.783 to 0.883 (SAM-Adapter, Tables 1-2; non-industrial).
- **For zero-shot anomaly segmentation without any images, plan for rule filters and per-product expert prompts, and still expect low accuracy.** SAA+ F_p is 34.85 in total but 28.82 on objects (SAA, Table 1). Removing the location and area filter drops texture F_p from 53.79 to 21.83 (SAA, Table 2).
- **Use the largest SAM encoder you can afford for zero-shot use; past ViT-L the gain is small.** CDS2K MAE goes 0.372 (B), 0.281 (L), 0.265 (H) (Not Always Perfect, Table 1g). COD10K S_alpha goes 0.585, 0.751, 0.781 (Concealed, Table 1).
- **On edge devices, a distilled SAM encoder (MobileSAM) plus a small head is a working OK/NG classifier.** It costs about 211-218 ms per image on Jetson NX/AGX (KairosAD, Sec. 4.4). MVTec I-AUROC is 99.10, but VisA is 92.3 and VisA Capsules only 69.7 (KairosAD, Tables 1-2). It needs labeled anomalous images (WBCE loss) and gives no masks.
- **For real-time pixel segmentation in production, deploy the distilled student, not SAM.** Boxes2Pixels runs at 6.20 ms / 161.4 FPS on an H100 with 5.6 M trainable parameters (Boxes2Pixels, Table 5). SAM is not needed at inference (Sec. 1).
- **For a new imaging modality (e.g. thermography), budget for fine-tuning.** Zero-shot SAM misses defects on irregular R-type parts (F2 68.02). Fine-tuned DefectSAM reaches IoU 81.98 there (DefectSAM, Tables 2-3). Use a sequence-level split: the paper's random frame split may leak (our reading).
