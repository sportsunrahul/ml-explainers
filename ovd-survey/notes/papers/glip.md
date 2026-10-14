# GLIP -- Grounded Language-Image Pre-training

## 1. Identity
- Title: "Grounded Language-Image Pre-training"
- arXiv 2112.03857, read v2. v1 2021-12-07 (Tue, 7 Dec 2021 17:47:50 UTC). v2 2022-06-17 (Fri, 17 Jun 2022 10:32:21 UTC) (abs.html submission history).
- abs.html Comments: "CVPR 2022; updated visualizations; fixed hyper-parameters in Appendix C.1".
- Authors: Liunian Harold Li (first; equal contribution with Pengchuan Zhang and Haotian Zhang), ..., Jianfeng Gao (last). UCLA, Microsoft Research, U. Washington, UW-Madison, Microsoft Cloud and AI, IDEA. Corresponding author: Pengchuan Zhang.
- Venue: CVPR 2022.
- Repo: https://github.com/microsoft/GLIP (abstract). No README downloaded under `sources/glip/`, so no `code_only` values exist for this paper.
- Source read: `src/main.tex` -> `sections/1_intro.tex`, `2_relatedwork.tex`, `3_glip.tex`, `4_other.tex`, `5_appendix_for_all.tex`. (`sections/5_appendix.tex` and `legacy/` are older copies that main.tex does not input; ignored.) `paper.pdf` for table/figure/equation numbers.
- Section map (PDF): 3.1 Unified Formulation, 3.2 Language-Aware Deep Fusion, 3.3 Pre-training with Scalable Semantic-Rich Data; 4 Transfer to Established Benchmarks (4.1 COCO, 4.2 LVIS, 4.3 Flickr30K, 4.4 Analysis); 5 ODinW (5.1 Data Efficiency, 5.2 One Model for All Tasks). Appendix: A Visualization, B Equivalence, C.1 Pre-training Details, C.2 Evaluation Details, C.3 Public vs crawled data, D Deep-fusion cost/performance, E.1 ODinW datasets, E.2 Manual Prompt Tuning, E.3 Data Efficiency, E.4 One Model for All, E.5 All Results.

## 2. Core problem
CLIP learns image-level concepts from raw image-text pairs, but detection needs object-level, language-aware representations. Detection datasets cap the vocabulary at about 2,000 categories and are costly to scale. GLIP asks for one pre-trained detector that transfers to new categories and domains zero-shot or few-shot.

## 3. What prior methods did and why that hurt (per the paper)
- Standard detectors train a fixed class set; the largest datasets (COCO, OpenImages, Objects365, VG) have "no more than 2,000 object classes" and are "costly to scale up" (Sec. 2, Sec. 3.3).
- Self-training for detection (Zoph et al.): a teacher detector pseudo-labels raw images, but "the generated data are still limited in terms of the size of the concept pool, as the teacher can only predict labels defined in the concept pool" (Sec. 3.3).
- CLIP "fuses vision and language only at the last dot product layer" (late fusion); GLIP claims deep fusion "is crucial to learn high-quality language-aware visual representations" (Sec. 1).
- Zero-shot detection (Bansal et al. with GloVe as P; OVR-CNN with LM phrase features) is GLIP-T (A)-like: it relies "purely on the language encoder to generalize to new concepts" (Sec. 3.1, Sec. 4). GLIP adds deep fusion and image-text scaling.
- MDETR inspired the formulation; GLIP claims to go further with "an effective approach to reformulate detection as grounding and a simple unified loss for both detection and grounding tasks" (Sec. 3.1).
- Open-world proposal methods try to propose every novel object; GLIP "only needs to propose objects mentioned in the text prompt as the detection branch is conditioned on the prompt" (Sec. 2).
- Linear probing (Wang et al.) works for classical detectors on common domains but lags full tuning on ODinW's novel domains (App. E.4).

## 4. The one key idea
Replace the box classifier `S_cls = O W^T` with region-token alignment `S_ground = O P^T`, where P is the token features of a text prompt that lists the classes ("person. bicycle. car. ..."). Detection then becomes phrase grounding, so detection data, human grounding data and self-labelled web image-text pairs all train one model. Cross-attention between image and text in the last encoder layers (deep fusion) makes the image features depend on the prompt.

## 5. Mechanism, step by step
1. Build the prompt. Detection: class names joined by ". " (COCO: 80 names; App. B uses a fixed order). Grounding: the caption. Max 256 tokens (App. B, C.1). If classes do not fit: O365 pretraining keeps all positive classes and samples negatives up to 85 categories (p=0.5) or uniformly picks N in [1, 85 - |C_pos|] (p=0.5), order shuffled (App. C.1); LVIS evaluation uses 40 categories per prompt and queries the model several times (App. C.2).
2. Encode. Image -> Swin backbone -> visual features O^0. Prompt -> pre-trained BERT -> token features P^0 (Eq. 4-6 text).
3. Deep fusion, for i = 0..L-1 (L = number of DyHead modules): X-MHA computes O^i_t2i (image attends to text) and P^i_i2t (text attends to image) (Eq. 4); O^{i+1} = DyHeadModule(O^i + O^i_t2i) (Eq. 5); P^{i+1} = BERTLayer(P^i + P^i_i2t), where BERTLayer is a newly added layer on top of pre-trained BERT (Eq. 6). Output O = O^L, P = P^L. Drop the red context terms and the model is late fusion.
4. Score. S_ground = O P^T, shape N x M (N regions, M sub-word tokens) (Eq. 3).
5. Target. Expand T in {0,1}^{N x c} to T' in {0,1}^{N x M}: every sub-word token of a positive phrase is positive; added tokens ("Detect:", ",", LM special tokens, the [NoObj] token appended at the end) are negative for all regions. Loss: focal binary sigmoid loss on (S_ground; T') plus the detector's localization loss (Eq. 1). This is the loss used in Sec. 4 and 5. Footnote variant: with a multi-class CE loss (following MDETR), unmatched boxes are matched to [NoObj], and phrase probability = sum of token probabilities.
6. Inference. Phrase probability = average of its token probabilities (sigmoid case) (Sec. 3.1).
7. Self-training data. Teacher GLIP trained on gold detection + grounding data predicts boxes for noun phrases (found by an NLP parser, cite Bird et al. NLTK) in web image-text pairs; student trains on gold + pseudo data (Sec. 3.3). For Cap4M the teacher is GLIP-T (C) (Sec. 4). The teacher for GLIP-L's Cap24M is not named in Sec. 4 (App. A only says "our teacher model"). Pseudo labels kept at confidence > 0.5 (Sec. 1).
8. Harder caption training (App. C.1): with p=0.3 mix 19 negative captions into the text; with p=0.3 mix a uniform 1-19 negatives; else none. The model must find the positive caption, then ground its phrases.
9. Transfer (Sec. 5.2, App. E.4): zero-shot (write the prompt); manual prompt tuning (descriptive words, no training); prompt tuning (compute P^0 once with the language backbone, discard the backbone, tune only P^0); linear probing (box head + projection layer between region and prompt embeddings); full fine-tuning.

Trainable vs frozen:
- Pre-training: image and language encoders "trained end-to-end" (Sec. 3.1). LR 1e-5 for the language backbone, 1e-4 for everything else (App. C.1). Nothing is stated as frozen.
- ODinW fine-tuning: "we freeze the bottom 2 layers of the backbone" (App. E.3).
- Prompt tuning: only P^0 trains; all grounding-model weights fixed (Sec. 5.2).
- Linear probing: box head + region-prompt projection layer only (Sec. 5.2).

## 6. Equations
Custom macros (from `src/mlVecMat.tex`): `\R` = `\mathbb{R}`, `\Lcal` = `\mathcal{L}`, `\Ccal` = `\mathcal{C}`, `\Rcal` = `\mathcal{R}`. `\ie` from cvpr.sty. Numbering verified against PDF p. 3-4: (1) loss, (2) S_cls, (3) S_ground, (4)-(6) fusion. The prompt display and the X-MHA block are unnumbered.

Eq. 1 (`\label{eqn:loss}`):
```tex
\Lcal = \Lcal_{\text{cls}} + \Lcal_{\text{loc}}.
```
Meaning: classification loss plus localization loss. RPN loss (two-stage) is merged into L_loc because it "does not use semantic information of object classes"; L_loc may include centerness loss in one-stage detectors.

Eq. 2 (`\label{eqn:cls_logits}`):
```tex
O \!=\! \text{Enc}_{I}(\text{Img}), \   S_{\text{cls}} \!=\! O W^T, \  \Lcal_{\text{cls}} \!=\! loss(S_{\text{cls}}; T).
```
Gloss (footnote 3 and text): O in R^{N x d} region/box features; N = number of region/box features; d = visual feature hidden dimension; W in R^{c x d} box-classifier weights (bias ignored); c = number of object classes; S_cls in R^{N x c} logits; T in {0,1}^{N x c} target from many-to-1 matching or bipartite Hungarian matching; loss = cross-entropy (two-stage) or focal loss (one-stage).

Unnumbered display (Sec. 3.1), the "human-friendly" prompt that works worse:
```tex
$$\text{Prompt} = \text{``Detect: person, bicycle, car, ... , toothbrush''},$$
```
Paper: with BERT initializing Enc_L, "the prompt ``person. bicycle. car. ... . toothbrush'' works better than the more human-friendly prompt described above." No number is given for this comparison.

Eq. 3 (`\label{eqn:ground_logits}`):
```tex
O \!=\! \text{Enc}_{I}(\text{Img}),~ P \!=\! \text{Enc}_{L}(\text{Prompt}), ~ S_{\text{ground}} \!=\! O P^{\top},
```
Gloss: P in R^{M x d} contextual word/token features, "plays a similar role to the weight matrix W"; S_ground in R^{N x M}. M > c because (1) multi-word phrases ("traffic light"), (2) sub-word splits ("toothbrush" -> "tooth#" and "#brush", written `tooth\#` / `\#brush` in the TeX), (3) added tokens ("Detect:", ",", special tokens), (4) the [NoObj] token at the end. Target expanded to T' in {0,1}^{N x M}.

Eq. 4-6 (`align` environment; labels `eq:xmha`, `eq:i2t`, `eq:t2i`). Note: label `eq:i2t` sits on the O (image) update and `eq:t2i` on the P (text) update, the reverse of what the names suggest. The red wrappers mark the fusion terms.
```tex
& \textcolor{red}{ O^i_{\text{t2i}}, P^i_{\text{i2t}} } = \text{X-MHA}(O^i, P^i), \label{eq:xmha} \quad i \in \{0, 1, .., L-1\}\\
& O^{i+1} = \text{DyHeadModule}(O^i \textcolor{red}{ + O^i_{\text{t2i}}}), \quad O = O^{L} \label{eq:i2t}, \\
& P^{i+1} = \text{BERTLayer}(P^i \textcolor{red}{ + P^i_{\text{i2t}}}), \quad P = P^{L} \label{eq:t2i},
```
Gloss: L = number of DyHeadModules in DyHead; BERTLayer = "newly-added BERT Layers on top of the pre-trained BERT"; O^0 = visual features from the vision backbone; P^0 = token features from BERT; O_t2i / P_i2t = context vectors from the other modality. "Without added context vectors ... the model is reduced to a late-fusion model."

X-MHA attention lines (unnumbered `align*`):
```tex
& O^{(q)} \!=\! O W^{(q,I)}, P^{(q)} \!=\! P W^{(q,L)}, Attn \!=\! O^{(q)} (P^{(q)})^{\top}\!/\! \sqrt{d}, \\
& P^{(v)} = P W^{(v,L)}, ~~ O_{\text{t2i}} = \text{SoftMax}(Attn) P^{(v)} W^{(out,I)}, \\
& O^{(v)} = O W^{(v,I)}, ~~ P_{\text{i2t}} = \text{SoftMax}(Attn^{\top}) O^{(v)} W^{(out,L)},
```
Gloss: "each head computes the context vectors of one modality by attending to the other modality". W^{(q,.)}, W^{(v,.)}, W^{(out,.)} are trainable, like query/value/output layers of MHSA. There is no separate key projection: one score matrix Attn (N x M) is used in both directions, transposed for text->image. Our reading: SoftMax(Attn) normalizes over the M tokens for each region; SoftMax(Attn^T) over the N regions for each token (axis not stated). Shapes per head: O_t2i is N x d, P_i2t is M x d.

Why deep fusion (paper): "1) It improves the phrase grounding performance. 2) It makes the learned visual features language-aware, and thus the model's prediction is conditioned on the text prompt. This is crucial to achieve the goal of having one model serve all downstream detection tasks" (Sec. 3.2). Prompt tuning works only with deep fusion (Sec. 2, 5.2): "prompt tuning is most effective for a model with deep vision-language fusion such as GLIP, while being far less effective for shallow-fused models."

## 7. Config / setup
| Item | Value | Source |
|---|---|---|
| Backbone | Swin-T (GLIP-T*), Swin-L (GLIP-L) | Table 1 |
| Fusion encoder | DyHead with X-MHA in every module; L = number of DyHeadModules | Sec. 3.2 |
| L (DyHead modules) | NOT STATED in paper | -- |
| Text encoder | pre-trained BERT; "BERT (base-uncased)" named in App. B (equivalence experiment). GLIP-L text encoder not separately named | Sec. 3.1, App. B |
| Added BERT layers | "newly-added BERT Layers" in Eq. 6; count NOT STATED (our reading of Eq. 6: one per fusion step, i.e. L) | Sec. 3.2 |
| d | NOT STATED | -- |
| Input resolution | NOT STATED | -- |
| N (regions) | NOT STATED (DyHead is a dense one-stage head; our reading: N = anchor locations over all levels) | -- |
| Max text tokens | 256 ("BERT can only encode ... at most 512 tokens ... we limit the input length to 256") | App. B, C.1 |
| Prompt cap, O365 pretraining | 85 categories | App. C.1 |
| LVIS eval chunk | 40 categories per prompt | App. C.2 |
| Pretrain GPUs / batch | Swin-T: 32 GPUs, batch 64. Swin-L: 64 GPUs, batch 64. GPU type NOT STATED | App. C.1 |
| Pretrain LR | 1e-5 language backbone, 1e-4 all else; x0.1 at 67% and 89% of total steps; "decay the learning rate when the zero-shot performance on COCO saturates". Optimizer, total steps NOT STATED | App. C.1 |
| COCO fine-tune LR | 1e-5 | App. C.2 |
| ODinW fine-tune | batch 4, LR 1e-4 (BERT 1e-5), wd 0.05, freeze bottom 2 backbone layers, x0.1 on val plateau, 3 seeds per X-shot | App. E.3 |
| Prompt tuning | LR 0.05, batch 4, wd 0.25 | App. E.4 |
| Linear probing | LR 1e-4, batch 4, wd 0.05; cosine variant scale 20.0, LR 0.01 | App. E.4 |
| Equivalence test | DyHead Swin-T, 2x schedule, batch 32, LR 1e-4 (BERT 1e-5): detection and grounding both 49.4 AP COCO val2017 | App. B |
| Cost test | inference P100 batch 1; training DGX-2 with 16 V100, 2 img/GPU (T), 1 img/GPU (L); gradient checkpointing on the fusion module | App. D.1 |

Model variants (Table 1):
| Model | Backbone | Deep fusion | Detection | Grounding | Caption |
|---|---|---|---|---|---|
| GLIP-T (A) | Swin-T | no | Objects365 | - | - |
| GLIP-T (B) | Swin-T | yes | Objects365 | - | - |
| GLIP-T (C) | Swin-T | yes | Objects365 | GoldG | - |
| GLIP-T | Swin-T | yes | Objects365 | GoldG | Cap4M |
| GLIP-L | Swin-L | yes | FourODs | GoldG | Cap24M |

Pre-training data (Sec. 4 unless noted):
- O365 (Objects365): 0.66M images, 365 categories.
- GoldG: 0.8M human-annotated gold grounding data curated by MDETR: Flickr30K, VG Caption, GQA; COCO images removed.
- Cap4M: 4M web image-text pairs, boxes generated by GLIP-T (C). Alternative public set: CC3M (3M) + SBU (1M); "CC+SBU GLIP-T performs slightly better than Cap4M GLIP-T on COCO, but slightly worse on the other datasets" (Table 6).
- FourODs: 2.66M: Objects365, OpenImages, Visual Genome (excluding COCO images), ImageNetBoxes. "aligned vocabulary of over 1,500 categories" (Sec. 4.4).
- Cap24M: CC12M + SBU, 24M web image-text data with generated boxes.
- GLIP-L total: "27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs" (abstract, Sec. 1). The 24M set yields 78.1M phrase-box pseudo annotations at confidence > 0.5, with 58.4M unique noun phrases (Sec. 1).
- Unique phrases in gold data: Flickr30K 44,518; VG Caption 110,689 (Sec. 3.3).
- GoldG+ (MDETR): 1.3M; GLIP-L last row of Table 2 adds GoldG+ and COCO train2017 (test-dev only, because part of COCO val2017 is in GoldG+).
- No category is held out: "we do not explicitly exclude any categories from our training set" (Sec. 2).

## 8. Results worth remembering
COCO, Table 2 (box AP; zero-shot on 2017val; fine-tune 2017val / test-dev):
- DyHead-T O365 "zero-shot" (restricted to COCO's 80 classes) 43.6; GLIP-T (A) 42.9; GLIP-T (B) 44.9; GLIP-T (C) 46.7; GLIP-T (Cap4M) 46.3; GLIP-T (CC3M,SBU) 46.6; GLIP-L 49.8.
- Fine-tune: GLIP-T (C) 55.1; GLIP-T 54.9 (Cap4M) / 55.2 (CC3M,SBU); GLIP-L 60.8 / 61.0; GLIP-L with COCO in pretraining 61.5 test-dev.
- Gray baselines: Faster RCNN RN50-FPN 40.2, RN101-FPN 42.0, DyHead-T 49.7, DyHead-L 58.4/58.7, DyHead-L (O365, IN21K) 60.3/60.6, SoftTeacher 60.7/61.3.
- Objects365 covers all 80 COCO categories (Sec. 4.1), so COCO zero-shot is domain transfer, not novel categories. Contributions per Sec. 4.1: reformulation slight drop (42.9 vs 43.6), deep fusion +2 (44.9), gold grounding the largest (46.7), image-text "slight or no improvement on COCO".

LVIS, Table 3 (no LVIS data used; APr / APc / APf / AP; MiniVal = MDETR's 5,000-image subset; Val v1.0):
| Model | MiniVal APr | APc | APf | AP | Val APr | APc | APf | AP |
|---|---|---|---|---|---|---|---|---|
| MDETR RN101 (gray) | 20.9 | 24.9 | 24.3 | 24.2 | - | - | - | - |
| MaskRCNN RN101 (gray) | 26.3 | 34.0 | 33.9 | 33.3 | - | - | - | - |
| Supervised-RFS RN50 (gray) | - | - | - | - | 12.3 | 24.3 | 32.4 | 25.4 |
| GLIP-T (A) | 14.2 | 13.9 | 23.4 | 18.5 | 6.0 | 8.0 | 19.4 | 12.3 |
| GLIP-T (B) | 13.5 | 12.8 | 22.2 | 17.8 | 4.2 | 7.6 | 18.6 | 11.3 |
| GLIP-T (C) | 17.7 | 19.5 | 31.0 | 24.9 | 7.5 | 11.6 | 26.1 | 16.5 |
| GLIP-T | 20.8 | 21.4 | 31.0 | 26.0 | 10.1 | 12.5 | 25.5 | 17.2 |
| GLIP-L | 28.2 | 34.3 | 41.5 | 37.3 | 17.1 | 23.3 | 35.4 | 26.9 |
- Metric caveats: GLIP does not say mask vs box; our reading: box AP (DyHead has no mask head). The MDETR and MaskRCNN gray rows equal MDETR Table 6 (box AP *fixed*, minival); the MDETR row is MDETR fine-tuned on 10% of LVIS. Checkpoints were chosen by minival performance because "models tend to overfit on LVIS during the course of pre-training" (App. C.2).
- Sec. 4.2: gold grounding +4.2 MiniVal APr (C vs B); image-text +3.1 more (GLIP-T vs C).

Flickr30K, Table 4 (R@1, any-box protocol): GLIP-L 86.7 val / 87.1 test; MDETR-ENB5 83.6 / 84.3; GLIP-T (O365,GoldG,Cap4M) 85.7 / 85.7. "outperforms previous SoTA by 2.8 points on test R@1."

Data ablation, Table 5 (GLIP-T; COCO 2017val zero-shot AP; LVIS MiniVal APr/APc/APf/AP):
| Row | Data | COCO | APr | APc | APf | AP |
|---|---|---|---|---|---|---|
| 1 | VG w/o COCO | 26.9 | 4.9 | 10.4 | 23.2 | 16.1 |
| 2 | + GoldG | 29.2 | 7.8 | 14.0 | 24.5 | 18.5 |
| 3 | OpenImages | 29.9 | 12.8 | 12.1 | 17.8 | 14.9 |
| 4 | + GoldG | 33.6 | 15.2 | 16.9 | 24.5 | 20.4 |
| 5 | O365 | 44.9 | 13.5 | 12.8 | 22.2 | 17.8 |
| 6 | + GoldG | 46.7 | 17.7 | 19.5 | 31.0 | 24.9 |
| 7 | O365,GoldG,Cap4M | 46.3 | 20.8 | 21.4 | 31.0 | 26.0 |
| 8 | FourODs | 46.3 | 15.0 | 22.5 | 32.8 | 26.8 |
- The "0.8M gold grounding beats 2M more detection data on rare categories" claim (Sec. 3.3, 4.4) = Row 6 (0.66M detection + 0.8M gold) vs Row 8 (2.66M detection): APr 17.7 vs 15.0, COCO 46.7 vs 46.3. Row 8 is higher on APc, APf and overall AP (26.8 vs 24.9). With image-text: APr 20.8 vs 15.0.

Deep-fusion cost, Table 7 (FPS per GPU; GB):
| Model | Fusion | Inference P100 FPS | Inf. mem | Train V100 FPS | Train mem |
|---|---|---|---|---|---|
| GLIP-T | no | 4.84 | 1.0 GB | 2.79 | 11.5 GB |
| GLIP-T | yes | 2.52 | 2.4 GB | 1.62 | 16.0 GB |
| GLIP-L | no | 0.54 | 4.8 GB | 1.27 | 19.7 GB |
| GLIP-L | yes | 0.32 | 7.7 GB | 0.88 | 23.4 GB |
- "Deep fusion brings less than 1x additional computational cost." Without fusion "we could cache the language embeddings of the prompts; thus the inference time of GLIP without deep fusion is equivalent to that of DyHead" (App. D.1).

Late vs deep fusion, Table 8 (GLIP-T; ODinW = full-model tuning average AP):
| Fusion | Data | COCO ZS | COCO FT | LVIS APr | LVIS AP | Flickr R@1 | ODinW 0 | 1 | 3 | 5 | 10 | Full |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no | O365 | 42.9 | 52.9 | 14.2 | 18.5 | 46.4 (gray) | 28.7 | 43.5 | 48.8 | 50.4 | 54.1 | 63.6 |
| yes | O365 | 44.9 | 53.8 | 13.5 | 17.8 | 41.4 (gray) | 33.2 | 48.0 | 52.0 | 53.2 | 54.9 | 62.7 |
| no | O365,GoldG | 41.6 | 52.9 | 15.8 | 26.1 | 82.4 | 35.5 | 47.2 | 51.9 | 53.8 | 54.3 | 65.1 |
| yes | O365,GoldG | 46.7 | 55.1 | 17.7 | 24.9 | 84.8 | 44.4 | 49.6 | 53.8 | 54.8 | 57.2 | 63.9 |
- Fusion helps COCO, Flickr30K, low-shot ODinW; full-shot ODinW is slightly lower with fusion in both pairs; LVIS mixed (App. D.2).

Manual prompts, Table 10 (App. E.2; zero-shot AP of GLIP-T; the caption says "CLIP-T", a typo: the designed-prompt values equal the GLIP-T row of Table 14):
| Dataset | Original prompt | AP | Designed prompt | AP |
|---|---|---|---|---|
| Aquarium | penguin / puffin / stingray | 17.7 | "penguin, which is black and white" / "puffin with orange beaks" / "stingray which is flat and round" | 18.4 |
| Rabbits | Cottontail-Rabbits | 68.0 | rabbit | 70.2 |
| EgoHands | hand | 22.1 | hand of a person | 50.0 |
| Mushrooms | Cow. Chanterelle | 13.6 | flat mushroom. yellow mushroom | 73.8 |
| Packages | package | 50.0 | there is a package on the porch | 72.3 |
| Pothole | pothole | 17.8 | there are some holes on the road | 17.7 |
- Pothole prompt did not help GLIP-T but was kept because it helped GLIP-T (C) (caption). Sentence prompts were applied only to models with grounding data (C, GLIP-T, GLIP-L); A and B used the designed prompts only for Rabbits and Mushrooms (App. E.2).
- Fig. 6 / Sec. 5.2: adding "flat and round" raises stingray AP50 from 4.6 to 9.7.

ODinW-13 data efficiency, Table 11 (average AP over 13 datasets; zero-shot, then full-model tuning with X-shot; mean +- std over 3 runs):
| Model | 0 | 1 | 3 | 5 | 10 | All |
|---|---|---|---|---|---|---|
| DyHead-T COCO | - | 31.9+-4.1 | 44.2+-0.4 | 44.7+-2.1 | 50.1+-2.0 | 63.2 |
| DyHead-T O365 | - | 33.8+-4.3 | 43.6+-1.2 | 46.4+-1.4 | 50.8+-1.6 | 60.8 |
| GLIP-T (A) | 28.7 | 43.5+-1.5 | 48.8+-0.4 | 50.4+-0.7 | 54.1+-0.5 | 63.6 |
| GLIP-T (B) | 33.2 | 48.0+-0.8 | 52.0+-0.4 | 53.2+-0.9 | 54.9+-0.7 | 62.7 |
| GLIP-T (C) | 44.4 | 49.6+-0.3 | 53.8+-0.2 | 54.8+-1.0 | 57.2+-1.1 | 63.9 |
| GLIP-T | 46.5 | 51.1+-0.1 | 54.9+-0.3 | 56.4+-0.5 | 58.4+-0.2 | 64.9 |
| GLIP-L | 52.1 | 59.9+-1.7 | 62.1+-0.8 | 64.2+-0.4 | 64.9+-0.9 | 68.9 |
- Paper's readings: "A zero-shot GLIP-T outperforms 5-shot DyHead-T" (46.5 vs 46.4 O365) and "a one-shot GLIP-L is competitive with a fully supervised DyHead-T" (59.9 vs 60.8 O365 / 63.2 COCO) (Sec. 5.1); intro adds "a zero-shot GLIP-L outperforms a 10-shot supervised baseline (Dynamic Head) pre-trained on Objects365" (52.1 vs 50.8).
- In-paper discrepancy: GLIP-T (A) zero-shot average is 28.7 in Tables 8 and 11 but 28.8 in Table 14 Avg.

Linear probing, Table 12 (avg AP; 1 / 3 / 5 / 10 / All):
- DyHead-T COCO 22.7 / 32.7 / 30.5 / 34.1 / 43.1; DyHead-T O365 30.7 / 36.2 / 39.6 / 40.0 / 48.2.
- GLIP-T (A) 34.6 / 35.9 / 37.6 / 37.9 / 44.1; GLIP-T (B) 40.9 / 42.8 / 44.0 / 44.4 / 51.8; GLIP-T (C) 43.9 / 45.4 / 45.9 / 46.7 / 52.7; GLIP-T 48.9 / 50.5 / 50.4 / 51.2 / 55.1; GLIP-L 54.1 / 54.7 / 55.0 / 55.9 / 59.2.

Prompt tuning, Table 13 (avg AP; mean +- std):
| Model | 1 | 3 | 5 | 10 | All |
|---|---|---|---|---|---|
| GLIP-T (A) | 34.0+-0.1 | 37.0+-0.6 | 40.0+-0.4 | 39.2+-1.0 | 43.3 |
| GLIP-T (B) | 46.4+-0.5 | 49.0+-0.9 | 50.6+-0.5 | 52.7+-0.1 | 58.5 |
| GLIP-T (C) | 50.6+-0.5 | 52.9+-0.5 | 53.9+-0.7 | 55.8+-1.1 | 62.8 |
| GLIP-T | 49.9+-0.7 | 53.7+-1.6 | 55.5+-0.6 | 56.6+-0.3 | 62.4 |
| GLIP-L | 59.5+-0.4 | 61.4+-0.4 | 62.4+-0.6 | 64.1+-0.6 | 67.9 |
- Contrast (All column): GLIP-T (A) prompt 43.3 vs linear 44.1 vs full 63.6 (no fusion: prompt tuning ~ linear probing). GLIP-T prompt 62.4 vs full 64.9; GLIP-L 67.9 vs 68.9. "as the model and data size grow larger, the gap between full-model tuning and prompt tuning becomes smaller" (Sec. 5.2, Fig. 7).

Per-dataset zero-shot, Table 14 (AP; PascalVOC here is AP@[.5:.95] via the COCO script per Table 15 caption):
| Model | VOC | AerialDrone | Aquarium | Rabbits | EgoHands | Mushrooms | Packages | Raccoon | Shellfish | Vehicles | Pistols | Pothole | Thermal | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLIP-T (A) | 47.7 | 9.8 | 16.8 | 60.5 | 1.6 | 13.7 | 48.5 | 44.4 | 20.4 | 52.4 | 25.3 | 0.8 | 32.3 | 28.8 |
| GLIP-T (B) | 50.6 | 4.9 | 19.4 | 71.6 | 0.5 | 21.8 | 29.7 | 47.0 | 21.4 | 56.0 | 47.4 | 3.6 | 57.1 | 33.2 |
| GLIP-T (C) | 51.6 | 8.1 | 22.6 | 71.1 | 49.1 | 69.4 | 65.6 | 51.5 | 29.3 | 49.9 | 42.7 | 17.0 | 49.2 | 44.4 |
| GLIP-T | 56.2 | 12.5 | 18.4 | 70.2 | 50.0 | 73.8 | 72.3 | 57.8 | 26.3 | 56.0 | 49.6 | 17.7 | 44.1 | 46.5 |
| GLIP-L | 61.7 | 7.1 | 26.9 | 75.0 | 45.5 | 49.0 | 62.8 | 63.3 | 68.9 | 57.3 | 68.6 | 25.7 | 66.0 | 52.1 |
- Sec. 5.1 / Fig. 5: "on Pothole and EgoHands, models without grounding data (A&B) performs terribly, while models with grounding data (C) outperform them with ease." Fig. 5 shows Pothole, EgoHands, Raccoon (novel to Objects365) and Cottontail Rabbits, Pascal VOC (covered by Objects365); Fig. 5 plots VOC as AP50 (61.9 / 63.6 / 65.3 / 69.4 / 74.0), not the AP of Table 14.
- ODinW datasets (Table 9, train/val/test images): e.g. Packages 19/4/3, Mushrooms 41/5/5, AerialDrone 52/15/7, Pothole 465/133/67, EgoHands 3840/480/480.

## 9. Limitations stated and failure cases
- Deep fusion on LVIS is "unclear": with detection-only data it degrades LVIS; with grounding data it lowers APc but raises APr. Hypothesis: the language model "overfits" to O365 categories and fusion "amplifies" it because the visual features depend on the text (App. D.2).
- Splitting categories over several prompts "incurs minor performance drop" (43.6 DyHead-T vs 42.9 GLIP-T (A) on COCO) (App. B).
- Overfitting during pre-training on LVIS and Flickr30K; best checkpoints selected on LVIS minival / Flickr30K val (App. C.2).
- ODinW excludes Roboflow datasets "too challenging (e.g., detecting different kinds of chess pieces) or impossible to solve without specific domain knowledge (e.g., understanding sign language)" (App. E.1). This biases ODinW-13 toward easier sets.
- Linear probing lags full tuning on novel domains (App. E.4).
- Failure example: plain "stingray" misses most stingrays; the attribute prompt fixes some (Fig. 6). Mushrooms with original names "Cow. Chanterelle" scores 13.6 (Table 10). Pothole stays low for all models (best 25.7, GLIP-L).
- Future work: "a detailed study of how GLIP scales with text-image data size" (Sec. 6).

## 10. Outline feedback
- Ch 1: Eq. 2 and Eq. 3 numbering and dims are correct (O: N x d, W: c x d, P: M x d, S_ground: N x M). Note the paper says d is "the visual feature hidden dimension"; P must share d.
- Ch 7: "vocabulary grows from about 2,000 categories to hundreds of thousands of phrases" is not a paper number. Use: Flickr30K 44,518 and VG Caption 110,689 unique phrases (gold), 58.4M unique noun phrases in the 24M pseudo-labelled pairs.
- Ch 7: the 0.8M-vs-2M claim is APr-only (Table 5 rows 6 vs 8: 17.7 vs 15.0). Row 8 wins APc, APf and AP. "2M" = 2.66M FourODs minus 0.66M O365. Say this.
- Ch 7: GLIP's COCO "zero-shot" is not novel-category (O365 covers COCO's 80 classes), and GLIP holds no LVIS category out. Do not put GLIP LVIS APr next to ViLD APr as like-for-like: ViLD is mask APr with rare classes held out; GLIP is (our reading) box AP with no holdout and minival-selected checkpoints. The MDETR baseline in GLIP Table 3 is fixed AP after 10% LVIS fine-tuning.
- Ch 7: the paper's subword example is "toothbrush" -> "tooth#" + "#brush". Use the paper's spelling, or say "WordPiece" if showing BERT's real "##brush" (our reading).
- Ch 7 branch MDETR: MDETR cannot be evaluated zero-shot because its data has no text without a box, "biasing the model to always predict boxes for a given text" (MDETR Sec. 3.2.1). GLIP's detection prompts list absent categories, which supplies negatives (our reading). Worth one sentence.
- Ch 8: "DyHead x6", d, input size and added-BERT-layer count are not in the paper or any downloaded source. Either fetch the official config (label `code_only`) or keep the archflow symbolic (L modules, d dims).
- Ch 8: "the image encoder reruns per prompt" overstates it. Only the fusion part reruns: O^0 comes from the Swin backbone before fusion (Eq. 4-6), and the non-fused model can cache text embeddings (App. D.1). Use Table 7: GLIP-T 4.84 -> 2.52 FPS, 1.0 -> 2.4 GB (P100, batch 1).
- Ch 8: add Table 8 (late vs deep with GoldG: ODinW 0-shot 35.5 vs 44.4; COCO 41.6 vs 46.7) and the prompt-tuning contrast (Tables 11-13). These are the paper's actual evidence that fusion matters.
- Ch 8: the manual prompt table is Table 10 (App. E.2), values are GLIP-T zero-shot AP; caption typo "CLIP-T".
- Ch 12 item 3: the margin is thin. Zero-shot GLIP-T 46.5 vs 5-shot DyHead-T O365 46.4+-1.4. Show the std. Also note the ODinW-13 selection excluded the hard Roboflow sets (App. E.1); this supports the 13-vs-35 point in ch 6/12.
- Ch 13 item 1: "Cow. Chanterelle" 13.6 -> "flat mushroom. yellow mushroom" 73.8 (Table 10). EgoHands "hand" 22.1 -> "hand of a person" 50.0 is the second-best example.
