# Revisiting Few-Shot Object Detection with Vision-Language Models (Foundational FSOD)

## 1. Identity
- Title: Revisiting Few-Shot Object Detection with Vision-Language Models
- arXiv 2312.14494, read v4. v1 2023-12-22; v4 2024-10-14 (abs.html).
- Authors: Anish Madan (CMU, equal contribution) ... Deva Ramanan (CMU, equal senior authorship). Four authors: Madan, Peri, Kong, Ramanan.
- Venue: NeurIPS 2024 Datasets and Benchmarks Track (abs.html comment).
- Repo: github.com/anishmadan23/foundational_fsod; splits on HuggingFace (anishmadan23/foundational_fsod).

## 2. Core problem
Classic few-shot detection (FSOD) pre-trains on base classes and tests on disjoint novel classes. Web-scale VLM detectors break that protocol, and zero-shot VLMs already beat FSOD methods on COCO. The paper asks what few-shot detection should measure now: aligning a pre-trained VLM to a target concept whose definition differs from the web's (nuImages "truck" cab vs "trailer"; "debris"; "pushable-pullable").

## 3. What prior methods did and why that hurt (per the paper)
- Base/novel split protocols try to stop concept leakage, but leakage already happens through ImageNet pre-training ("cat", "person" are novel in COCO-FSOD but in ImageNet; "car" vs "sports car") (Sec. 1).
- COCO novel classes are common web concepts, so zero-shot GroundingDINO gets higher AP on novel than on base classes (54.3 vs 46.3, Table 1) (Sec. 4.1).
- Standard fine-tuning treats every unannotated class as a negative, but K-shot splits are federated: each image is exhaustively labeled for one class only. A "car" in the background of a "motorcycle" shot becomes a false negative (Sec. 3.2, App. B).

## 4. The one key idea
Replace base pre-training with any (web-scale, possibly private) pre-training, then align the VLM to the target classes with K multi-modal examples (images + text, like annotator instructions), and evaluate only on the target classes. Benchmark: nuImages, 18 classes, K in {5, 10, 30}.

## 5. Mechanism, step by step
1. Pre-train anywhere (no base/novel split). Evaluate on the C target classes only (Sec. 3.1).
2. Build K-shot splits per class with the TFA procedure: pick a class and a random image; add it if its count for that class keeps the total <= K; repeat until exactly K per class; three random splits, report the mean (Sec. 4).
3. Align with one of seven strategies (Sec. 3.2): prompt engineering (synonyms from annotator instructions, embeddings averaged; App. A, Table 6), standard fine-tuning (last layers / box head), federated fine-tuning, language prompt tuning (tune only the text embedding P^0 of GLIP), visual prompting (MQ-GLIP-Image), multi-modal prompting (MQ-GLIP: images + text into a frozen GLIP via MQ-Det's Gated Class Scalable Perceiver), multi-modal chat assistants (GPT-4o).
4. Federated fine-tuning (App. B, Alg. 1): run the zero-shot detector on the training image; classes with no prediction above 0.2 confidence are pseudo-negatives; the BCE loss is computed only over (ground-truth classes) union (pseudo-negative classes); other classes are ignored. Baseline FedLoss instead samples |S| = 6 negative classes per image by square-root frequency.
5. Evaluate on a subset of nuImages val: images with at least one Few or Medium class instance, 1600 x 900, AP per cohort Many / Medium / Few (App. A).

## 6. Equations
No displayed equations. The federated loss is given only as pseudo-code (App. B, Algorithm 1 "Psuedo-Negative Federated Loss"):
```
pred = Detector(img)
pseudo_pos = filter(pred, thresh=0.2)
neg_classes = get_neg(pseudo_pos, all_classes)
select_classes = or(neg_classes, gt_classes)
loss = 0
for cls in select_classes:
    loss += BCE(pred[cls], gt[cls])
```
Gloss: `pseudo_pos` = classes the zero-shot detector believes are present (confidence > 0.2); `neg_classes` = all other classes, treated as surely absent; `gt_classes` = classes annotated in this K-shot image. Classes that the detector thinks are present but are not annotated get no gradient (neither positive nor negative).

## 7. Measurement setup (analysis lens)
- Claim A: zero-shot VLMs beat specialized FSOD methods on the classic benchmark. Measured: COCO FSOD 30-shot, AP / base AP / novel AP; GroundingDINO zero-shot vs 9 FSOD methods (Table 1). Falsified if any base-pretrained FSOD method beat 48.3 AP. None does (best DiGeo 33.1).
- Claim B: on nuImages, language + web pre-training matter more than shots. Measured: TFA pre-trained on COCO-base or LVIS-base vs on LVIS+IN-21K+COCO with a CLIP classifier, 5/10/30 shots (Table 2).
- Claim C: federated fine-tuning beats standard fine-tuning. Measured: Detic Swin-B, 10-shot, same splits (Table 3, Table 7). Falsified if fed FT <= standard FT.
- Claim D: multi-modal in-context prompting (MQ-GLIP) is the best open baseline and does not need gradients. Measured: MQ-GLIP vs MQ-GLIP-Text vs MQ-GLIP-Image (Table 3, Tables 10-11, Table 12 on LVIS).
- Held fixed: test subset, 3 random splits averaged. Not held fixed across rows: backbone (RN50 to Swin-L) and pre-training data differ per method (Table 3 lists them). The paper itself says comparisons should consider backbone and data (Sec. 4.5).
- Compute: one RTX 3090 for all baselines (App. A). Detic fed FT: batch 8, AdamW lr 3.75e-6, 8000 iterations. GLIP prompt tuning: 60 epochs, lr 0.025, batch 4, weight decay 0.25.

## 8. Results worth remembering
nuImages AP (All / Many / Med / Few), 10-shot unless noted.
- Zero-shot (Table 3): Detic Swin-B 14.40 / 25.83 / 16.59 / 2.32; GroundingDINO Swin-T 12.05 / 17.29 / 15.45 / 3.72; GLIP Swin-L 17.01 / 23.36 / 19.86 / 8.40; RegionCLIP RN50 2.50.
- Detic, zero-shot -> 10-shot (Table 3): standard FT 16.09; federated FT 17.24; federated FT + prompt engineering 17.71. Few cohort: 2.32 -> 3.73 -> 4.18 -> 4.75.
- Same idea with other numbers (App. B, Table 7, 10-shot): Detic zero-shot 14.26; + standard FT 15.53; w/ FedLoss 15.57; w/ pseudo-negatives 16.67; oracle true negatives 16.99; oracle exhaustive annotations 18.51. Caption: "baseline (14.3 AP), our approach (16.7 AP) ... (18.5 AP) upper-bound".
- Third value for Detic 10-shot fine-tune: 15.28 (Table 8, RegionCLIP comparison). Three tables, three numbers for the "same" setting; record all, do not reconcile.
- Language prompt tuning, GLIP Swin-L: 19.41, Few 10.39 (Table 3).
- In-context, no gradients (Table 3): MQ-GLIP-Text 17.01; MQ-GLIP-Image 14.07 (Few 3.34); MQ-GLIP 21.42 (Few 10.26); MQ-GLIP + GPT-4o-written descriptions 22.03 (text says 22.09; caption says "+0.6%").
- Shots barely matter for in-context: MQ-GLIP 21.45 (5-shot, Table 10), 21.42 (10-shot), 21.40 (30-shot, Table 11). Detic fed FT grows 16.58 -> 17.24 -> 18.64 (Tables 10, 3, 11).
- Rare stays low even with oracle data: Few cohort upper bound 4.21 (true negatives) and 3.93 (exhaustive annotations) vs 2.36 zero-shot (Table 7).
- Specific classes: pushable-pullable 3.6 -> 15.29 AP and barrier 11.6 -> 15.31 AP when MQ-GLIP gets GPT-4o descriptions; debris unchanged (Sec. 4.3 text).
- Example choice: stroller AP 13.09 (random 5-shot split) vs 18.54 (best split) (Fig. 4). Best split +1.72 AP at 5-shot (16.58 -> 18.30), +0.11 at 30-shot (18.64 -> 18.75) (Table 5).
- COCO FSOD, 30-shot (Table 1): GroundingDINO zero-shot 48.3 AP, 46.3 base, 54.3 novel; DiGeo 33.1 / 39.4 / 14.2.
- TFA on nuImages without language: < 3 AP at every K (Table 2: 1.33, 2.02 at 5-shot); TFA + CLIP classifier + more data 15.12 (5-shot), 16.09 (10), 17.22 (30). These equal Detic standard FT in Tables 10/3/11 (our reading: same runs).
- LVIS v0.5 10-shot (App. F, Table 12): Detic base-only R50 APr 16.3 without seeing rare data; naive FT 15.5; FedLoss 17.4; pseudo-negatives 19.8. Swin-B: 21.4 -> 26.7 -> 30.4 -> 32.6. MQ-GLIP (Swin-L) APr 40.1 vs MQ-GLIP-Text 33.0 vs MQ-GLIP-Image 25.1.
- CVPR 2024 challenge (Table 4): winner PHP_hhh 45.35 (private method) vs baseline MQ-GLIP 21.51 ("23.3 mAP" gap in abstract and caption is 45.35 - 22.03 = 23.32; our arithmetic).

## 9. Limitations the paper states, and failure cases
- Rare (Few) classes stay far below common ones; "Our work does not explicitly improve detection performance on rare classes" (Sec. 4.5).
- Different VLMs have very different zero-shot AP on the same novel class (stroller); ensembling may help (Sec. 4.5).
- Text prompts used are nouns only (class names and synonyms), not rich descriptions (Sec. 4.5).
- Cannot guarantee foundation models never saw the test data; urban imagery is web-like; out-of-domain data (e.g. medical) not tested (Sec. 4.5). (RF100-VL is the follow-up that does this.)
- Failure cases: GPT-4o classifies RPN crops "with high confidence" but wrongly, especially debris (Fig. 3, Sec. 4.3); RegionCLIP zero-shot fails at foreground vs background on nuImages (2.34 AP with LVIS-RPN, 26.44 AP with ground-truth boxes) (App. C, Table 9).

## 10. Outline feedback
- Supports (ch 12): "zero-shot is a strong baseline when the class is web-common" (COCO 48.3 vs 33.1) and "zero-shot is weak when the target definition is application-specific" (nuImages zero-shot 12.05-17.01 All, 2.32-8.40 Few).
- Supports a nuance the outline lacks: few-shot gains on nuImages are small. Gradient methods at 10 shots add +2.4 to +3.3 AP over their own zero-shot start (GLIP 17.01 -> 19.41 with language prompt tuning; Detic 14.40 -> 17.24 / 17.71 with federated FT); in-context multi-modal prompting gives more (21.42) with no training. Note the Table 4 baseline MQ-GLIP is 21.51, not 21.42. This contrasts with RF100-VL, where fine-tuning beat in-context by a wide margin. Our reading: the domain decides which adaptation wins; urban scenes are close to web data.
- Supports (ch 13): the concept-definition failure. Same word, different annotation policy ("truck" cab vs whole vehicle; rider counted as part of "bicycle"; Fig. 1). This is a failure mode the outline does not list. Suggest adding "the label policy differs from the web meaning of the name" as a taxonomy item, sourced to this paper Fig. 1.
- Supports "visual + text > either alone" for MQ-GLIP, and "visual alone < text alone" (14.07 vs 17.01 All; 3.34 vs 8.41 Few). This does NOT support "visual prompts for rare" in ch 11: on the Few cohort, image-only prompting is worse than names.
- Federated loss idea belongs in ch 12 as a one-line practitioner tip: when fine-tuning on a handful of sparsely labeled images, do not treat unlabeled classes as negatives. Effect size is small (Table 7: 15.53 -> 16.67).
- Use the version-consistent numbers per table and say which table. The page should not mix 16.09 (Table 3) with 16.67 (Table 7).
- Practical question, plainly: on an application whose class definitions differ from web usage (nuImages), zero-shot text prompts reach 12.05-17.01 AP and the Few cohort stays at or below 11.54 AP for every baseline in Tables 3, 10, 11 and 7 (30 shots and oracle labels included); only CVPR 2024 challenge entries go higher (best 20.19, Table 4). The evidence supports "a prompt alone is not enough when the label policy is specific", and it does not show any setting where 10 shots reach supervised quality (no fully supervised nuImages baseline is reported).
