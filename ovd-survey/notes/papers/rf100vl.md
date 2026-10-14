# RF100-VL (Roboflow100-VL)

## 1. Identity
- Title: Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models
- arXiv 2505.20612, read v4. v1 2025-05-27; v4 2025-10-22 (abs.html submission history).
- Authors: Peter Robicheaux (Roboflow, equal contribution) ... Neehar Peri (CMU, last). Seven authors: Robicheaux, Popov, Madan, Robinson, Nelson, Ramanan, Peri.
- Venue: NeurIPS 2025 Datasets and Benchmarks Track (abs.html comment, PDF footer).
- Repo: github.com/roboflow/rf100-vl; data: universe.roboflow.com/rf100-vl; site rf100-vl.org.

## 2. Core problem
VLM detectors score well on COCO-like classes and on ODinW, but these benchmarks hold mostly web-common concepts. The paper asks how detectors do on concepts that are rare on the web (defects, material classes, acronyms, X-ray, thermal, spectrograms), and how that changes from zero-shot to 10-shot to 10% labels to full labels, on one shared test split.

## 3. What prior benchmarks did and why that hurt (per the paper)
- RefCOCO and ODinW "typically evaluate performance on classes commonly found in internet pre-training"; the paper says such benchmarks "artificially inflate model performance" (Sec. 1, Table 1).
- Qwen2.5-VL's reported ODinW number uses a referential-grounding protocol: the model is prompted only with the classes that are present in the test image. Detection protocol prompts with all classes. The former "dramatically reduces the number of false positives" (Sec. 4.1). Reported 43.1 vs reproduced 30.9 (Table 1).
- YOLO papers report COCO mAP with Ultralytics (trapezoidal PR integration), which reads up to 2.7% higher than pycocotools (App. B, Table 4).

## 4. The one key idea
Build a benchmark that cannot be solved by prompting with class names, then measure every model on the same test split across four label budgets (zero-shot, 10-shot, 10% labels, full labels), and attach multi-modal "annotator instructions" (text + 10-shot images) to every dataset.

## 5. Mechanism (benchmark construction), step by step
1. Sort Roboflow Universe detection datasets by stars (500,000+ public datasets on the platform; Sec. 3.1, Fig. 4).
2. Manually drop datasets with common classes, single-focal-object images, or watermarks; prefer multi-object images and semantically ambiguous names (e.g. "button") (Sec. 3.1, Fig. 4 caption).
3. Re-annotate where annotations are not exhaustive. Paper text: 1693 hours "labeling, reviewing, and preparing" (Sec. 3.1). App. O: 30 annotators spent 2168 hours validating annotations. Two different hour counts; both as printed.
4. Cluster the 100 datasets into seven categories using per-dataset CLIP embeddings (Fig. 1 caption). Counts per category (Fig. 5 pie; matches App. R tables): Aerial 11, Document 10, Flora & Fauna 23, Industrial 22, Medical 13, Sports 6, Other 15.
5. Build 10-shot splits with the Wang et al. (TFA) K-shot protocol: 10 total instances per class; pick, among random candidate splits, the one with the largest average box area (Fig. 4 caption, App. E).
6. Generate annotator instructions with GPT-4o (prompted with nuImages guidelines as in-context example, metadata, class names, and the 10-shot images), then manually edit (Sec. 3.1, App. P, App. Q sample).
7. Evaluate each dataset independently with pycocotools AP, maxDets 500; average per category; no per-dataset tuning of parameters or prompts (Sec. 3.2, Sec. 4.1).

## 6. Equations
None. The paper has no displayed equations.

## 7. Measurement setup (analysis lens)
- Claim 1: RF100-VL is much harder than ODinW-13 for zero-shot VLMs. Measured: same model, two benchmarks, detection protocol. GroundingDINO 49.2 on ODinW-13 (Sec. 4.2 text only) vs 15.7 on RF100-VL (Table 2). Falsified if zero-shot All on RF100-VL were near the ODinW-13 number.
- Claim 2: gradient fine-tuning on 10 shots beats in-context visual and multi-modal prompting. Measured: Table 2 few-shot block, same splits. Falsified if MQ-GLIP (frozen, prompted with images) matched fine-tuned GroundingDINO or Detic.
- Claim 3: multi-modal annotator instructions give "limited" and inconsistent benefit to MLLMs (Sec. 4.2, App. J). Measured: Qwen and Gemini with class names vs instructions vs instructions + images.
- Claim 4: COCO gains do not transfer; YOLOv11 is similar to YOLOv8 on RF100-VL; within-family size spread is at most 1.9 mAP (Sec. 4.2).
- Noise floor: 10 seeds of YOLOv8 with federated loss give All std +-.230 (v8n) and +-.216 (v8s); per-dataset std averaged per category is +-2.50 / +-2.71 (Tables 8 and 9). A difference smaller than about 0.5 on All, or about 2.5 on a single dataset, is not a finding.
- Model configs that matter for comparisons (App. A): "GroundingDINO" = MM-GroundingDINO-L* from mmdetection (not the IDEA release); prompted with all class names in one prompt; fine-tuned 1000 iterations, batch 4, lr 3e-4, images (640, 1333), no augmentation. OWLv2 (HuggingFace) is prompted one class at a time. Detic = Swin-L, LVIS+COCO+IN-21K checkpoint. MQ-GLIP = GLIP Swin-L (Objects365, FourODs, GoldG, Cap24M), frozen. MLLMs get confidence 1.0 because they output no scores; multi-class prompts in Table 2, single-class in Table 1 (captions); Table 3 does not say.
- Fully-supervised baselines: YOLOv8/YOLOv11 n/s/m (Ultralytics defaults, 100 epochs, batch 16) and LW-DETR s/m (App. A, App. F). Semi-supervised: random 10% labeled, STAC pseudo-labels (App. F).

## 8. Results worth remembering
All AP, pycocotools, averaged over the datasets in a category. Full rows in `notes/data/rf100vl.json`.
- Label-budget ladder, Industrial (22 datasets):
  - zero-shot best: GroundingDINO 10.3 (Table 2, Zero-Shot, GroundingDINO, Industrial). OWLv2 7.8, Detic 6.0, MQ-GLIP-Text 7.8, Gemini 8.6, Qwen 4.1.
  - 10-shot fine-tuned GroundingDINO 37.8 (Table 2, Few-Shot, GroundingDINO, Industrial). Detic w/ fed loss 25.9; YOLOv8n (no fed loss) 28.1.
  - 10-shot in-context: MQ-GLIP-Image 3.9; MQ-GLIP (names + images) 7.8; Gemini (images) 9.5; Qwen (instr. + images) 5.8.
  - 10% labels: 50.7 (YOLOv8n) .. 52.9 (YOLOv8m) (Table 7, Semi-Supervised, Industrial).
  - full labels: 64.3 (YOLOv8n) .. 68.2 (LW-DETRm) (Table 7, Fully-Supervised, Industrial).
- Same ladder, All: zero-shot GroundingDINO 15.7; 10-shot GroundingDINO 33.6; 10% labels 39.3 .. 43.3; full 55.4 .. 59.8 (Tables 2, 7).
- Medical is the floor: zero-shot GroundingDINO 2.1, OWLv2 2.1, Detic 0.8, Qwen 1.6 (Table 2, Medical). 10-shot GroundingDINO 18.3.
- Industrial is the category where fully supervised models score highest of all seven (64.3-68.2 vs 49.2-61.0 elsewhere, Table 7). Our reading: the Industrial datasets are hard for zero-shot naming but easy once labels exist.
- Visual-only in-context prompting is worse than text-only: MQ-GLIP-Image All 6.4 vs MQ-GLIP-Text All 12.2; adding images to names gives MQ-GLIP 12.2, identical to text only (Table 2).
- MLLMs vs specialists: Gemini 2.5 Pro zero-shot All 11.6, Qwen 5.6, vs GroundingDINO 15.7 (Table 2). Gemini with instructions only: 4.5 (worse than names). Qwen improves with instructions: 5.6 -> 7.6.
- Benchmark comparison (Table 1, MLLMs single-class prompt): Qwen ODinW-13 reported 43.1, reproduced 30.9, RF100-VL 7.8; Gemini 41.9 / 33.7 / 8.0.
- Toolkit: Ultralytics mAP is 1.7-2.7 points above pycocotools on the same YOLO runs (Table 4: e.g. YOLOv8m 56.9 vs 59.6).
- RF20-VL challenge (Table 3, 20-dataset subset, not comparable to Table 2): best baseline 10-shot GroundingDINO All 33.4; winner BEATON 50.4 (Industrial 62.0). "17 mAP" in the abstract = 50.4 - 33.4.
- Failure attribution heuristic (App. L): if YOLOv8 (promptless) does well but the VLM (Detic) does not, the failure is semantic ambiguity; if both fail, feature extraction. Documents: Detic 19.6 vs YOLOv8m 23.3; Medical: Detic 8.5 vs YOLOv8m 16.0 (both 10-shot w/ fed loss, Table 2). Industrial is close: Detic 25.9 vs YOLOv8m 26.6.

## 9. Limitations the paper states, and failure cases
- Crowdsourced annotations; verifying specialized domains (medical) is hard (Sec. 4.4).
- Annotator instructions are GPT-4o-generated plus edits, may not reflect real instructions, and often do not encode typical cases, edge cases and negatives (Sec. 4.4).
- Generalist MLLMs can ingest instructions but are imprecise; specialists are precise but cannot use rich text (Sec. 4.4).
- MLLM boxes have no confidence; adding SigLIPv2 crop-text scores lifts RF20-VL Qwen 7.4 -> 9.7 and Gemini 5.7 -> 8.3; NMS changes almost nothing (App. K, Table 12).
- Hard-example types the paper uses to explain failure (Fig. 2, Sec. 1): (a) context-dependent names: "Block" (volleyball), "Thunderbolt" (a defect on a wire rope); (b) scientific acronyms and concepts: "MCP" (hand X-ray), "Fibrosis" (liver histology); (c) catch-all categories: "Normal Paper", "Soft Plastic"; (d) imaging modalities: "Bluetooth" (RF spectrogram), "Insulator" (thermal image). Intro also lists material properties (paper vs soft plastic) and "DIP" as an acronym.

## 10. Outline feedback
- Supports (ch 12 card 1): the ladder numbers in the outline are correct for v4 when taken from Table 2 (zero-shot 10.3, 10-shot 37.8) and Table 7 (full 64.3-68.2). Add the 10% rung: Industrial 50.7-52.9 (Table 7). The outline's "10% labels" step had no number.
- Supports: on rare industrial concepts a text prompt alone does not replace a custom detector. Best zero-shot Industrial 10.3 vs 64.3+ fully supervised. Ten labeled instances per class, with fine-tuning, closes about half of that gap (37.8; our reading).
- Supports (ch 13 item 1): name-is-jargon failure, with Fig. 2 examples; "Thunderbolt" on a wire rope is the industrial one. Use it instead of a made-up example.
- Does NOT support "visual prompts fix rare classes" (ch 11): in-context image prompts were the weakest specialist option here (MQ-GLIP-Image Industrial 3.9, All 6.4; names + images = names only). Every gradient fine-tuning baseline beat every in-context method (Sec. 4.2). Chapter 11 must show this next to T-Rex2.
- Does NOT support a per-defect-type claim. "Industrial" is a CLIP cluster of 22 datasets. By name, only about seven are inspection/defect sets (asphaltdistressdetection, cable-damage, deeppcb, defect-detection, wheel-defect-detection, screwdetectclassification, smd-components). Others are FRC robotics (2024-frc), aircraft turnaround, conveyor t-shirts, RF spectrograms (ism-band-packet-detection), water meters, recode-waste, infrared power equipment, truck movement. new-defects-in-wood is in Other; label-printing-defect-version-2 is in Document. Per-dataset class lists are not in the paper. Say "Industrial cluster", not "defect detection".
- Does NOT support "zero-shot from MLLMs is the answer": Gemini/Qwen are below GroundingDINO zero-shot and gain little from instructions.
- Comparability flags for the page:
  - "GroundingDINO" here = MM-GroundingDINO-L*, not the Swin-T/Swin-L IDEA checkpoints of ch 9. OWLv2 is prompted per class, GroundingDINO with all classes at once.
  - Table 1 fully-supervised RF100-VL column (54.9 .. 56.5) disagrees with Table 7 All (55.4 .. 57.0) and Table 4. Table 1 matches commented-out older LaTeX rows. Use Table 7.
  - Abstract: GroundingDINO and Qwen "less than 2%" zero-shot on "challenging medical imaging datasets". Table 2 Medical average: GroundingDINO 2.1, Qwen 1.6. The abstract may mean single datasets; do not quote "less than 2%" for GroundingDINO's category average.
  - Table 2 caption mentions semi/fully supervised rows that are not in v4 Table 2.
  - Two annotation-hour figures (1693 in Sec. 3.1; 2168 by 30 annotators in App. O).
  - 49.2 ODinW-13 for GroundingDINO is a text claim (Sec. 4.2), not a Table 1 cell; the variant and protocol are not stated.
- Our reading for ch 12 decision guide: RF100-VL supports "few-shot fine-tune an open-vocab detector" over "prompt alone" for the Industrial cluster, and it says nothing about forgetting (no out-of-domain eval after fine-tuning).
