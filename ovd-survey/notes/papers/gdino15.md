# Grounding DINO 1.5 -- Advance the "Edge" of Open-Set Object Detection (BRANCH, short note)

## 1. Identity
- Title: "Grounding DINO 1.5: Advance the "Edge" of Open-Set Object Detection"
- arXiv 2405.10300, read v2. v1 2024-05-16 (Thu, 16 May 2024 17:54:15 UTC); v2 2024-06-01 (abs.html).
- Authors: Tianhe Ren (first) ... Lei Zhang (last); 16 authors, IDEA Research.
- Venue: none stated (abs.html Comments: "homepage"). Technical report.
- Repo: https://github.com/IDEA-Research/Grounding-DINO-1.5-API . The abstract promises "Model examples and demos with API", not weights (our reading: API-only model).
- Source read: `src/sec/01_abstract.tex`, `03_model_training.tex`, `04_model_evaluation.tex`, `src/table/gd1.5_zero_shot_v2.tex`, `gd1.5_edge.tex`, `gd1.5_zero_shot_odinw13.tex`.

## 2. Core problem
Scale Grounding DINO for accuracy (Pro) and shrink it for edge devices (Edge) while keeping the open-set prompt interface.

## 4. The one key idea
Same dual-encoder-single-decoder with early (deep) fusion, but with a bigger backbone (ViT-L, EVA-02) and much more grounding data (Grounding-20M, over 20M images) for Pro. Edge fuses text only into the coarsest image level (P5), so the fusion cost drops enough to run on an NVIDIA Orin NX.

## 5. Mechanism (at most 5 steps)
1. Pro: Grounding DINO architecture + EVA-02 ViT-L backbone; "deep early fusion" (cross-attention between language and image before decoding) is kept (Sec. 3.1.1).
2. Early vs late fusion, per the paper: early fusion gives "higher detection recall and better bounding box precision" but "increased model hallucinations"; late fusion resists hallucination but has lower recall. They keep early fusion and add "a more comprehensive training sampling strategy, which increases the proportion of negative samples during training" (Sec. 3.1.1). No ratio is given.
3. Edge: an efficient feature enhancer. Cross-modality fusion only on P5; deformable self-attention -> vanilla self-attention; a cross-scale fusion module mixes in P3/P4; backbone EfficientViT-L1 (Sec. 3.1.2, Fig. "EFE").
4. Data: Grounding-20M, "over 20M grounding images ... collected from publicly available sources", with annotation pipelines and post-processing rules (Sec. 3.2). Composition not given.
5. Evaluation: LVIS is reported as fixed AP (Sec. 4.1). ODinW: "we performed prompt engineering to refine category names on datasets where their performance was particularly poor" (Sec. 4.1).

## 8. Key table (Table 1 `tab:gd-1.5-zero-shot` and Table `tab:edge`; zero-shot)
| Model | COCO AP | LVIS minival fixed AP (APr) | ODinW35 avg | ODinW13 avg |
|---|---|---|---|---|
| Grounding DINO 1.5 Pro (ViT-L, Grounding-20M) | 54.3 | 55.7 (56.1) | 30.2 | 58.7 |
| Grounding DINO 1.5 Edge (EfficientViT-L1, 800x1333) | 45.0 | 36.2 | - | - |
- Edge speed: 75.2 FPS TensorRT FP32 on A100 at 800x1333 (18.5 FPS PyTorch); 10.7 FPS on Orin NX at 640x640 (Table `tab:edge`, col FPS).
- The same Table 1 prints ODinW13 for the original Grounding DINO: Swin-T (O365,GoldG,Cap4M) 49.8; Swin-L (listed as O365,OID,GoldG) 56.9. These are the only primary-source ODinW-13 values for Grounding DINO in our sources.

## 10. Outline feedback
- ch 12 card 2: use this paper's Table 1 for Grounding DINO's ODinW-13 (49.8 T, 56.9 L) next to the ODinW-35 mean (22.3 T, 26.1 L). Caveats: the GD 1.5 table pairs Swin-L "O365,OID,GoldG" with ODinW35 26.1, which the original paper reports for the L checkpoint that also saw Cap4M, COCO and RefC. The per-dataset ODinW-13 values for Swin-L here (e.g. Thermal 78.4) differ from the original paper's Table 14 (72.02), so they come from a different checkpoint.
- ch 12/13: GD 1.5 admits prompt engineering of ODinW category names "where their performance was particularly poor". This is direct support for "the name is the failure" and a caveat for comparing its ODinW numbers.
- ch 13 item 4 / ch 8: the early-vs-late fusion trade-off (recall vs hallucination) is a primary-source statement worth one sentence in ch 8.
- ch 10 speed chart: Edge's FPS uses A100 TensorRT FP32 and an end-to-end setting with the text encoder (no language cache). It is not on YOLO-World's axis.
- Text says Pro improves "upon Grounding DINO by 4.2 AP" on ODinW35; 30.2 - 26.1 = 4.1. Cite table values.
