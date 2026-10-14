# DINO-X -- A Unified Vision Model for Open-World Object Detection and Understanding (BRANCH, short note)

## 1. Identity
- Title: "DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding"
- arXiv 2411.14347, read v3. v1 2024-11-21 (Thu, 21 Nov 2024 17:42:20 UTC); v2 2024-12-06; v3 2025-05-15 (abs.html).
- Authors: Tianhe Ren (first) ... Lei Zhang (last); 20 authors, IDEA Research.
- Venue: none stated (abs.html Comments "Technical Report").
- Repo: https://github.com/IDEA-Research/DINO-X-API ("Our demo and API will be released"; API, no weights stated).
- Source read: `src/sec/01_abstract.tex`, `03_model_architecture.tex`, `04_training_data.tex`, `05_model_evaluation.tex`, `src/table/grounding/dinox_pro_main_results.tex`, `dinox_edge_main_results.tex`.

## 2. Core problem
Long-tailed and prompt-less open-world detection: text alone fails for rare or hard-to-name objects, and users may not know what to ask for. The model should also give masks, keypoints and captions per object (abstract, Sec. 2.1.1).

## 4. The one key idea
Grounding DINO 1.5's early-fusion encoder-decoder, trained on Grounding-100M, with three prompt types (text, visual box/point from T-Rex2, and a prompt-tuned "universal object prompt" for prompt-free detection). Extra heads (mask, keypoint, language) are added on a frozen detector.

## 5. Mechanism (at most 5 steps)
1. Pro: EVA-02 ViT backbone, deep early fusion, language-guided query selection, L1 + GIoU box loss and contrastive classification "Similar to Grounding DINO" (Sec. 2.1.1, Box Head).
2. Text encoder changes from BERT to CLIP (MetaCLIP): "the BERT model is trained solely on text data, which limits its effectiveness for perception tasks requiring multimodal alignment" (Sec. 2.1.1, Text Prompt Encoder).
3. Data: Grounding-100M, "over 100 million images collected from the web"; T-Rex2 data "with some additional industrial scenario data" for visual-prompt pre-training; COCO, LVIS and V3Det excluded from stage 1 so the benchmarks are zero-shot (Sec. 3).
4. Two-stage training: stage 1 joint text-prompt detection, visual-prompt detection and segmentation. Stage 2 freezes the DINO-X backbone and trains the keypoint heads, the language head and the universal object prompt (prompt-tuning) (Sec. 3, Model Training).
5. Edge: EfficientViT-L2 backbone, same CLIP text encoder with precomputed prompt embeddings, feature and logit distillation from Pro (Sec. 2.1.2).

## 8. Key table (Table `tab:dino-x-pro-zero-shot` and the Edge table; zero-shot box AP)
| Model | COCO-val box AP | LVIS-minival box AP (APr) | LVIS-val box AP (APr) |
|---|---|---|---|
| DINO-X Pro (ViT-L) | 56.0 | 59.8 (63.3) | 52.4 (56.5) |
| DINO-X Edge (EfficientViT-L2, 800x1333) | 50.9 | 48.3 | - |
- No ODinW results in this paper. Fixed AP is not restated; the evaluation "Following Grounding DINO 1.5 Pro", which used fixed AP.
- Same Pro table, for context: Grounding DINO 1.5 Pro 55.7 (APr 56.1) and Grounding DINO 1.6 Pro 57.7 (APr 57.5) on LVIS-minival.

## 10. Outline feedback
- Family-tree endpoint for the Grounding DINO lineage (2023 -> 1.5 2024-05 -> DINO-X 2024-11). The text encoder changes from BERT to CLIP. This is a "what text encoder" change worth one line in ch 9 or ch-tree.
- ch 11 (visual prompts): DINO-X states why it adds visual prompts: "visual prompts enhance the model's detection capability in situations where text prompts fall short due to data scarcity and descriptive limitations" (Sec. 2.1.1). It also mentions "additional industrial scenario data" for visual-prompt training (Sec. 3).
- ch 12: DINO-X's "customized prompt" (prompt-tuning for "long-tailed, domain-specific" scenarios) is a middle option between a zero-shot prompt and full fine-tuning. The paper gives no number for it on a custom domain.
- DINO-X and Grounding DINO 1.5/1.6 are API-only (our reading of "demo and API will be released"). The page should say that their numbers cannot be reproduced from released weights.
