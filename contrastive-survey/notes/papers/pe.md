# PE: Perception Encoder: The best visual embeddings are not at the output of the network

## Identity
- arXiv 2504.13181, version read: v2 (v1 Thu 17 Apr 2025; v2 Mon 28 Apr 2025). Daniel Bolya ... Christoph Feichtenhofer (18 authors, Meta FAIR et al.).
- Venue: not stated. Code: https://github.com/facebookresearch/perception_models

## Key idea
A carefully scaled CLIP-loss (global vision-language contrastive) recipe alone yields strong general features, but the
best features for OCR/VQA, detection, depth and tracking sit in intermediate layers, not at the output. Two alignment
stages (language alignment, spatial alignment) draw them out.

## Config (Sec. 2.4 Model Training)
PE core image pretraining: 5.4B MetaCLIP-curated image-text pairs, 86B samples seen (58B for B and L), global batch
131K, progressive resolution 98 -> up to 448; then image+video fine-tuning. Baseline ablation recipe: batch 32K, 12B samples.

## Defining equation
Standard CLIP contrastive loss ("despite being pretrained with CLIP loss", Sec. 1); no new equation.

## Key number
PE core G (1.9B params, 448 px, 5.4B data): zero-shot ImageNet val 85.4; avg robustness 86.6 (Table 5, row "PEcore G", cols ImageNet val / Avg Class.).
