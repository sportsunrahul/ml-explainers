# Web-SSL (branch)

- Title: Scaling Language-Free Visual Representation Learning. arXiv 2504.01017, read v1 (2025-04-01).
- Authors: David Fan ... Saining Xie. Venue not stated (abs.html comment: project page).
- Key idea: train DINOv2-style SSL ("Web-DINO") and CLIP on the same 2B-image web set (MC-2B, MetaCLIP-curated) at 1B-7B parameters, and evaluate with VQA (Cambrian-1, 16 tasks) as well as classic benchmarks. With matched data, language-free SSL scales better.
- No new loss (uses off-the-shelf DINOv2 and CLIP recipes).
- Headline (abstract): "visual SSL models scale better than CLIP models in terms of data and model capacity, and visual SSL performance does not saturate even after scaling up to 7B parameters"; "At 5B parameters and above, DINO can exceed the Average VQA performance of CLIP" (Sec 3, category-specific paragraph). CLIP VQA "largely saturates after 3B parameters".
- Key number: Web-DINO ViT-7B, 8.0B samples seen, 224 res: IN1k linear 86.5, Average VQA 55.2 (Table 3 `tab:final_table`, third table in src). MetaCLIP ViT-G (12.8B seen, 224): IN1k linear 86.4, Average VQA 54.8.
- Note: table number is our reading from table order in src (pdftotext dropped captions).
