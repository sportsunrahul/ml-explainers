# OV-DINO -- Unified Open-Vocabulary Detection with Language-Aware Selective Fusion (BRANCH, short note)

## 1. Identity
- Title: "OV-DINO: Unified Open-Vocabulary Detection with Language-Aware Selective Fusion"
- arXiv 2407.07844, read v2. v1 2024-07-10 (Wed, 10 Jul 2024 17:05:49 UTC); v2 2024-07-22 (abs.html).
- Authors: Hao Wang (first) ... Xiaodan Liang (last); 11 authors.
- Venue: none stated (abs.html Comments "Technical Report"; IEEE journal template).
- Repo: https://github.com/wanghao9610/OV-DINO (abstract).
- Source read: `src/texs/00_abstract.tex`, `03_method.tex`, `04_experiment.tex`, `src/tabs/lvis_results.tex`, `coco_results.tex`, `ablation_lasf.tex`, `hyper_parameters.tex`.

## 2. Core problem
Pre-training on detection + grounding + image-text data needs pseudo-labels for the image-text part, which bring noise. GLIP/Grounding-DINO-style fusion is not tuned to region-level alignment (abstract).

## 4. The one key idea
Text should not rewrite the encoder. OV-DINO keeps an image-only Transformer encoder. Text enters by picking the encoder tokens most similar to the prompt and then feeding those picked object embeddings into the decoder's content queries through zero-initialized tanh-gated cross-attention ("Language-Aware Selective Fusion", LASF). Image-text pairs need no pseudo-labels: each caption is one "category" with an image-sized box ("Caption Box", after Detic).

## 5. Mechanism (at most 5 steps)
1. Unified prompts: every category or phrase is wrapped in a CLIP template ("a photo of {category}."); captions stay as they are. BERT-base text embeddings are mean-pooled to one vector per prompt (sentence level), then linearly projected to D = 256 (Sec. 3.2, Sec. 4.2).
2. Up to 150 prompts per image: the positives plus "randomly selected negative texts from all other data sources" (Sec. 4.2; Table 3 "# of prompted text (C) 150"; 80 for COCO fine-tuning).
3. Swin-T, 4 scales (8x-64x), 6-layer encoder with no text fusion; language-aware query selection `E_so, E_sp = RankTop(E_enc x E_t^T)` takes the top Q = 900 tokens. `E_sp` initializes the anchors, and `E_so` feeds query fusion (Sec. 3.3).
4. Each of the 6 decoder layers: self-attn -> cross-attn to E_enc -> `Q + tanh(alpha_a) * Attn(q=Q, kv=E_so)` -> `Q + tanh(alpha_b) * FFW(Q)` -> FFW, with alpha initialized to zero (Sec. 3.3 equations). Classification = similarity of each query with each prompt embedding (Q x C).
5. Loss as DINO: sigmoid focal cls (weight 2.0) + L1 (5.0) + GIoU (2.0) + denoising loss; matching class cost 1.0. 24 epochs, batch 128, lr 2e-4, input [800, 1333] (Sec. 4.2, Table 3).

## 8. Key table (TABLE 4 p8 LVIS MiniVal zero-shot, and TABLE 5 p9 COCO; Swin-T)
| Model (pre-training data) | LVIS minival fixed AP | LVIS minival APr | COCO zero-shot AP |
|---|---|---|---|
| OV-DINO (O365, GoldG) | 39.4 | 32.0 | 50.6 |
| OV-DINO (O365, GoldG, CC1M) | 40.1 | 34.5 | 50.2 |
- No ODinW evaluation in this paper. LVIS uses Fixed AP; COCO uses standard AP (Sec. 4.1 "Pre-Training Data and Evaluation Metric").
- LASF ablation (TABLE 7 p10, O365-100K, LVIS minival AP): baseline 18.3, + Grounding-DINO-style "Typical-CMF" 18.9, + Later-LASF 19.2.

## 10. Outline feedback
- Good branch for ch 9 / the hero frame: OV-DINO moves text out of the encoder (no feature enhancer). It keeps only query selection plus gated injection into the decoder content queries. This is a "where text enters" contrast to Grounding DINO's phase A.
- Interlude negatives: OV-DINO states its prompt budget (150 prompts, positives + random negatives from other sources). Grounding DINO gives no number.
- Copy errors to avoid: OV-DINO TABLE 4 lists G-DINO-T (O365, GoldG, Cap4M) APr/APc/APf as 20.8/21.4/31.0. These are GLIP-T's values; Grounding DINO's own Table 3 has 18.1/23.3/32.7. Do not source Grounding DINO numbers from OV-DINO.
- OV-DINO TABLE 5 lists G-DINO-T pre-training as 50 epochs. Grounding DINO itself states no epochs, and MM-G's accounting uses 30. Treat both as third-party values.
- OV-DINO's own text-encoder lr is inconsistent: Sec. 4.2 says "0.1 times the base learning rate (specifically set to 1e-5)" with base 2e-4. TABLE 3 says "lr of text encoder 2e-5".
- OV-DINO's matching/loss weights (class cost 1, class loss 2) match Grounding DINO's Table 8 and not its Sec. 4.2 text. This is context for the Grounding DINO weight conflict; it does not resolve it.
