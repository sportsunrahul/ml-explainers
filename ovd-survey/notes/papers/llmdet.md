# LLMDet -- Learning Strong Open-Vocabulary Object Detectors under the Supervision of Large Language Models (BRANCH, short note)

## 1. Identity
- Title: "LLMDet: Learning Strong Open-Vocabulary Object Detectors under the Supervision of Large Language Models"
- arXiv 2501.18954, read v1. v1 2025-01-31 (Fri, 31 Jan 2025 08:27:31 UTC); only version (abs.html).
- Authors: Shenghao Fu (first) ... Wei-Shi Zheng (last); 8 authors, Sun Yat-sen University (emails).
- Venue: none stated (abs.html has no Comments; CVPR-style template, not a venue claim).
- Repo: https://github.com/iSEE-Laboratory/LLMDet (code, model and dataset released, abstract).
- Source read: `src/main.tex` (single file), lines 190-300 (method, Tables `tab:lvis_result`, `tab:odinw`), 299-330 (implementation).

## 2. Core problem
Region-level grounding labels are short phrases. They miss object details, relations and context. Can an open-vocabulary detector learn better region features if a language model must also write a long, detailed caption of the whole image from the detector's features? (abstract, Sec. 4).

## 4. The one key idea
Co-train a Grounding-DINO-family detector (MM-Grounding-DINO) with a small LLM. The LLM generates long image-level captions from the encoder's feature maps and short region-level phrases from each matched object query. The caption losses push language knowledge back into the detector. The LLM is discarded at inference, so it adds no cost.

## 5. Mechanism (at most 5 steps)
1. Data: GroundingCap-1M, 1120k samples, each (image, grounding text, boxes, detailed caption): COCO 210k, V3Det 166k, GoldG 437k (merged from 769k "to save computation and increase negatives"), LCS 307k with MM-GDINO pseudo-boxes. Captions average about 115 words (Sec. 3, Table `tab:dataset`).
2. Step 1: train only a projector from the detector's encoder p5 features into the LLM (LLaVA-OneVision-0.5b-ov) with caption loss (Sec. 4, Fig. `pipeline`).
3. Step 2, end to end: total loss `\mathcal{L} = \mathcal{L}_{align} + \mathcal{L}_{box} + \mathcal{L}_{lm}^{image} + \mathcal{L}_{lm}^{region}` (Sec. 4, the one `\begin{equation}` in the paper, no `\label`, so printed as (1)). Image-level input is p4 resized to 27x27 plus p5 at 20x20; region-level input is each positive (matched) object query, up to 16 regions, with new cross-attention layers in the LLM (Sec. 4, Sec. 5.1).
4. Frozen vs trained: the detector's vision backbone is frozen ("Since MM-GDINO is fully-pretrained, the vision backbone of MM-GDINO is frozen"); the rest of the detector, the projector and the LLM train (Sec. 5.1).
5. Budget: 150k iterations (about two epochs), batch 16, about two days on eight NVIDIA L20 GPUs (Sec. 5.1).

## 8. Key table (Table 2 `tab:lvis_result`, zero-shot fixed AP; Table 3 `tab:odinw`)
| Model | LVIS minival fixed AP | LVIS minival APr | ODinW13 | ODinW35 |
|---|---|---|---|---|
| MM-GDINO Swin-T (baseline, as listed by LLMDet) | 41.4 | 34.2 | 52.5 | 23.1 |
| LLMDet Swin-T | 44.7 | 37.3 | 52.1 | 23.8 |
- LLMDet Swin-L: LVIS minival fixed AP 51.1, APr 45.1 (same table).

## 10. Outline feedback
- Late branch on the Grounding DINO line (2025): text enters training as an extra caption-generation signal, not as a new fusion point. This is a fourth answer to "where text enters": as a training-time teacher that is dropped at inference. Worth one sentence in ch-tree.
- ch 12 ODinW: LLMDet's ODinW rows reuse other papers' values. Its Grounding-DINO Swin-T (51.4 / 22.7) matches MM-G's re-evaluation. Its MM-GDINO (52.5 / 23.1) does not match MM-G's own Table 6 (53.3 / 28.4 for c3). The gain on ODinW13 is negative (52.1 vs 52.5), so do not chart LLMDet as an ODinW improvement.
- The LVIS table here is explicitly fixed AP. Grounding DINO's own 27.4 is placed in the same column, though the original paper does not say it is fixed AP. Mixed-metric risk for any LVIS bar chart across the family.
