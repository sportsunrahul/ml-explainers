# GLIPv2 -- Unifying Localization and VL Understanding (BRANCH, short note)

## 1. Identity
- Title: "GLIPv2: Unifying Localization and VL Understanding"
- arXiv 2206.05836, read v2. v1 2022-06-12 (Sun, 12 Jun 2022 20:31:28 UTC). v2 2022-10-11 (Tue, 11 Oct 2022 23:27:03 UTC) (abs.html).
- Authors: Haotian Zhang (first; equal contribution with Pengchuan Zhang), ..., Jianfeng Gao (last). U. Washington, Meta AI, Microsoft, UCLA.
- Venue: NeurIPS 2022 (abs.html Comments).
- Repo: https://github.com/microsoft/GLIP (abstract; same repo as GLIP).
- Source read: `src/arxiv_new.tex` -> `sections/1_intro.tex`, `3_glip.tex`, `4_exps.tex`; `paper.pdf` for numbering.

## 2. Core problem
Localization (detection, segmentation) and VL understanding (VQA, captioning) are usually pre-trained in separate stages with separate heads, so they gain little from each other. GLIPv2 wants one grounded VL model that serves both, with all weights shared.

## 4. The one key idea
Keep GLIP's detection-as-grounding (S_ground = O P^T) and add an inter-image region-word contrastive loss: phrases from other image-text pairs in the batch become negatives, computed on features *before* fusion. GLIP's within-prompt task has few negatives ("in the order of 10's" under the 256-token limit); the batch version raises this "to the order of 1000's, with nearly negligible additional computational cost" (Sec. 3.2).

## 5. Mechanism (5 steps)
1. Architecture Pi (Eq. 1): dual encoder Enc_V (Swin), Enc_L (BERT-Base for T; pre-layernorm text transformer for B/H), fusion encoder Enc_VL = DyHead with GLIP's deep fusion; segmentation head = Hourglass network (Sec. 3.1, Sec. 4 setup).
2. Loss (Eq. 2): `\Lcal_{\text{GLIPv2}} = \underbrace{\Lcal_{\text{loc}} + \Lcal_{\text{intra}}}_{\Lcal_{\text{ground}}} + \Lcal_{\text{inter}} + \Lcal_{\text{mlm}}`. L_intra = loss(O P^T; T) on post-fusion features (Eq. 3) is GLIP's loss; L_mlm is BERT masked LM via a 2-layer MLP on P.
3. Inter-image loss (Eq. 4-5): batch similarity block [i,j] = pre-fusion O-ring^i (P-ring^j)^T; off-diagonal targets by label propagation (a "person" region is positive for every "person" phrase in detection-type texts in the batch, not in grounding sentences); bi-directional cross-entropy. Pre-fusion features are used because post-fusion features "would have seen the paired information".
4. Data: GLIPv2-T = GLIP-T's data (O365, GoldG, Cap4M). GLIPv2-B/H: FiveODs (2.78M: O365, COCO, OpenImages, VG, ImageNetBoxes), GoldG, CC15M+SBU (16M) with boxes from GLIP-L. Segmentation head pre-trained second, other weights frozen (Sec. 4).
5. Transfer: zero-shot, prompt tuning (tune only the pre-fusion prompt embedding P-ring; it changes text, visual and fused features because of deep fusion), or fine-tuning (Sec. 3.3).

## 8. Key results
Table 2 (direct evaluation = zero-shot unless gray; ODinW = average AP over 13 sets; LVIS-Det minival AP):
| Model | COCO (minival) | ODinW | LVIS minival |
|---|---|---|---|
| GLIP-T | 46.6 | 46.5 | 26.0 |
| GLIPv2-T | 47.3 | 48.5 | 29.0 |
| GLIPv2-H | 64.1 (gray, not zero-shot) | 55.5 | 50.1 (gray) |
- For GLIPv2-B/H, LVIS and Flickr30K training sets are in the pre-training data, so those numbers are not zero-shot (gray). COCO 61.9/64.1 for B/H are "artificially high due to some overlap between COCO-minival and VisualGenome-train" (Table 2 caption).
- Table 3 (ODinW, GLIPv2-T ablation): zero-shot 46.5 (L_loc + L_intra, i.e. GLIP-T) -> 48.4 (+ L_inter) -> 48.5 (+ L_mlm).
- Sec. 4.3: "one-shot GLIPv2-H (61.3) surpasses a all-shot fully supervised DyHead-T (60.8)".
- Cross-paper discrepancy: GLIPv2 Table 3 lists GLIP-T 1-shot fine-tuning as 51.3; GLIP Table 11 says 51.1+-0.1. GLIPv2 Table 2 lists GLIP-T COCO zero-shot 46.6 (the CC3M+SBU variant in GLIP Table 2; the Cap4M variant is 46.3).

## 10. Outline feedback
- Ch 7 branch: GLIPv2's delta is a loss, not an architecture change: text still enters at the DyHead fusion layers. One sentence: "GLIPv2 adds batch-wide negatives computed before fusion and an MLM loss; zero-shot ODinW 46.5 -> 48.5 and LVIS minival AP 26.0 -> 29.0 at Tiny scale."
- Its point that GLIP's within-image grounding has "only two negatives" in a 3-phrase caption (Sec. 1) is a crisp way to explain why GLIP mixes in 19 negative captions (GLIP App. C.1).
- Do not cite GLIPv2-B/H LVIS or COCO direct-evaluation numbers as zero-shot.
