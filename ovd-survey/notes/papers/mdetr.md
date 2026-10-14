# MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding (BRANCH, short note)

## 1. Identity
- Title: "MDETR - Modulated Detection for End-to-End Multi-Modal Understanding"
- arXiv 2104.12763, read v2. v1 2021-04-26 (Mon, 26 Apr 2021 17:55:33 UTC). v2 2021-10-12 (Tue, 12 Oct 2021 00:49:54 UTC) (abs.html).
- Authors: Aishwarya Kamath (first), Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, Nicolas Carion (last). NYU CDS, Facebook AI Research, NYU Courant.
- Venue: ICCV camera-ready (src uses `iccv.sty` with `\iccvfinalcopy`; copy folder "MDETR camera ready ICCV_arXiv"); year not printed in the source; abs.html has no Comments line.
- Repo: https://github.com/ashkamath/mdetr (abstract).
- Source read: `src/main.tex` -> `parts/method.tex`, `parts/experiments.tex`, `appendix.tex`; `paper.pdf` for numbering.

## 2. Core problem
Multi-modal reasoning systems use a frozen, pre-trained detector with a fixed vocabulary as a black box, so they miss long-tail concepts in free-form text (abstract). MDETR trains the detector end to end, conditioned on the text query.

## 4. The one key idea
DETR whose encoder reads the image and the text together. Project CNN image features and RoBERTa token features into one space, concatenate them on the sequence axis, and run a joint "cross encoder"; the decoder's object queries then predict, for each box, which text tokens it refers to (not a class). This is early fusion and grounding-style supervision on 1.3M aligned image-text pairs.

## 5. Mechanism (5 steps)
1. Image -> ResNet-101 or EfficientNet-B3/B5 (frozen batchnorm) -> flatten + 2D positional embeddings. Text -> RoBERTa-base (12 layers, hidden 768, 12 heads). Modality-specific linear projections into a shared space; concatenate (Sec. 2.2.1, 3.1).
2. Joint transformer "cross encoder" over the concatenated sequence; DETR decoder with object queries (100 for detection, Sec. 3.2 GQA paragraph) cross-attends to the cross encoder output (Sec. 2.2.1).
3. Soft token prediction: max L = 256 tokens; each Hungarian-matched box predicts a uniform distribution over the token positions of its phrase (BPE token spans); unmatched queries predict "no object" (Sec. 2.2.2).
4. Contrastive alignment (Eq. 1-2): InfoNCE between decoder object embeddings o_i and cross-encoder token embeddings t_j, both directions, tau = 0.07, averaged. Matching cost and total loss = L1 + GIoU + soft-token CE + contrastive alignment (Sec. 2.2.2).
5. Pre-train 40 epochs on 1.3M aligned image-text pairs (images from Flickr30k, COCO, VG; about 200k images per Table 2 and 5 rows) on 32 V100, effective batch 64, about a week; then fine-tune per task (Sec. 3.1, App. hyperparameters).

Eq. 1 (TeX, verbatim): `l_{o} = \sum_{i=0}^{N-1} \frac{1}{\vert T^+_i \vert } \sum_{j \in T^+_i}  - \log \bigg( \frac{\exp(o_i^\top t_j / \tau)}{\sum_{k=0}^{L-1} \exp(o_i^\top t_k/\tau) } \bigg)`. Eq. 2 is the symmetric token-to-object version. N = max objects, L = max tokens, T^+_i = tokens object o_i should align to.

## 8. Key results
LVIS-v1 few-shot, Table 6 (box AP *fixed*, 5k-image minival never seen in training; ResNet-101; % of LVIS train images used for fine-tuning):
| Method | Data | AP | APr |
|---|---|---|---|
| Mask R-CNN | 100% | 33.3 | 26.3 |
| DETR | 100% | 17.8 | 3.2 |
| MDETR | 1% | 16.7 | 11.2 |
| MDETR | 10% | 24.2 | 20.9 |
- The MDETR 10% row (24.2 AP, 20.9 APr) is the "MDETR RN101" gray baseline in GLIP Table 3.
- MDETR 100% APr is 7.4 in Table 6 but "7.5" in the text (Sec. 3.2.1); cite the table.
- Zero-shot not possible: "By construction, our dataset has no training instances where there are zero boxes aligned to the text, biasing the model to always predict boxes for a given text. This prevents evaluating in a true zero-shot transfer setting" (Sec. 3.2.1). LVIS inference queries each class name separately: "about 10s/image on a GPU".
- Flickr30k any-box R@1 test (Table 3): MDETR-ENB5 84.3, MDETR-R101 83.4.

## 10. Outline feedback
- Ch 7 branch: MDETR's grounding loss is per-token soft CE plus contrastive alignment, the ancestor of GLIP's token-level target T'. GLIP Sec. 3.1 credits it ("Our grounding formulation is inspired by MDETR"). GLIP's [NoObj] footnote follows MDETR.
- MDETR fuses at the encoder (concatenated sequence), i.e. earlier than GLIP's head-level fusion. The "where text enters" diagram could place MDETR as "joint encoder over image + text tokens".
- The one-class-per-forward-pass LVIS protocol (10 s/image) is a concrete cost argument for prompts that pack many classes, which GLIP adopts.
- GoldG (GLIP's 0.8M) is MDETR's data minus COCO images; GoldG+ = MDETR's 1.3M.
