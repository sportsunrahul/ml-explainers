# RegionCLIP -- Region-based Language-Image Pretraining (BRANCH, short note)

## 1. Identity
- Title: "RegionCLIP: Region-based Language-Image Pretraining"
- arXiv 2112.09106, read v1 (only version). v1 2021-12-16 (Thu, 16 Dec 2021 18:39:36 UTC) (abs.html).
- Authors: Yiwu Zhong (first), Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, Jianfeng Gao (last).
- Venue: abs.html says "Technical report"; README says CVPR 2022 (code_only).
- Repo: https://github.com/microsoft/RegionCLIP
- Read: `src/method.tex`, `src/experiments.tex`, README.md.

## 2. Core problem
CLIP matches a whole image to a caption, so applying it to image regions "leads to poor performance due to a domain shift" (abstract). Region-text pairs with labels are scarce.

## 4. The one key idea
Make pseudo region-text pairs at scale: propose regions with an RPN, score each region against prompt sentences for a large concept pool using CLIP itself as teacher, link each region to its best concept, then pretrain a region-level visual encoder on these pairs (contrastive + distillation). Use that encoder as the detector backbone.

## 5. Mechanism (5 steps)
1. Concept pool: parse nouns from captions (COCO Cap or CC3M) with a language parser, drop concepts with frequency below 100 -> 4,764 (COCO) / 6,790 (CC3M) concepts; fill prompts ("A photo of a kite"), encode with the CLIP text encoder L (Sec. 3.2, 4).
2. Regions: an RPN trained on LVIS base boxes (no labels used) proposes 100 regions per image; RoIAlign pools features (Sec. 3.2, 4).
3. Pseudo-labels: the teacher V_t (CLIP RN50) pools each region's feature, cosine with every concept, the argmax concept becomes the region's text (Sec. 3.2).
4. Pretraining: the student V (initialized from V_t) is trained with L = L_cntrst (region-concept contrastive over the batch, tau = 0.01) + L_dist (KL to the teacher's soft concept distribution) + L_cntrst-img (image-caption contrastive). SGD, batch 96 images, 600k iterations, CC3M (Sec. 3.2, 4).
5. Transfer: initialize a Faster R-CNN R50-C4 (Detectron2) backbone with V; classify regions by cosine with class-name text embeddings; background is a fixed all-zero embedding with weight 0.2 (COCO) / 0.8 (LVIS); focal scaling gamma = 0.5 on COCO; 1x schedule (Sec. 3.3, 4).

## 8. Key table
Table 1 (COCO, AP50, 48/17 split) and Table 2 (LVIS v1, 866/337 split). Our pick, rows pretrained on CC3M:
| Benchmark | Metric | RegionCLIP RN50 | RegionCLIP RN50x4 | Comparison in the same table |
|---|---|---|---|---|
| COCO generalized | Novel AP50 | 31.4 | 39.3 | OVR 22.8 |
| LVIS | APr | 17.1 | 22.0 | ViLD* RN152-FPN 19.8 |
- LVIS metric: the paper does not say box or mask. The ViLD* rows (16.7 / 26.5 / 34.2 / 27.8 and 19.8 / 27.1 / 34.5 / 28.7) equal ViLD's Table 8 box AP for ViLD-ensemble R50 and R152. So the table is box AP (our reading). Do not compare 22.0 with ViLD's mask 16.1.
- RegionCLIP describes ViLD* as "16x+Copy-paste" training; RegionCLIP uses "1x+Standard".
- Abstract claims +3.8 AP50 (COCO novel) and +2.2 AP (LVIS novel) over the state of the art.

## 10. Outline feedback
- Ch. 4 branch paragraph is right ("region-text pretraining on caption-derived pseudo pairs"). Add that CLIP itself labels the regions (the teacher picks the concept), and the student is a CLIP image encoder adapted to regions, not a detector trained from scratch as in ViLD.
- Any APr comparison with ViLD must use ViLD's box AP (Table 8), not Table 3 mask AP.
- Venue: only the README says CVPR 2022; the arXiv page says "Technical report".
