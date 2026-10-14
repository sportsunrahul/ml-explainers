# Branch papers (ch-efficient, ch-family)

Short notes per the brief, branch depth: identity, delta, mechanism, the numbers the page needs.
Sources read: `sources/<name>/latex/**/*.tex` (tables and equations verbatim), `abs.html` (dates).
All speed and quality numbers are each paper's OWN setup. Do not put them in one comparable column
unless the row says the paper itself compares them.

## Cross-paper chart table (ch-efficient)

Each row is one paper's own report. "SAM ref" is SAM as that paper reports it. Hardware differs per row.

| Paper | Student encoder / model params | SAM ref params (as reported) | Speed (own setup) | Quality vs SAM (metric, dataset) |
|---|---|---|---|---|
| MobileSAM | encoder 5.78M (TinyViT); whole 9.66M | ViT-H encoder 632M | encoder ~8ms vs ~452ms, "single GPU" (type not stated); whole pipeline 12ms (Tab. 6) | mIoU vs SAM-ViT-H masks: 0.7447 (Tab. 4, best row); prompt type and dataset not stated for Tab. 4 |
| MobileSAMv2 | (reuses ViT-H, TinyViT or EfficientViT-L2 encoder) | not reported | mask decoder 97ms (object-aware, max 320 prompts) vs 6464ms (64x64 grid) (Tab. 2), hardware not stated | mask AR@1000 on LVIS: 59.3 vs SAM 59.2 (Tab. 3); avg over AR@10/100/1000: 42.5 vs 38.9 (Tab. 4) |
| EfficientSAM | Ti 10M, S 25M ("Params (M)", whole model by our reading) | SAM 636M | throughput 54 / 47 vs 2 img/s, one ViTDet box per image, single NVIDIA A100 (Tab. 8) | zero-shot instance seg AP with ViTDet-H boxes, COCO: S 44.4 vs SAM 46.5 (Tab. 5) |
| EfficientViT-SAM | L0 34.8M ... XL1 203.3M (whole model) | SAM-ViT-H 641.1M | throughput 762 (L0) ... 182 (XL1) vs 11 img/s, single A100, TensorRT, fp16 (Tab. 1) | zero-shot COCO mAP with ViTDet boxes: L2 46.6 vs 46.5; XL1 47.8 (Tab. 1, Tab. 3) |
| EdgeTAM | not reported | not reported | 15.7 FPS iPhone 15 Pro Max (CoreML, CPU+NPU) vs SAM 2-B+ 0.7; A100 (torch compile) 150.9 vs 64.8 (Tab. 2) | J&F, SA-V val: 72.3 vs SAM 2.1-B+ 76.8 (Tab. 2) |
| EfficientSAM3 (v1) | students 0.7M ... 21M, "approximate" (Tab. 1) | not reported | no speed results in v1 | no results in v1 ("results will be reported in a future revision") |
| HQ-SAM (adds, not shrinks) | +5.1M learnable on SAM-L | SAM-L "1191" M learnable (as printed) | 4.8 vs 5.0 FPS (Tab. 1), GPU for FPS not stated | mBIoU on DIS val: 70.4 vs 52.8 (Tab. 2) |

Resolution notes: EfficientViT-SAM-L trains/infers at longest side 512, XL at 1024 (Sec. 3 Training); EdgeTAM 1024x1024;
MobileSAM and EfficientSAM use SAM's 1024 input (our reading; not restated as a speed condition).

Ratios printed in the papers (use these, do not compute new ones): MobileSAM "more than 60 times smaller", encoder
"100 times" fewer params; EfficientViT-SAM "48.9x" (= L2 538 / SAM 11 img/s, our reading) and "17 to 69 times";
EdgeTAM "22x faster than SAM 2" on iPhone (= 15.7 / 0.7, our reading); MobileSAMv2 "at least 16 times" on the decoder;
HQ-SAM "less than 0.5%" params, "96%" of SAM speed.

---

## MobileSAM -- Faster Segment Anything: Towards Lightweight SAM for Mobile Applications
1. Identity: 2306.14289v2 (read v2, 2023-07-01); v1 2023-06-25. Chaoning Zhang ... Choong Seon Hong. Repo github.com/ChaoningZhang/MobileSAM.
2. Problem: SAM's ViT-H encoder (632M) is too heavy for phones; the decoder (<4M) is already light.
3. Prior: retraining SAM with a small encoder (coupled distillation through the decoder, SAM recipe) costs 128 GPUs for days and trains poorly at low compute (Sec. 3).
4. Key idea (delta from SAM): decoupled distillation. Distil only the image encoder, by matching ViT-H's image embedding; the unchanged SAM decoder then works with the student as is.
5. Mechanism:
   1. Precompute ViT-H embeddings for 1% of SA-1B (saved once, following TinyViT).
   2. Train TinyViT (4 stages, last downsampling stride set to 1 so the output matches 64x64) with an MSE loss to those embeddings. No decoder, no prompts, no focal/dice loss in this step.
   3. Plug the student into SAM's prompt encoder and mask decoder unchanged (decoder fine-tuning "omitted in this version").
6. Equations: none printed (loss is "a simple MSE loss" on the image embedding, Sec. 3).
7. Config: 1% of SA-1B, 8 epochs, single GPU, "less than a day" (Sec. 4.1). Preliminary ViT-B test (Tab. 2): coupled 128 GPUs / batch 128 / 180k it / 11M images -> mIoU 0.72; decoupled 2 GPUs / batch 4 / 55k it / 11K images -> 0.75.
8. Results: Tab. 1 SAM encoders ViT-H 632M, ViT-L 307M, ViT-B 86M. Tab. 3 encoder 632M vs 5.78M, ~452ms vs ~8ms, "single GPU". Abstract: ~10ms per image (8ms encoder + 4ms decoder). Tab. 6: MobileSAM 9.66M / 12ms vs FastSAM 68M / 64ms. Tab. 4 ablation mIoU vs SAM: 0.7057, 0.7286, 0.7447. Tab. 7: mIoU vs SAM with fg+bg points 100-500 px apart: MobileSAM 0.71-0.74, FastSAM 0.27-0.41.
9. Limitations / inconsistencies: GPU type never named. Text says 8ms+4ms = "10ms" while Tab. 6 says 12ms; FastSAM 40ms in text vs 64ms in Tab. 6 and "4 times" vs "5 times"; "1%" SA-1B (Sec. 4.1) vs "0.1% (11k)" (Sec. 3.2, ViT-B preliminary). Quality is agreement with SAM, not ground truth. EfficientSAM (Tab. 5) later reports MobileSAM at 38.7 COCO AP vs SAM 46.5 under ViTDet boxes.
10. Outline feedback: "distil a small encoder to match SAM's embedding" is correct. Add "agreement with SAM masks, not GT" to the chart label.

## MobileSAMv2 -- Faster Segment Anything to Everything
1. Identity: 2312.09579v1, 2023-12-15. Chaoning Zhang ... Choong Seon Hong. Same repo as MobileSAM.
2. Problem: "segment everything" (the automatic mask generator) is slow because of the decoder, not the encoder: a 64x64 grid gives 12288 masks, then filtering.
3. Prior: SAM's grid search: redundant prompts, then NMS/filters (Tab. 1 beside Fig. 1: decoder ~6400ms at 64x64 vs encoder ~450ms).
4. Key idea: object-aware prompt sampling. A YOLOv8 object-discovery model (trained on a small SA-1B subset, class head dropped) proposes boxes; after NMS, the boxes are the prompts. One mask per box, no post-filtering.
5. Mechanism: 1) YOLOv8 boxes, 2) NMS, 3) up to 320 box prompts to SAM's decoder (single-mask output), 4) optionally swap the encoder for a distilled one (TinyViT, EfficientViT-L2).
7. Setup: zero-shot object proposal on LVIS, mask AR@K (SAM's protocol).
8. Results: Tab. 2 decoder time: grid 32x32 1616ms, 64x64 6464ms; object-aware 97ms. Tab. 3 AR@1000: SAM 64x64 59.2, MobileSAMv2 (320 boxes) 59.3. Tab. 4 average: 42.5 vs 38.9. Tab. 5 encoders: ViT-H 59.3, TinyViT 51.1, EfficientViT-L2 56.3 ("around 20ms" vs ViT-H "more than 400ms", Sec. 5).
9. Limitations: hardware for timings not stated.
10. Outline feedback: not in the ch-efficient intuition line; it attacks the automatic-mask-generator cost (ch-engine's grid), worth one line on the family tree.

## EfficientSAM -- Leveraged Masked Image Pretraining for Efficient Segment Anything
1. Identity: 2312.00863v1, 2023-12-01. Yunyang Xiong ... Vikas Chandra (Meta). Venue not stated in source.
2. Problem: small SAMs (MobileSAM, FastSAM) lose several AP versus SAM.
3. Prior: MobileSAM distils the embedding directly from a random/ImageNet init; MAE pretrains on pixels.
4. Key idea (SAMI): MAE-style masked pretraining where the reconstruction target is SAM ViT-H's feature map, not pixels. Then fine-tune the small encoder with SAM's decoder on SA-1B.
5. Mechanism:
   1. ImageNet-1K (1.2M images, no labels), 224x224 crops, mask 75% of tokens.
   2. Small ViT encoder sees unmasked tokens; a cross-attention decoder (queries = masked tokens; 8 blocks, 512 dims) fills the masked positions.
   3. Linear head maps merged features to SAM's feature dimension; MSE to SAM ViT-H features on all tokens. 400 epochs.
   4. Discard the decoder; put the encoder (ViT-Ti / ViT-S) in SAM with SAM's mask decoder; fine-tune encoder+decoder on SA-1B for 5 epochs.
6. Equation (Eq. `equ:recon`, Sec. 3):
   `L_{W_e, W_d, W_{\theta}} = \frac{1}{N} \cdot \sum_{j = 1}^N ||f^{\text{sam}}(\mathbf{x}) - f^h(\mathbf{x})||^2,`
   with `f^h(\mathbf{x}) = h^{\theta}(\phi(g^{e}\{\mathbf{x}_i\}_{i \in \mathcal{U}} \oplus g^{d}\{\mathbf{x}_i\}_{i \in \mathcal{M}}))`.
   Gloss: f^sam = frozen SAM ViT-H encoder (target); g^e = small encoder on unmasked set U; g^d = decoder on masked set M; (+) merge, phi reorder, h^theta linear head; N tokens; l2 norm.
7. Config: SAMI batch 4096, AdamW lr 2.4e-3, 40-epoch warmup, V100s. SA-1B fine-tune: 5 epochs, batch 128, lr 4e-4 (Sec. 4).
8. Results: Tab. 8 Params/throughput: SAM 636M, 2 img/s; Ti 10M, 54; S 25M, 47 (one ViTDet box, single A100). Tab. 5 zero-shot instance seg, ViTDet-H boxes, AP COCO / LVIS: SAM 46.5 / 44.7; MobileSAM 38.7 / 34.4; FastSAM 37.9 / 34.5; Ti 42.3 / 39.9; S 44.4 / 42.3. Tab. 4 single-box mIoU COCO: SAM 78.4, S 76.9 ("only 1.5 mIoU lower").
9. Limitations: none stated beyond the gap to SAM. Text says "9.8M" for Ti where Tab. 8 says 10.
10. Outline feedback: "pretrain the small encoder to reconstruct SAM features" is correct; note it is MAE-style with SAM features as the target and masking at 75%.

## EfficientViT-SAM -- Accelerated Segment Anything Model Without Accuracy Loss
1. Identity: 2402.05008v2 (read v2, 2024-05-16); v1 2024-02-07. Zhuoyang Zhang ... Song Han (MIT Han Lab). Repo github.com/mit-han-lab/efficientvit.
2. Problem: earlier fast SAMs trade AP for speed.
3. Prior: MobileSAM/EdgeSAM/EfficientSAM lose 2-8 COCO AP (Tab. 3).
4. Key idea: replace the encoder with EfficientViT (ReLU linear attention + convs, multi-scale fusion), distil it from SAM-ViT-H with L2 on the embedding, then train end to end on SA-1B. Keeps SAM's prompt encoder and mask decoder.
5. Mechanism: 1) EfficientViT backbone (5 stages, convs early, EfficientViT modules in last two; fuse last three stages; fused-MBConv neck); 2) L2 distillation of SAM-ViT-H image embedding; 3) end-to-end on SA-1B, 2 epochs, batch 256, box or 1-10 points, focal:dice 20:1, best of 3 masks; L at 512, XL at 1024 longest side.
8. Results: Tab. 1 (A100, TensorRT, fp16) Params / MACs / img/s / COCO mAP: SAM-ViT-H 641.1 / 2973 / 11 / 46.5; MobileSAM 9.8 / 39 / 278 / 38.7; EdgeSAM 9.6 / 20 / 449 / 42.1; EfficientSAM 25.3 / 247 / 183 / 44.4; L0 34.8 / 35 / 762 / 45.7; L1 47.7 / 49 / 638 / 46.2; L2 61.3 / 69 / 538 / 46.6; XL0 117.0 / 185 / 278 / 47.5; XL1 203.3 / 322 / 182 / 47.8. Headline "48.9x" without mAP drop (abstract, intro).
   This is the one paper that puts several efficient SAMs in one column under one setup (Tab. 1). It is safe to chart that table as is.
9. Limitations: on LVIS the L-series is below SAM (Tab. 3: SAM 44.2, L0 41.8, L1 42.1, L2 42.7); XL1 is above (44.4). So "without accuracy loss" holds for XL1 on COCO and LVIS, and for L2 (the 48.9x model) on COCO only.
10. Outline feedback: "redesign the encoder" is correct.

## EdgeTAM -- On-Device Track Anything Model
1. Identity: 2501.07256v1, 2025-01-13. Chong Zhou ... Bilge Soran (Meta). 
2. Problem: SAM 2 runs ~1 FPS on iPhone 15 Pro Max; shrinking the image encoder is not enough.
3. Prior: efficient SAMs compress only the image encoder; for SAM 2 the memory attention is also a latency bottleneck (Fig. 2 benchmark).
4. Key idea (delta from SAM 2): compress each stored frame memory before memory attention with a 2D Spatial Perceiver (fixed learnable latents), plus feature distillation from SAM 2 in both image and video training stages.
5. Mechanism:
   1. Image encoder RepViT-M1 (ImageNet init), 1024x1024 input.
   2. Each memory feature map M_t (C x H x W) is summarised by 256 global latents (attend to the whole map, Perceiver-style) plus 256 2D latents (each attends to one non-overlapping window, positional embedding added at output), concatenated.
   3. Two memory attention blocks attend to the compressed memories (memory bank 7 frames, 16 object pointers, as SAM 2).
   4. Distillation: feature alignment with SAM 2 teacher as an auxiliary loss during image pre-training and video training; no inference cost.
8. Results: Tab. 2 J&F: MOSE 70.0, DAVIS 2017 87.7, SA-V val 72.3, test 71.7; FPS A100 150.9, iPhone 15.7 (SAM 2-B+ 64.8 / 0.7; SAM 2.1-B+ 64.1 / 0.7). Tab. 1 SA-23 1-click mIoU 55.5 vs SAM 58.1; 5-click 81.7 vs 81.3; iPhone 40.4 FPS image mode vs SAM 2 1.3. Tab. 3a (1/3 schedule): baseline 2.5 FPS -> 2D Perceiver 15.7 FPS; distillation +1.3 / +3.3 J&F on SA-V val / test. Intro: memory attention sped up "8x".
9. Limitations: A100 speedup is small (SM use under 50%, CPU/IO bound; App.). Intro says "6.4x" while Sec. 4 says "6.3x" for the same ablation. No parameter count printed.
10. Outline feedback: chart needs FPS, not params, for EdgeTAM (no params in paper). "do the same for video memory" is correct.

## EfficientSAM3 -- Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1, 2, and 3
1. Identity: 2511.15833v1, 2025-11-19. Chengxi Zeng ... Aaron Zhang.
2. Problem: SAM 3 (shared backbone, DETR-style detector, dense-memory tracker) is too heavy for devices.
4. Key idea: three-stage distillation (PHD). What comes from which SAM (our reading of Sec. 3): SAM 1's SA-1B data and SAM-style prompt encoder/mask decoder interface (EdgeSAM-style prompt-in-the-loop); SAM 2's SA-V data and EdgeTAM's Perceiver memory in place of the dense memory; SAM 3 is the teacher throughout and its SA-Co PCS data drive stage 3.
5. Mechanism:
   1. Stage 1 (SA-1B): student backbone (RepViT / TinyViT / EfficientViT) mimics the SAM 3 encoder; L_total = L_task + l1 L_feat + l2 L_mask.
   2. Stage 2 (SA-V): replace dense memory with a 2D Spatial Perceiver (K=128 latents); encoder frozen.
   3. Stage 3 (SA-Co): fine-tune Perceiver + mask decoder on noun-phrase / exemplar prompts; encoder frozen by default.
6. Equations (Sec. 3.2, unnumbered): `\mathcal{L}_{\text{feat}} = \big\| \text{Proj}_S(F^S) - F^T \big\|_2^2,` and
   `\mathcal{L}_{\text{mask}} = \sum_i \mathcal{L}_{\text{Dice}}(M_i^S, M_{\sigma(i)}^T) + \mathcal{L}_{\text{Focal}}(M_i^S, M_{\sigma(i)}^T).` (sigma = bipartite matching to teacher masks.)
8. Results: none. v1 footnote: "Coming Soon ... results will be reported in a future revision." Tab. 1 lists nine students with "approximate" params: ES-RV-S/M/L 5.1M/6.8M/8.2M, ES-TV-S/M/L 5.4M/11M/21M, ES-EV-S/M/L 0.7M/4.8M/15M.
9. Limitations: no measurements; the LaTeX `sec/0_abstract.tex` is the CVPR template (real abstract is in main.tex).
10. Outline feedback: show as a recipe card only. Do not chart it. Say "v1 reports no results".

## HQ-SAM -- Segment Anything in High Quality
1. Identity: 2306.01567v2 (read v2, 2023-10-23); v1 2023-06-02. Lei Ke ... Fisher Yu. NeurIPS 2023 (style file `neurips_2023`, final). Repo github.com/SysCV/SAM-HQ.
2. Problem: SAM's masks are coarse on thin or intricate structures (kite lines, chairs).
4. Key idea (delta from SAM): freeze all of SAM; add one learnable HQ-Output Token (1x256) to the decoder and a global-local feature fusion; train only those on 44K fine masks.
5. Mechanism:
   1. HQ-Output Token is concatenated with SAM's 4 output tokens and prompt tokens and passes through SAM's frozen 2-layer decoder.
   2. HQ-Features: early ViT feature (after first global attention block; block 6 of 24 for ViT-L) + final ViT feature (both 64x64, upsampled by transposed conv to 256x256) + SAM decoder mask feature (256x256), fused by simple convs.
   3. A new 3-layer MLP turns the token into a dynamic kernel; point-wise product with HQ-Features gives the mask.
   4. Inference: sum SAM's mask logits and HQ logits at 256x256, upsample to 1024.
7. Config: HQSeg-44K (44,320 masks from DIS, ThinObject-5K, FSS-1000, ECSSD, MSRA-10K, DUT-OMRON; >1000 classes); 12 epochs, lr 0.001, batch 32, 8 RTX 3090, 4 hours, 16.6K iterations.
8. Results: Tab. 1 (ViT-L): learnable params SAM 1191 vs HQ-SAM 5.1 (M); FPS 5.0 vs 4.8; memory 7.6G both. Tab. 2 DIS mBIoU 52.8 -> 70.4; four-dataset average mIoU 79.5 -> 89.1, mBIoU 71.1 -> 81.8. Tab. 7 COCO (FocalNet-DINO boxes, ViT-L): AP 48.5 -> 49.5, AP_B 33.3 -> 34.4.
9. Limitations: the "1191M" SAM learnable-param figure is as printed (MobileSAM Tab. 1 gives the SAM ViT-L encoder as 307M; flag, do not reuse). The "less than 0.5%" ratio is 5.1/1191; it stands or falls with that denominator. Zero-shot gains on COCO are about 1 AP.
10. Outline feedback: "tiny add-on for fine structures" correct. Use "5.1M" learnable params; quote "less than 0.5%" only as the paper's claim (see item 9). Do not compute a ratio of your own.

## MedSAM -- Segment Anything in Medical Images
1. Identity: 2304.12306v3 (read v3, 2024-04-01); v1 2023-04-24. Jun Ma ... Bo Wang. Nature Communications 15, 654 (2024) (abs.html).
2. Problem: SAM does poorly on medical targets with weak boundaries and low contrast.
4. Key idea (delta from SAM): full fine-tuning of SAM ViT-B on a large curated medical set with box prompts only.
5. Mechanism: 1) init from SAM ViT-B; 2) freeze the prompt encoder (boxes already encoded well); 3) train image encoder (89,670,912 params) and mask decoder (4,058,340) with Dice + cross-entropy (unweighted sum); boxes from GT with 0-20 px jitter.
7. Config: 1,570,263 image-mask pairs, 10 modalities, over 30 cancer types; 20 A100 (80G), 150 epochs, batch 160, AdamW lr 1e-4.
8. Results: evaluation on 86 internal and 60 external tasks, median DSC vs SAM, U-Net, DeepLabV3+ (Figs. 2-3). Example: nasopharynx cancer (external) median DSC 87.8%, +52.3 over SAM (text, Sec. 2).
9. Limitations: modality imbalance (CT, MRI, endoscopy dominate); vessel-like branching structures where a box is ambiguous.
10. Outline feedback: "MedSAM fine-tuning" correct; specify "encoder + decoder, prompt encoder frozen, boxes only".

## MedSAM2 -- Segment Anything in 3D Medical Images and Videos
1. Identity: 2504.03600v1, 2025-04-04. Jun Ma ... Bo Wang.
- One line: full-model fine-tune of SAM2.1-Tiny on over 455,000 3D image-mask pairs and 76,000 video frames; human-in-the-loop annotation study claims manual cost cut by more than 85% (abstract).
- Why Tiny: SAM2.1 sizes showed no significant difference in median DSC on their 3D tests (Sec. 2).

## SAM-Adapter -- SAM Fails to Segment Anything? Adapting SAM in Underperformed Scenes
1. Identity: 2304.09148v3 (read v3, 2023-05-02); v1 2023-04-18. Tianrun Chen ... Ying Zang.
2. Problem: SAM fails on camouflaged objects, shadows, polyps.
4. Key idea: keep SAM's ViT-H encoder frozen; add small adapters that inject task-specific prompts into each transformer layer; tune the mask decoder (no prompts given to it).
6. Equation (Eq. `get_context`): `P^i = {\rm MLP}_{up}\left({\rm GELU}\left({\rm MLP}_{tune}^i\left(F_i\right)\right)\right)` -- MLP_tune^i per adapter, MLP_up shared; F_i = task information (e.g. texture or frequency features), optionally a weighted sum `F_i = \sum_{1}^{N}w_jF_j`.
7. Config: 4 A100; COD 20 epochs, shadow 90, polyp 120; BCE+IoU loss (balanced BCE for shadow).
8. Results: Tab. 1 COD10K S_alpha SAM 0.783 -> 0.883 (CHAMELEON 0.727 -> 0.896; CAMO 0.684 -> 0.847). Tab. 2 ISTD shadow BER SAM 40.51 -> 1.43. Tab. 3 Kvasir mDice 0.778 -> 0.850.
9. Note: the "SAM" baseline is run with no point prompt (whole-image box or everything mode, Fig. 2 caption), so it is not prompted SAM at its best.
10. Outline feedback: belongs in ch-family (adaptation) and ch-inspection (concealed / low-contrast). Folder tag says "inspection".

## Grounding DINO (only what Grounded SAM uses)
1. Identity: 2303.05499v5 (read v5, 2024-07-19; ECCV template); v1 2023-03-09. Shilong Liu ... Lei Zhang.
- Text in, boxes out: an open-set DETR-style detector (DINO + grounded pre-training). Image and text backbones, feature enhancer, language-guided query selection (top N_q = 900 image tokens by max similarity to the text tokens), cross-modality decoder; each box is tied to a phrase.
- Headline: 52.5 AP zero-shot COCO detection (no COCO training data), Grounding DINO L, Swin-L, trained on O365, OI, GoldG (Tab. 2; abstract).

## Grounded SAM -- Assembling Open-World Models for Diverse Visual Tasks
1. Identity: 2401.14159v1, 2024-01-25. Tianhe Ren ... Lei Zhang (IDEA).
2. Key idea: compose, do not train: text -> Grounding DINO boxes -> each box as a SAM box prompt -> masks.
3. Enables: text-to-mask; automatic labelling with RAM or BLIP tags as the text; inpainting/editing with Stable Diffusion; promptable 3D human mesh with OSX (Sec. 2).
4. Number: SGinW zero-shot (25 datasets) mean AP 48.7 for Grounding DINO-Base + SAM-Huge; 49.6 with HQ-SAM (Tab. 1, both marked "tested by the SAM-HQ team").
5. Timeline note: the paper is from 2024-01, but the code project predates it (MobileSAM, June 2023, already cites "Grounded SAM" as GroundedSegmentAnything2023). On the tree, date by the paper (2024-01-25) or label "repo 2023, paper 2024".

## SEEM -- Segment Everything Everywhere All at Once
1. Identity: 2304.06718v4 (read v4, 2023-07-11); v1 2023-04-13. Xueyan Zou ... Yong Jae Lee.
- One line: concurrent with SAM; one decoder takes text, visual, and memory prompts and returns masks with semantic labels.
- Prompt types: visual prompts (points, boxes, scribbles, masks/polygons, and a referred region from another image), text prompts, memory prompts (previous mask, for multi-round interaction); text and visual prompts share one joint space and can be composed.
- One number: trained on COCO+LVIS (0.12M images) vs SAM's 11M (Tab. 1, training-data column); text says SAM uses "x100 more segmentation data".

## Track Anything (TAM)
1. Identity: 2304.11968v2 (read v2, 2023-04-28); v1 2023-04-24. Jinyu Yang ... Feng Zheng.
- Pipeline (Sec. 3, no training): 1) SAM masks the object from user clicks on one frame; 2) XMem propagates the mask; 3) when quality is poor, XMem's probes and affinities become SAM point prompts and the XMem mask a mask prompt, and SAM refines it; 4) the user corrects with more clicks.
- Number: DAVIS-2016 val J&F 88.4, DAVIS-2017 test-dev 73.1 from click initialisation; XMem from a mask gets 92.0 / 81.2 (Tab. 1).

## SAM2Long -- Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree
1. Identity: 2410.16268v3 (read v3, 2025-07-29); v1 2024-10-21. Shuangrui Ding ... Jiaqi Wang.
- One line: training-free; keep P memory pathways; each frame each pathway branches into SAM 2's 3 masks; keep the top P by cumulative log IoU; output the best pathway at the end. Memory bank per pathway keeps only confident, unoccluded frames.
- Equation (unnumbered, Sec. 3.2): `S_{p,k}[t] = S_p[t-1] + \log(\text{IoU}_t^{p,k} + \epsilon), \quad \text{for } k = 1, 2, 3,`
- Number: SA-V test J&F 80.8 with SAM2Long-L, +5.3 over SAM 2-L (Tab. 1); average +3.7 over 12 comparisons (abstract). P=3: 19 vs 22 FPS on an RTX 3090 (Tab. 6).

## SAMURAI -- Adapting SAM for Zero-Shot Visual Tracking with Motion-Aware Memory
1. Identity: 2411.11922v2 (read v2, 2024-11-30); v1 2024-11-18. Cheng-Yen Yang ... Jenq-Neng Hwang.
- One line: no retraining; a Kalman filter on the box picks among SAM 2's candidate masks, and memory frames are chosen by mask affinity, occurrence, and motion scores instead of the last N.
- Equation (Sec. 3.1): `\mathcal{M}^* = \argmax_{\mathcal{M}_i} (\alpha_{kf} \cdot s_{kf}(\mathcal{M}_i) + (1-\alpha_{kf}) \cdot s_{mask}(\mathcal{M}_i)).`
- Number: LaSOT AUC 74.23 for SAMURAI-L vs 68.54 for SAM2.1-L (+5.69) (Tab. 5). Abstract's "7.1% AUC gain on LaSOT_ext" is vs prior trackers, not vs SAM 2.

---

## Outline feedback (ch-efficient, ch-family)
1. Chart: do not plot one "params" axis across papers. Scopes differ: MobileSAM reports the encoder (5.78M vs 632M); EfficientSAM (636M) and EfficientViT-SAM (641.1M) report the whole model; EdgeTAM reports no params; EfficientSAM3 has only approximate student sizes and no results. Use one panel per paper, or chart EfficientViT-SAM Tab. 1 as the single same-setup comparison.
2. EfficientSAM3 v1 has no results. Show it as a recipe, not a data point.
3. Quality metrics differ: MobileSAM = mIoU agreement with SAM masks; EfficientSAM / EfficientViT-SAM = zero-shot COCO AP with ViTDet boxes; EdgeTAM = J&F and SA-23 mIoU; HQ-SAM = mBIoU on fine-structure sets.
4. Add MobileSAMv2 as a one-liner: it speeds up the automatic mask generator (prompt count), not the encoder. It links back to ch-engine's grid.
5. Grounded SAM date: paper 2024-01-25, but the repo existed by 2023-06 (cited by MobileSAM). Label the tree node.
6. SAM-Adapter's SAM baseline is unprompted; say so wherever its numbers appear.
