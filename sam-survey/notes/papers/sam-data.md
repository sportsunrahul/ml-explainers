# SAM (2304.02643v1) -- data engine, SA-1B, zero-shot transfer, ablations, limitations

Scope of this note: chapters ch-engine and ch-zeroshot. The model/architecture/training side is in `sam-model.md`
(parallel agent). Claims: `notes/claims/sam-data.json`.

## 1. Identity
- Title: Segment Anything. arXiv 2304.02643, version read: v1 (abs.html "Submitted on 5 Apr 2023"; only [v1] listed).
- Authors: Alexander Kirillov ... Ross Girshick (12 authors). Data card: "created by the FAIR team of Meta AI" (App. F).
- Venue: abs.html gives no journal-ref (Comments: project web-page only); LaTeX uses the `iccv.sty` template.
- Repo: https://github.com/facebookresearch/segment-anything (linked in App. F, model license); project page
  https://segment-anything.com (abstract). README at `sources/repos/segment-anything.README.md` (not used here).
- Sources read: `latex/segany.tex` (Sec. 4, 5, 6, 7, 8; App. B, C, D, E, F), figure sources under `latex/figs/`
  (`becnhmark_ritm_comparison.pgf` and the figure PDFs), `paper.pdf` renders of p9 and p12.
- PDF page map (corrected by the fact check): Sec. 4 stage paragraphs p6 (stage 1 starts p5); mask properties and Fig. 6 p7; Sec. 7.6 text p11, Fig. 13 p12; App. B p18. See notes/factcheck-claims.md.

## 2. Core problem
Masks are not abundant on the internet, so a segmentation foundation model has no web-scale data to train on.
The paper builds the data with the model in the loop (a "data engine"), then tests whether the resulting model
transfers zero-shot to new image distributions and to tasks it was not trained for.

## 3. What prior data and methods did (per the paper)
- Existing segmentation datasets are small: the largest, Open Images, has 1M images and 2.7M masks (Fig. 6 legend).
- Manual polygon annotation is slow: COCO mask annotation is 6.5x slower than SAM-assisted annotation at 14 s/mask (Sec. 4).
- Interactive segmenters (RITM, SimpleClick, FocalClick) are single-mask methods trained on a few datasets
  (COCO+LVIS for RITM and SimpleClick). One point is ambiguous, so one output often picks the "wrong" valid object (Sec. 7.1).

## 4. The one key idea
Co-develop model and data: annotate with the current SAM, retrain SAM on the new masks, repeat; once the model is
ambiguity-aware and strong, let it annotate alone. The final dataset (SA-1B) contains only automatic masks.

## 5. Mechanism, step by step (data engine, Sec. 4 + App. B)
1. Assisted-manual stage. Professional annotators click foreground/background points in a browser tool powered by SAM
   (image embeddings precomputed, decoding real-time in browser); refine with pixel-precise "brush" and "eraser".
   No semantic constraints ("stuff" and "things"); label in order of prominence; move on once a mask takes over 30 s.
   SAM starts trained on public segmentation datasets, then is retrained on only newly annotated masks.
   Encoder scaled ViT-B -> ViT-H; retrained 6 times. Time per mask 34 s -> 14 s. Masks per image 20 -> 44.
   Output: 4.3M masks, 120k images.
2. Semi-automatic stage (goal: diversity). A Faster R-CNN box detector trained on all stage-1 masks with one generic
   "object" class pre-fills confident masks; annotators label the remaining unannotated objects.
   Retrained 5 times. Time per mask back up to 34 s (excluding the automatic masks). Masks per image 44 -> 72
   (including the automatic masks). Output: +5.9M masks in 180k images (10.2M total).
3. Fully automatic stage. Enabled by (a) enough diverse masks and (b) the ambiguity-aware model. Prompt with a 32x32
   regular grid of points; each point returns a set of masks (subpart, part, whole object).
   a. Crops (App. B): full image 32x32 grid, plus 20 zoomed crops from 2x2 and 4x4 partially overlapping windows with
      16x16 and 8x8 grids. Crops come from the original high-resolution images (the only use of them).
      Masks touching inner crop boundaries are removed.
   b. Filters: predicted IoU >= 88.0 ("confident"); stability >= 95.0 ("stable", see Sec. 6 below); drop masks that
      cover 95% or more of the image.
   c. NMS: greedy box-based, threshold 0.7, two phases: within crop (ranked by predicted IoU), then across crops
      (ranked most zoomed-in first: 4x4 crop -> 2x2 -> full image).
   d. Postprocessing: ~4% of masks have small spurious components -> remove components < 100 px (drop the mask if its
      largest component is < 100 px); ~4% have small holes -> fill holes < 100 px (holes = components of the inverted mask).
   e. Run on all 11M images -> 1.1B masks.
4. Special automatic-generation model (App. B). Differs from default SAM: trained on manual + semi-automatic data only;
   177656 iterations (default 90k) with large-scale jitter; simulated interaction with point and mask prompts only (no
   boxes); 4 sampled points per mask (default 9); mask decoder 3 layers (default 2). It "sacrifices some inference speed".
5. The chain the flywheel scene should show: stage-1/2 masks -> generation model -> SA-1B (automatic only) ->
   default SAM (trained on SA-1B only). Ablation: training on SA-1B alone is ~0.5 mIoU below all three stages with
   10x oversampling of manual/semi-automatic masks (Sec. 7.6).

## 6. Equations (prose definitions; no numbered equations in this scope)
- Stability (Sec. 4): "we consider a mask stable if thresholding the probability map at $0.5-\delta$ and $0.5+\delta$
  results in similar masks". App. B operationalizes: "We kept the prediction (\ie, the binary mask resulting from
  thresholding logits at 0) only if the IoU between its pair of -1 and +1 thresholded masks was equal to or greater than 95.0."
  Gloss: threshold the same soft mask (logits) at -1 and at +1; IoU of the two binary masks >= 95.0 (0-100 scale) -> keep.
  Logit -1/+1 corresponds to probability ~0.27/~0.73 (our reading, sigmoid).
- Predicted-IoU filter: model's IoU head score >= 88.0 (0-100 scale).
- Relative mask size (Sec. 5): "square root of the mask area divided by image area", i.e. sqrt(area_mask / area_image).
- Concavity (Sec. 5): "1 minus mask area divided by area of mask's convex hull", i.e. 1 - area_mask / area_hull;
  compared after stratified sampling over binned mask sizes.
- mIoU (App. D.1): per-dataset mIoU = per-mask IoU averaged over all objects; top line = mean of the 23 per-dataset mIoUs.
- Macros: `\sad` = SA-1B, `\sam` = SAM, `\x` = times, `\app` = approx.

## 7. Measurement setup
### SA-1B mask quality (Sec. 5)
500 random images (~50k masks); professional annotators corrected all masks with SAM + brush/eraser; IoU between each
automatic mask and its corrected version. Falsifier: low IoU pairs would mean automatic masks are poor.
### 23-dataset point benchmark (Sec. 7.1, App. D.1, Table 7)
- Datasets (short names, Fig. 8 order): ADE20K, BBBC038v1, Cityscapes, DOORS, DRAM, EgoHOS, GTEA, Hypersim, IBD, iShape,
  LVIS, NDD20, NDISPark, OVIS, PPDLS, Plittersdorf, STREETS, TimberSeg, TrashCan, VISOR, WoodScape, PIDRay, ZeroWaste-f.
  The LVIS here is LVIS v0.5 validation (Table 7). Datasets with > 15k masks are subsampled to ~10k masks. Faces blurred.
- Table 7 (image type; # images / # masks sampled): PPDLS plants 182/2347; BBBC038v1 microscopy 227/10506; DOORS
  boulders 10000/10000; TimberSeg logs 220/2487; NDD20 underwater 4402/6100; LVIS scenes 945/9642; STREETS traffic camera
  819/9854; ZeroWaste-f recycling 2947/6155; iShape irregular shapes 754/9742; ADE20K scenes 302/10128; OVIS occlusions
  2044/10011; Hypersim simulation 338/9445; NDISPark parking lots 111/2577; VISOR egocentric 1864/10141; Plittersdorf
  stereo 187/546; EgoHOS egocentric 2940/9961; IBD drones 467/11953; WoodScape fisheye driving 107/10266; Cityscapes
  driving (panoptic) 293/9973; PIDRay X-ray 3733/8892; DRAM paintings (semantic) 718/1179; TrashCan underwater 5936/9540;
  GTEA egocentric 652/1208. All instance masks except Cityscapes (panoptic) and DRAM (semantic).
- Prompt: first point = "center" (max of the mask's interior distance transform); each next point = farthest from the
  boundary of the error region. Variant: random first point. N in {1,2,3,5,9}.
- SAM default output: its most confident mask. "Oracle": best of the 3 masks vs ground truth.
- Baselines (largest public models): RITM HRNet32 IT-M (COCO+LVIS); FocalClick SegFormerB3-S2 (8-dataset "combined");
  SimpleClick ViT-H448 (COCO+LVIS). RITM is best at 1 point, so it is the default baseline.
- Human study (App. E): 7 of the 23 datasets (LVIS v0.5, VISOR, DRAM, IBD, NDD20, OVIS, iShape), 1000 random masks per
  dataset, center point prompt; RITM, SAM single-output, SAM, ground truth (4000 jobs/dataset). Masks < 24^2 px excluded;
  images rescaled to max side 2000. 30 annotators; one rater per job; 5-rater pilot average std 0.83; ~90 s per job.
  Scale 1-10 (1 = no object, 2-4 huge errors, 5-6 significant errors, 7-9 minor boundary errors, 10 no visible errors).
  Stats: paired t-test + paired bootstrap (10k) 99% CI (Table 8).
### Other zero-shot tasks
- Edges (App. D.2): BSDS500 200-image test; 16x16 grid -> 768 masks; no IoU/stability filter; NMS; Sobel on
  unthresholded probability maps, keep only outer-boundary pixels, pixel-wise max, normalize to [0,1], edge NMS.
- Proposals (App. D.3): LVIS v1 val, mask AR@1000; no crops; no IoU/stability filter; 64x64 grid, NMS 0.9 -> ~900
  masks/image; rank by mean of confidence and stability, truncate to 1000. Single-output ablation: 128x128 grid, NMS 0.95,
  ~950 masks/image, random ranking. Baseline: cascade ViTDet-H with score thresholds disabled, 1000 per stage (a
  "Detector Masquerading as Proposal generator" that games AR).
- Instance seg (App. D.4): prompt SAM with ViTDet-H boxes on COCO and LVIS v1 val; one extra refinement iteration
  (most confident mask + box fed back). Human study: 1000 LVIS v1 val masks with LVIS ground-truth boxes; Fig. 11 uses 794
  inputs (matching 794 COCO GT ratings).
- Text-to-mask (Sec. 7.5, App. D.5): train with CLIP *image* embeddings, infer with CLIP *text* embeddings. CLIP
  ViT-L/14@336px, embeddings l2-normalized. Masks from data-engine stages 1-2 only, area >= 100^2 px. Prompt: expand mask
  box by random 1x-2x, square crop, resize 336x336, with 50% probability zero pixels outside the mask, masked attention in
  CLIP's last layer restricted to inside-mask positions, use the output token. CLIP prompt first, then iterative points.
  Large-scale jitter, 120k iterations, batch 128.

## 8. Results worth remembering
### Data engine and SA-1B
- Stage numbers: see Sec. 5 above. 14 s/mask is 6.5x faster than COCO mask annotation, 2x slower than extreme-point boxes.
- SA-1B: 11M images, avg 3300x4950 px, released with shortest side 1500 px (COCO ~480x640); faces and plates blurred.
  1.1B masks (Fig. 6 legend: "1129M (1.1B) masks"); 99.1% of engine masks fully automatic; released SA-1B has only
  automatic masks. ~100 masks per image on average (Fig. 2 caption p3, App. F data card).
- Mask quality: 94% of pairs > 90% IoU; 97% of pairs > 75% IoU; prior inter-annotator consistency 85-91% IoU.
- Comparison (Sec. 5, Fig. 6 legend): 11x more images and 400x more masks than Open Images; 36x more masks per image than
  Open Images; ADE20K has 3.5x fewer masks per image. Legend: LVIS v1 0.120M images / 1.5M masks; COCO 0.123M / 0.9M;
  ADE20K 0.028M / 0.7M; Open Images 1M / 2.7M. Masks-per-image bins (Fig. 6 left): <10, 11-50, 51-100, 101-200, >200;
  no printed percentages. Concavity "broadly similar" to other datasets; no numbers printed.
- RAI (Table 1, p7): Europe 49.8% of images, Asia & Oceania 36.2%, Africa 2.8%, low income 0.9%; every region >= 28M
  masks; masks per image 94-108 across regions. People (Table 2): 1-point mIoU feminine 54.4 vs masculine 55.7, CIs overlap.
  Clothing (Table 6): feminine 76.3 vs masculine 81.0 at 1 point, intervals disjoint.
### Fig. 9a, SAM minus RITM, IoU at 1 center point (printed on bars; p9; also in figs/becnhmark_ritm_comparison.pgf)
PPDLS +46.9, BBBC038v1 +44.7, DOORS +41.1, TimberSeg +28.9, NDD20 +21.1, LVIS +18.5, STREETS +17.3, ZeroWaste-f +9.1,
iShape +8.8, ADE20K +7.8, OVIS +7.0, Hypersim +6.1, NDISPark +2.7, VISOR +1.8, Plittersdorf +1.5, EgoHOS +0.8,
IBD -0.3, WoodScape -0.6, Cityscapes -2.0, PIDRay -5.8, DRAM -6.5, TrashCan -15.0, GTEA -21.4.
16 of 23 positive (text: "higher results on 16 of the 23 datasets, by as much as ~47 IoU"). Verified against the p9 render.
Note: the bars are deltas, not absolute mIoU. Per-dataset absolute mIoU is not given anywhere in v1.
### Fig. 9a oracle circles (NOT printed; reconstructed from pgf vector x-coordinates; uncertain)
Delta of SAM oracle (best of 3) minus RITM: PPDLS 53.2, BBBC038v1 52.6, DOORS 50.3, TimberSeg 37.6, NDD20 26.4, LVIS 25.9,
STREETS 30.3, ZeroWaste-f 22.3, iShape 22.8, ADE20K 19.6, OVIS 22.5, Hypersim 16.6, NDISPark 16.0, VISOR 20.0,
Plittersdorf 6.6, EgoHOS 21.1, IBD 13.4, WoodScape 9.6, Cityscapes 8.5, PIDRay 13.9, DRAM 9.7, TrashCan 5.7, GTEA 6.3.
All positive (text: "with the oracle ... SAM outperforms RITM on all datasets").
Why trust the reconstruction (~0.1): the same axis scale applied to the bar ends reproduces all 23 printed deltas exactly;
mean printed delta 9.24 = Fig. 9c SAM 59.2 - RITM 50.0; mean reconstructed oracle delta 22.21 = Fig. 9c oracle 72.2 - 50.0.
### Fig. 9b / Fig. 18 human ratings (printed in Fig. 18 legends, p24; mean +- 95% CI)
| dataset | RITM | SAM single-out | SAM | GT |
|---|---|---|---|---|
| LVIS v0.5 | 6.5 +- 0.15 | 7.7 +- 0.12 | 8.1 +- 0.10 | 8.5 +- 0.09 |
| VISOR | 6.3 +- 0.16 | 7.5 +- 0.13 | 8.3 +- 0.09 | 8.5 +- 0.13 |
| DRAM | 5.9 +- 0.14 | 6.8 +- 0.15 | 7.7 +- 0.13 | 8.0 +- 0.15 |
| IBD | 7.1 +- 0.12 | 7.9 +- 0.11 | 8.3 +- 0.08 | 8.4 +- 0.09 |
| NDD20 | 6.4 +- 0.17 | 8.2 +- 0.11 | 8.6 +- 0.10 | 8.9 +- 0.06 |
| OVIS | 6.1 +- 0.15 | 7.2 (see note) | 7.7 (see note) | 8.8 +- 0.09 |
| iShape | 4.9 +- 0.16 | 6.2 +- 0.17 | 7.1 +- 0.15 | 9.3 +- 0.06 |
OVIS note: the Fig. 18f legend prints "7.2 +- 0.13, SAM" and "7.7 +- 0.12, SAM - single output", but Fig. 9b (orange SAM
marker ~7.7, yellow single-output ~7.2, from vector extraction and the p9 render), the Fig. 18f histogram shapes, Table 8
(SAM > single-output on OVIS, CI (0.27, 0.63)) and the text ("consistently lower ratings" for single-output) all say SAM
is 7.7 and single-output 7.2. Treat the Fig. 18f legend as swapped; claims marked uncertain.
SAM means fall between 7 and 9 ("A high score (7-9)..."). DRAM and IBD are worse for SAM on mIoU but rated higher.
Table 8: all SAM > RITM and SAM > single-output tests significant (e.g. LVIS v0.5 vs RITM p=4e-69, CI99 (1.40, 1.84)).
### Fig. 9c/9d N-point curves (NOT printed; reconstructed from figure PDF vectors; uncertain)
mIoU over 23 datasets at N = 1, 2, 3, 5, 9:
- Center points (9c): SAM 59.2, 70.7, 77.1, 81.9, 85.2; RITM 50.0, 66.8, 75.3, 82.6, 86.1; SimpleClick 46.2, 67.1, 76.0,
  83.9, 88.6; FocalClick 40.1, 53.9, 63.8, 75.7, 84.6; SAM oracle at 1 point 72.2.
- Random points (9d): SAM 56.7, 67.9, 73.6, 79.0, 83.5; RITM 36.5, 51.6, 60.6, 71.3, 80.7; SimpleClick 41.8, 57.1, 65.8,
  75.9, 84.6; FocalClick 30.5, 39.2, 47.0, 58.7, 72.6; SAM oracle at 1 point 70.4.
- Reading: SAM leads at 1-3 points; by 5-9 points RITM and SimpleClick are slightly above SAM (paper: "on par").
### Table 3 (BSDS500 edges, p10): ODS / OIS / AP / R50
HED (2015) .788/.808/.840/.923; EDETR (2022) .840/.858/.896/.930; zero-shot: Sobel (1968) .539/-/-/-; Canny (1986)
.600/.640/.580/-; Felz-Hutt (2004) .610/.640/.560/-; SAM (2023) .768/.786/.794/.928.
### Table 4 (LVIS v1 mask AR@1000, p10): all / small / med / large / freq / com / rare
ViTDet-H 63.0/51.7/80.8/87.0/63.1/63.3/58.3; SAM single out. 54.9/42.8/76.7/74.4/54.7/59.8/62.0;
SAM 59.3/45.5/81.6/86.9/59.1/63.9/65.8. SAM beats ViTDet-H on medium, rare, common (and is 86.9 vs 87.0 on large;
text says it "outperforms ViTDet-H on medium and large objects", which the table does not support for large).
### Table 5 (instance seg, mask AP, p11): COCO AP/APS/APM/APL | LVIS v1 AP/APS/APM/APL
ViTDet-H 51.0/32.0/54.3/68.9 | 46.6/35.0/58.0/66.3; SAM 46.5/30.8/51.0/61.7 | 44.7/32.5/57.6/65.5.
Fig. 11 human ratings (LVIS v1 boxes): LVIS GT 8.6 +- 0.06, SAM 8.1 +- 0.07, ViTDet-H 7.9 +- 0.08. Separately,
COCO GT 7.6 +- 0.12 rates 794 COCO ground-truth masks collected while the review process was tested (App. E). Table 8: SAM > ViTDet-H p=2e-05, CI99 (0.11, 0.42).
### Fig. 13 ablations (p12; bar/line values NOT printed, reconstructed from vectors; uncertain)
- Data stages, 1 point / 1 point oracle: manual 49.2 / 66.4; + semi-automatic 55.2 / 70.8; + automatic 59.8 / 72.8;
  automatic only 59.2 / 72.2. Text: automatic only is ~0.5 mIoU below all data; all-data run oversamples manual and
  semi-automatic masks 10x.
- Data volume (0.1M / 1M / 11M images): 1 point oracle 70.9 / 72.2 / 72.3; 2 points 69.3 / 70.4 / 70.8; 3 points
  75.0 / 76.9 / 77.2; 5 points 79.2 / 81.8 / 82.0. Text: 1M images (~10%, ~100M masks) comparable to full.
- Encoder (printed x labels: ViT-B 91M, ViT-L 308M, ViT-H 636M params): 1 point 57.4 / 58.6 / 59.2; oracle 69.1 / 71.7 /
  72.2. Text: ViT-H "improves substantially over ViT-B, but has only marginal gains over ViT-L".

## 9. Limitations the paper states (Sec. 8, p12, verbatim)
"While \sam performs well in general, it is not perfect. It can miss fine structures, hallucinates small disconnected
components at times, and does not produce boundaries as crisply as more computationally intensive methods that
``zoom-in'', \eg~\cite{chen2022focalclick}. In general, we expect dedicated interactive segmentation methods to outperform
\sam when many points are provided, \eg~\cite{liu2022simpleclick}. Unlike these methods, \sam is designed for generality and
breadth of use rather than high IoU interactive segmentation. Moreover, \sam can process prompts in real-time, but
nevertheless \sam's overall performance is not real-time when using a heavy image encoder. Our foray into the text-to-mask
task is exploratory and not entirely robust, although we believe it can be improved with more effort. While \sam can
perform many tasks, it is unclear how to design simple prompts that implement semantic and panoptic segmentation.
Finally, there are domain-specific tools, such as~\cite{berg2019}, that we expect to outperform \sam in their respective
domains."
Later fixes (from the outline's paper list; not verified in these sources): fine structures/boundaries -> HQ-SAM; heavy
encoder -> MobileSAM, EfficientSAM, EfficientViT-SAM; text/concepts and semantic prompting -> SAM 3; video -> SAM 2.

## 10. Outline feedback
- ch-engine: the scene lists grid -> 3 masks -> filters (IoU, stability, NMS). Add the missing steps: the 95%
  image-cover filter, the 20 zoomed crops (2x2 and 4x4 windows with 16x16 and 8x8 grids) with two-phase NMS at 0.7, and
  the 100-px component/hole postprocessing. Thresholds: IoU 88.0, stability 95.0 (logits -1/+1).
- ch-engine: "seconds per mask per stage" exists only for stages 1-2 (34 -> 14, then 34 excluding auto masks). Stage 3
  has no human time. Masks per image: 20 -> 44 -> 72 (stage 2 includes auto masks); SA-1B ~100.
- ch-engine flywheel: SA-1B was produced by a special model trained on stage 1+2 masks (3-layer decoder, 177656 iters),
  not by the default SAM. The default SAM is then trained on SA-1B only.
- ch-engine claim card: the mask-quality sample is 500 images (~50k masks); result 94% > 90 IoU, 97% > 75 IoU.
- ch-zeroshot: "sourced bar chart per dataset (SAM minus RITM)" is good: all 23 deltas are printed. The "toggle oracle"
  values are not printed; they are reconstructed from the pgf (uncertain, labelled). Label them "read from figure geometry".
- ch-zeroshot: human ratings cover only 7 of 23 datasets; the OVIS legend in Fig. 18 is swapped (see Sec. 8).
- ch-zeroshot: at 5-9 points SAM is slightly below RITM/SimpleClick in Fig. 9c (paper says "on par"). Do not claim a win.
- ch-zeroshot: keep LVIS v0.5 (23-suite, point human study) distinct from LVIS v1 (proposals, instance seg, box human study).
- ch-zeroshot: Table 4 text claims SAM beats ViTDet-H on large objects, but the table shows 86.9 vs 87.0. Quote the table.
- Inspection chapter: useful SAM-paper anchors are the weakest datasets (GTEA -21.4, TrashCan -15.0, DRAM -6.5, PIDRay
  -5.8 X-ray) and the fine-structure limitation.
