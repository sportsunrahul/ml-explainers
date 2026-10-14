# Fact check: ch-sam-arch, ch-training, ch-engine, ch-zeroshot (fork b-sam)

Sources: sources/sam/latex/segany.tex (SAM v1), sources/sam2/latex/sam2.1_arxiv.tex, sources/vit/latex/04_experiments.tex,
sources/sam/latex/figs/*.pdf|pgf. Line numbers are segany.tex unless stated.

## (a) WRONG

1. ch-engine, last paragraph: "SA-1B: 11M licensed images and 1.1B masks, 99.1% of them automatic".
   SA-1B is 100% automatic. The 99.1% is the share of all masks the data engine produced (stages 1-3), not of SA-1B.
   Source: L237 "Our data engine produced 1.1B masks, 99.1% of which were generated fully automatically"; L326 "SA-1B ...
   includes only automatically generated masks from the final stage of our data engine"; data card L1175 "The released
   masks ... were automatically inferred by our segmentation model". The claim record (sam-sa1b-auto-pct, "Share of
   data-engine masks") is right; the sentence misattributes it.
   Fix: "The data engine made about 1.1B masks, 99.1% of them automatically; SA-1B releases only the automatic ones."

2. ch-sam-arch, viz-samflow stage 8 text: "Layer-wise learning-rate decay of 0.8 gives the deepest encoder layers the
   smallest steps." Backwards under the usual meaning of "deepest" (furthest from the input). Layer-wise decay
   (Bao 2021 / Clark 2020, cited at L664) scales the rate by ld^(L - l): the layers nearest the INPUT get the smallest
   steps, the top layers the largest. SAM does not define it further (L664); BEiT/ELECTRA are not downloaded.
   Fix: "gives the earliest encoder layers (nearest the input) the smallest steps".

## (b) MISLEADING / over-claimed

1. ch-zeroshot, Instance segmentation bullet: "Prompt SAM with ViTDet-H's boxes. Mask AP is lower ... Yet people rated
   SAM's masks higher (8.1 vs 7.9)." The AP numbers use ViTDet boxes (App. D.4, L1033), but the human study used LVIS
   GROUND-TRUTH boxes for both models (Fig. 11 caption L504 "both applied to LVIS ground truth boxes"; App. E L1070
   "the model input was the LVIS ground truth box"). The sentence presents two setups as one. Say "given ground-truth
   LVIS boxes, raters scored ...".

2. ch-training item 2: "The IoU head learns to predict each mask's quality." SAM v1 does not say the IoU loss covers all
   three masks (L652, L656). SAM 2 lists supervising the IoU of *all* masks as its change "Unlike SAM"
   (sam2.1_arxiv.tex:986), and the page's own SAM 2 note below presents "all three masks" as a change. The two
   statements conflict. Mark item 2 as our reading, or drop "each".

3. ch-training, SAM 2 note: "an l1 loss on the IoU predictions of all three masks". The source restricts this to
   multi-mask predictions on the first click, in the SA-1B pre-training stage (sam2.1_arxiv.tex:986, App. D.2.1).
   Add "on the first click".

4. ch-engine, Semi-automatic paragraph: "SAM first filled in the masks it was confident about". The confident masks
   came from a separate Faster R-CNN box detector trained on stage-1 masks with one generic "object" class (L206).
   Same issue in scene-engine.js SCENES.stages: who: 'SAM pre-fills; 5 retrains'. Say "a box detector trained on
   stage-1 masks pre-filled the confident objects".

5. ch-engine, prediction prompt: "Most candidates are duplicates ... so box NMS removes far more than any quality
   filter." Stated as fact about SAM; no source gives per-filter survival counts. True only of the toy. Say
   "in this toy".

6. ch-engine viz-engine figcaption: "The grid, the order of the filters and their thresholds are SAM's (App. B)".
   App. B (L673-677) gives thresholds but not an order; Sec. 4 (L208) gives only IoU and stability before NMS. The
   position of the 95% cover filter and of clean-up relative to NMS is our reading. Cropping is listed with NMS
   inside App. B, before filtering.

7. ch-zeroshot, after the figure: "Most of SAM's losses are ranking errors on ambiguous clicks: a click on a hand in an
   egocentric video (GTEA) can mean the hand, the arm, or the person." The paper supports the general point (oracle
   beats RITM on all datasets, L384) but says nothing about GTEA or hands/arms. Label the example as our reading.

8. ch-sam-arch, first prose paragraph: "the prompt encoder and mask decoder together need less than 1% of the image
   encoder's compute". The source bounds only "our lightweight mask decoder" (L662). The prompt encoder is small,
   but the paper does not state the combined figure. Minor.

9. ch-zeroshot intro: "on image types it had not seen". Several of the 23 sets (LVIS v0.5, ADE20K, Cityscapes) are
   natural photos close to SA-1B. The paper says "diverse image distributions" (L376) and "never trained on these
   datasets". Minor: "datasets it had not seen".

10. ch-sam-arch prompt encoder: "Text goes through CLIP's text encoder." True of the text-to-mask variant only
    (L630, L513); the released SAM takes no text. Minor; ch-zeroshot already says text is exploratory.

11. ch-training item 2 link text "(the loss is in the previous chapter)" points to #ch-sam-arch; the loss equation
    eq-samloss is in ch-promptable (explainer.src.html:1396). Navigation, not a fact.

## (c) Honesty labels

- viz-costsplit (toy badge, data-toy, "Toy model" caption): correct. Caption claim "the paper gives no running time for
  the image encoder" holds (no encoder timing in segany.tex; only ~50ms decoder, L109, L193).
- viz-samflow ("Computed" badge, data-computed): correct. All variant numbers verified: ViT-B/L/H D,L,H = 768/12/12,
  1024/24/16, 1280/32/16 (vit/latex/04_experiments.tex:49-51); params 91M/308M/636M (figs/ablation_backbone_params.pdf
  axis labels, Fig. 13); recipe lr 8e-4/4e-4/8e-4, ld 0.6/0.8/0.8, dp 0.6/0.4/0.4, 180k/180k/90k, batch and GPUs
  128/128/256, wd 0.1 (L664, L666). Warmup 250 for B/L is inherited from the default recipe (the deviations list at
  L666 does not change it): acceptable.
- viz-rounds (toy): correct; sampling rules match L658-662.
- viz-engine (toy): correct; thresholds 88.0, 95.0 at -1/+1, 95% cover, NMS 0.7, 100 px match L673-677.
- viz-stages ("Sourced", no data-toy): 20->44, 44->72, 4.3M/120k, 5.9M/180k, 34->14 s, 34 s, 6 and 5 retrains,
  11M/1.1B verified (L202-208). Problem: the third bar draws a line from 72 to 100 as if masks per image rose within
  the fully automatic stage. The paper gives no starting value for stage 3; 100 is the SA-1B average of automatic
  masks only (Fig. 2 caption), not the end of a within-stage rise, and 72 counted manual plus automatic masks. Draw
  stage 3 as a single value (about 100) or label the 72 start as ours.
- viz-zeroshot ("Sourced", no data-toy): all 23 bar values match the labels in figs/becnhmark_ritm_comparison.pgf
  exactly. All 23 oracle ring values reproduce to 0.1 IoU from the pgf marker x positions (tick -20 at x=2.0617in,
  0 at 3.6126in, +20 at 5.1635in). Image-type tags match Table 7. Label is honest.

## (d) Equations

- ch-sam-arch, M_k[h,w] = MLP_k(t_k)^T U[:,h,w], labelled "Our notation for SAM v1, App. A". Faithful to L646
  ("a small 3-layer MLP ... spatially point-wise product between the upscaled image embedding and the MLP's output").
  Note: the paper speaks of one MLP; the per-token subscript k (a separate MLP per mask token) comes from the code,
  not the paper. Acceptable under "our notation"; U = 32 channels at 1/4 resolution matches L646, L650.
- No other display equations in these four sections.

## (e) Atlas

Not in scope for this fork.

## (f) Checked

- ch-sam-arch: all 53 claim uses; priority ids f1-sam-mlp-out (32, arithmetic from L646, L650), f1-sam-no-aug (L664),
  f1-sam-pe (L630), f1vit-h-width/layers/heads (vit 04_experiments.tex:51) verified. Prose on encoder, neck (1x1 then
  3x3, L626), prompt encoder (L630-632), decoder 4 steps, 8 heads, 2048, 128 (L642, L648), upscaling (L646, L650),
  ambiguity and fourth token (L652-654), loss 20:1 and IoU MSE weight 1.0 (L656), recipe (L664), compute 256 A100 /
  68 h (model card L1308). Inline ViT table and all samflow readouts checked. Costsplit scene checked.
- ch-training: all 29 uses; sam2-corr-clicks, sam2-no-empty-rounds, sam2-iou-l1, sam2-iou-all-masks verified at
  sam2.1_arxiv.tex:986 (App. D.2.1 confirmed via PDF heading). SAM rules L658-662. Rounds scene logic checked against
  the rules (random no-point round among rounds 2..10, last at 11).
- ch-engine: all 40 uses; stage paragraphs L202-208, App. B L673-679 (generation model: 3 decoder layers, 4 vs 9
  points, stage 1+2 data only), mask quality L239, 400x L251, SA-1B composition L237/L326. Both engine scenes checked.
- ch-zeroshot: all 61 uses plus scene-zeroshot.js badge spans; Sec. 7.1-7.5 L328-520, App. D.1 L957-963 (baselines
  at largest public sizes, 15k/10k subsampling, center point), App. D.3 L998-1004 (64x64 grid, 1000 proposals), App. E
  L1064-1072 (7 datasets, 1000 masks, 1-10 scale wording "no object at all" ... "no visible errors"), Table 3 rows,
  Limitations L596 (all six quotes verbatim). Fig. 9c read from the rendered figure: at 5 and 9 points SAM sits
  slightly below RITM and SimpleClick, so the "on par ... slightly below" sentence holds. LVIS v0.5 for the suite
  confirmed (Table 7 "Validation (v0.5)", L796).
- Not re-reported: sam-data page-number loc errors from notes/factcheck-claims.md.
