# RT-DETR: DETRs Beat YOLOs on Real-time Object Detection

arXiv 2304.08069v3 (CVPR 2024 camera-ready). Zhao, Lv, Xu, Wei, Wang, Dang, Liu, Chen (Baidu Inc; Peking University). Source read: `sources/rt-detr/tex/` (main.tex plus sec/*.tex, including sec/X_suppl.tex which is the appendix; its sections are numbered 1-5 in the PDF while its tables and figures are lettered Table A-C, Figure A-D, so "Appendix Sec. N" below means appendix section N, not Table N). Table/figure/equation numbers below were checked against paper.pdf.

Official code: https://github.com/lyuwenyu/RT-DETR [rtdetr-repo]. The arXiv abs page does not link it, and in the v3 LaTeX the sentence "Source code and pre-trained models are available at https://github.com/lyuwenyu/RT-DETR" is commented out in sec/abstract.tex. The rendered abstract links only the project page https://zhao-yian.github.io/RTDETR [rtdetr-project-page], and that project page's "Code" button links https://github.com/lyuwenyu/RT-DETR. The repo README says "This is the official implementation of papers" this one and RT-DETRv2 (arXiv 2407.17140).

Naming note: v3 calls the query-selection idea "uncertainty-minimal query selection". The Figure 6 label is still `fig:iou-aware` (the included file is `figs/uncertainty.pdf`; an unused `figs/iou_aware.pdf` also sits in the source), the Table 4 label is `tab:iou-aware`, and commented-out text says "IoU-aware query selection", so earlier arXiv versions used the IoU-aware name. Same idea, renamed.

## Delta from predecessors (the problem)

- YOLO-style detectors emit many overlapping boxes and need NMS afterward. NMS "not only slows down the inference speed but also introduces hyperparameters that cause instability in both the speed and accuracy" (Sec. 1). YOLO papers usually report model speed without NMS time, so the paper builds an end-to-end speed benchmark that includes NMS (Sec. 3.2).
- DETRs (set prediction with bipartite matching) need no NMS, but "their high computational cost prevents them from meeting real-time detection requirements" (Sec. 1). Example: DINO-Deformable-DETR-R50 runs at 5 FPS [rtdetr-dino-fps].
- Two specific bottlenecks the paper names:
  1. The encoder. Multi-scale features make the encoder's input sequence very long. Citing Lin et al., the encoder is 49% of GFLOPs but only 11% of AP in Deformable-DETR [rtdetr-deform-enc-cost] (Sec. 4.2).
  2. Query initialization. Earlier query-selection schemes (Deformable DETR two-stage, Efficient DETR, DINO) pick the top-K encoder features by classification score only. Some of the chosen features then have low localization confidence (Sec. 4.3).

## The one key idea

Make a DETR fast enough to beat YOLOs end to end, NMS time included. The redesigned encoder does self-attention only on the coarsest feature map (AIFI) and fuses scales with a cheap CNN path (CCFF). Queries are selected with a training loss that makes a high classification score also mean a good box. Because there is no NMS, the only post-processing knob is a score threshold, and changing it does not change speed (Appendix Sec. 4).

## Mechanism step by step

1. Backbone (ResNet-50/101 in the headline models) gives the last three stages {S3, S4, S5} (Sec. 4.1, Figure 4).
2. Efficient hybrid encoder (Sec. 4.2, Eq. 1):
   - AIFI (Attention-based Intra-scale Feature Interaction): flatten S5 only and run one Transformer encoder layer on it [rtdetr-aifi-layers]. Reshape the result back to the S5 shape, giving F5. The paper's reason: self-attention on high-level features connects "conceptual entities", while intra-scale attention on lower levels is unnecessary and risks "duplication and confusion".
   - CCFF (CNN-based Cross-scale Feature Fusion): a PANet-style fusion path over {S3, S4, F5}. Fusion blocks are inserted along the path (Figure 5). Each block takes two adjacent scales, applies two 1x1 convs to adjust channels, runs N RepBlocks (built from RepConv) on one path, and adds the two paths element-wise. Headline models use 3 RepBlocks [rtdetr-repblocks].
   - The path to this design is an ablation ladder of encoder variants A to E (Figure 3, Table 3). All start from DINO-Deformable-R50 with a smaller data reader and a lighter decoder:
     - A: no encoder.
     - B: single-scale Transformer encoder, shared across scales.
     - C: B plus cross-scale fusion via a multi-scale Transformer encoder, so intra-scale and cross-scale interaction happen together.
     - D: decoupled. The single-scale Transformer handles intra-scale interaction and a PANet-style CNN handles cross-scale fusion.
     - D_S5: D with intra-scale attention on S5 only.
     - E: AIFI + CCFF, the final design.
3. Uncertainty-minimal query selection (Sec. 4.3, Eq. 2-3): select the top K = 300 encoder features by classification score [rtdetr-K]. The selected features become the content queries, and their predicted boxes become the initial position queries (Sec. 5.3). What changes from vanilla selection is training. A feature's "uncertainty" is the discrepancy between its predicted localization quality and its classification score. Folding it into the classification loss teaches the encoder to give high class scores only where the box is also good.
4. Transformer decoder with auxiliary prediction heads refines the queries over 6 layers [rtdetr-dec-layers]. The decoder's "training strategy and hyperparameters almost follow DINO" (Appendix Sec. 1), including denoising (denoising number 200 [rtdetr-dn-number]).
5. Speed tuning without retraining: every decoder layer has a prediction head, so inference can stop at an earlier layer. For the model trained with 6 layers, using the 5th layer's output costs 0.1 AP (53.1 to 53.0) and saves 0.5 ms (9.3 to 8.8 ms) [rtdetr-dec-5of6]. See Table 5 below.
6. Post-processing: filter by a score threshold only. The paper says this "does not affect the inference speed" (Appendix Sec. 4).

## NMS analysis (exact numbers and where)

Section 3.1 "Analysis of NMS", Figure 2 and Table 1. Section 3.2 has the benchmark and Appendix Sec. 4 the qualitative visualization.

How NMS works as the paper describes it: boxes scoring below the confidence threshold are dropped. Whenever two boxes have IoU above the IoU threshold, the lower-scoring one is discarded. This repeats until all boxes of every category are processed. "Thus, the execution time of NMS primarily depends on the number of boxes and two thresholds."

Figure 2, "The number of boxes at different confidence thresholds":
- Models: YOLOv5 (anchor-based) and YOLOv8 (anchor-free), counted "on the same input". The paper does not say which image or model size.
- x-axis confidence thresholds: 0.001, 0.005, 0.01, 0.05, 0.1, 0.25. The text says "We sample values from 0.001 to 0.25" [rtdetr-fig2-thresholds].
- The paper gives no numeric bar values. I measured the bar heights from the vector geometry of `tex/figs/num_boxes.pdf`: y-axis 0 to 10k, bar top converted linearly, rounded to the nearest box. These are derived values, not stated ones [rtdetr-fig2-counts, uncertain]:

| conf thr | YOLOv5 boxes (derived) | YOLOv8 boxes (derived) |
|---|---|---|
| 0.001 | ~9281 | ~3570 |
| 0.005 | ~5877 | ~2523 |
| 0.01 | ~4690 | ~2057 |
| 0.05 | ~2142 | ~1124 |
| 0.1 | ~1304 | ~764 |
| 0.25 | ~581 | ~381 |

- Stated in text (Sec. 3.2): anchor-based detectors produce more prediction boxes than anchor-free ones, "three times more in our tested detectors" [rtdetr-anchor-3x]. The derived ratio at 0.001 is ~2.6x (9281/3570), so "three times" is the paper's rounded statement about its tested detectors. It may not come from this single figure.

Table 1, "The effect of IoU threshold and confidence threshold on accuracy and NMS execution time":
- Model: "YOLOv8", evaluated on COCO val2017. Sec. 3.1 does not give the size. The AP of 52.9 at conf 0.001 / IoU 0.7 equals the YOLOv8-L AP in Table 2, and Figure A visualizes YOLOv8-L, so the model is probably YOLOv8-L. That is an inference, not stated [rtdetr-t1-model, uncertain].
- Timing: T4 GPU, TensorRT FP16, TensorRT `efficientNMSPlugin`. Only the `EfficientNMS` kernel time is reported, not `EfficientNMSFilter`, `RadixSort`, etc. [rtdetr-t1-setup].

Left half (confidence fixed at 0.001, IoU threshold varied) [rtdetr-t1-iou05, rtdetr-t1-iou06, rtdetr-t1-iou08]:

| IoU thr | AP (%) | NMS (ms) |
|---|---|---|
| 0.5 | 52.1 | 2.24 |
| 0.6 | 52.6 | 2.29 |
| 0.8 | 52.8 | 2.46 |

Right half (IoU fixed at 0.7, confidence threshold varied) [rtdetr-t1-conf0001, rtdetr-t1-conf001, rtdetr-t1-conf005]:

| Conf thr | AP (%) | NMS (ms) |
|---|---|---|
| 0.001 | 52.9 | 2.36 |
| 0.01 | 52.4 | 1.73 |
| 0.05 | 51.2 | 1.06 |

Paper's conclusions:
- "the execution time of the EfficientNMS kernel increases as the confidence threshold decreases or the IoU threshold increases". A high confidence threshold removes more boxes up front. A high IoU threshold removes fewer boxes per round.
- The best YOLOv8 AP comes at conf 0.001 / IoU 0.7, "but the corresponding NMS time is at a higher level" [rtdetr-t1-conf0001].
- Appendix Sec. 4, qualitative: a higher confidence threshold gives more false negatives, while a low one (e.g. 0.001) gives many redundant boxes and false positives. A higher IoU threshold gives more false positives, while a lower one can delete true positives among overlapping or occluded objects. The IoU threshold "is difficult to set accurately".

End-to-end benchmark (Sec. 3.2): COCO val2017. NMS plugin appended to YOLOs, each run at the NMS thresholds its reported accuracy used. Timing is averaged over images, excludes I/O and MemoryCopy, on T4 with TensorRT FP16 [rtdetr-fps-setup]. The paper's conclusion: at equal accuracy, anchor-free YOLOs beat anchor-based ones on end-to-end speed because they need less NMS time.

Slider caveat: the paper gives exactly 3 confidence points with measured NMS time (0.001, 0.01, 0.05 at IoU 0.7) and 6 confidence points with (unlabeled) box counts. It gives no NMS time at 0.005, 0.1 or 0.25, and no timing for YOLOv5. Any interpolated curve is illustrative only.

## Equations

Rendering note: `\mathbfcal` is a paper-local macro (`\DeclareMathAlphabet\mathbfcal{OMS}{cmsy}{b}{n}` in main.tex). KaTeX/MathJax will not know it, so define it or substitute `\boldsymbol{\mathcal{S}}_5` etc. The equations below are copied verbatim from sec/method.tex, including the `split` wrappers.

Eq. (1), hybrid encoder (Sec. 4.2):

```latex
\begin{split}
    \mathbfcal{Q} & = \mathbfcal{K} = \mathbfcal{V} = \texttt{Flatten}(\mathbfcal{S}_5), \\
    \mathbfcal{F}_5&=\texttt{Reshape}(\texttt{AIFI}(\mathbfcal{Q},\mathbfcal{K},\mathbfcal{V})), \\
    \mathbfcal{O} & = \texttt{CCFF}(\{\mathbfcal{S}_3,\mathbfcal{S}_4,\mathbfcal{F}_5\}),
\end{split}
```

In words:
- S5 is the last backbone stage (coarsest map). Flatten turns its HxW grid into a token sequence, and that same sequence is the query, key and value.
- AIFI is the single-scale Transformer encoder layer (self-attention + FFN).
- Reshape restores the S5 spatial shape, giving F5.
- S3 and S4 are the finer backbone stages. They get no attention at all and go straight into CCFF.
- CCFF is the CNN fusion path. O is the encoder output, the multi-scale features flattened into one sequence for query selection and the decoder.

Eq. (2), feature uncertainty (Sec. 4.3):

```latex
\mathcal{U}(\hat{\mathbfcal{X}}) = \| \mathcal{P}(\hat{\mathbfcal{X}}) - \mathcal{C}(\hat{\mathbfcal{X}}) \|, \hat{\mathbfcal{X}} \in \mathbb{R}^D
```

In words:
- X-hat is one encoder feature, a D-dimensional vector.
- P(X-hat) is its predicted localization distribution and C(X-hat) its predicted classification distribution.
- U is the norm of their difference. When a feature is confident about the class but its box is poor (or the reverse), U is large. The paper calls this "epistemic uncertainty" modeling the "joint latent variable" of feature quality.

Eq. (3), loss (Sec. 4.3):

```latex
\begin{split}
\mathcal{L}(\hat{\mathbfcal{X}}, \hat{\mathbfcal{Y}}, \mathbfcal{Y}) = \mathcal{L}_{box}(\hat{\mathbf{b}}, \mathbf{b}) + \mathcal{L}_{cls}(\mathcal{U}(\hat{\mathbfcal{X}}), \hat{\mathbf{c}}, \mathbf{c})
\end{split}
```

In words:
- Y-hat = {c-hat, b-hat} is the prediction (category c-hat, box b-hat) and Y = {c, b} is the ground truth. X-hat is the encoder feature.
- L_box compares predicted and ground-truth boxes.
- L_cls compares predicted and ground-truth categories, and it also takes the uncertainty U as input. The classification target therefore depends on how good the box is, so a high score after training implies a good box.
- v3 does not give the concrete form of L_box or L_cls. Commented-out LaTeX near Eq. 3 says IoU replaces the one-hot label in the classification loss, names VFL / QFL as alternatives, and sketches L_box as L1 + GIoU. Table A lists "class loss weight 1.0, alpha in class loss 0.75, gamma in class loss 2.0, bbox loss weight 5.0, GIoU loss weight 2.0" (the commented row "class loss type & varifocal loss" is hidden). So the IoU-as-soft-target (varifocal-style) reading is well supported, but it is not stated in the rendered v3 text [rtdetr-loss-form, uncertain].

## Results worth remembering

Headline (Table 2, COCO val2017, input 640x640 for RT-DETR and YOLOs; FPS_{bs=1} on T4 GPU, TensorRT FP16, end-to-end incl. NMS for YOLOs) [rtdetr-fps-setup, rtdetr-input-size]:

| Model | #Epochs | #Params (M) | GFLOPs | FPS | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|---|---|---|
| RT-DETR-R50 | 72 | 42 | 136 | 108 | 53.1 | 71.3 | 57.7 | 34.8 | 58.0 | 70.0 |
| RT-DETR-R101 | 72 | 76 | 259 | 74 | 54.3 | 72.7 | 58.6 | 36.0 | 58.8 | 72.1 |

Claims: [rtdetr-r50-ap, rtdetr-r50-fps, rtdetr-r50-params, rtdetr-r50-gflops, rtdetr-r101-ap, rtdetr-r101-fps, rtdetr-r101-params, rtdetr-r101-gflops, rtdetr-epochs, rtdetr-r50-aps, rtdetr-r101-aps].

YOLO rows from Table 2 (FPS measured by the authors with their end-to-end benchmark using official pre-trained models):

| Model | #Params (M) | GFLOPs | FPS | AP | APS |
|---|---|---|---|---|---|
| YOLOv5-L | 46 | 109 | 54 | 49.0 | - |
| YOLOv5-X | 86 | 205 | 43 | 50.7 | - |
| PPYOLOE-L | 52 | 110 | 94 | 51.4 | 31.4 |
| PPYOLOE-X | 98 | 206 | 60 | 52.3 | 33.3 |
| YOLOv6-L | 59 | 150 | 99 | 52.8 | 34.4 |
| YOLOv7-L | 36 | 104 | 55 | 51.2 | 35.2 |
| YOLOv7-X | 71 | 189 | 45 | 52.9 | 36.9 |
| YOLOv8-L | 43 | 165 | 71 | 52.9 | 35.3 |
| YOLOv8-X | 68 | 257 | 50 | 53.9 | 35.7 |

Claims: [rtdetr-yolov5l, rtdetr-yolov5x, rtdetr-ppyoloel, rtdetr-ppyoloex, rtdetr-yolov6l, rtdetr-yolov7l, rtdetr-yolov7x, rtdetr-yolov8l, rtdetr-yolov8x].

Paper's stated YOLO comparisons (Sec. 5.1 text):
- R50 vs YOLOv5-L / PP-YOLOE-L / YOLOv6-L: +4.1 / +1.7 / +0.3 AP, FPS +100.0% / +14.9% / +9.1%, params -8.7% / -19.2% / -28.8% [rtdetr-cmp-r50-l1].
- R101 vs YOLOv5-X / PP-YOLOE-X: +3.6 / +2.0 AP, FPS +72.1% / +23.3%, params -11.6% / -22.4% [rtdetr-cmp-r101-x1].
- R50 vs YOLOv7-L / YOLOv8-L: +1.9 / +0.2 AP, FPS +96.4% / +52.1% [rtdetr-cmp-r50-l2].
- R101 vs YOLOv7-X / YOLOv8-X: +1.4 / +0.4 AP, FPS +64.4% / +48.0% [rtdetr-cmp-r101-x2].

The gap that matters most for the chapter: against YOLOv8, RT-DETR is barely more accurate (+0.2 / +0.4 AP) but about 1.5x faster end to end.

DETR comparison: DINO-Deformable-DETR-R50 has 36 epochs, 47M params, 279 GFLOPs, 5 FPS and 50.9 AP. It is timed with TensorRT FP16 at input (800, 1333) [rtdetr-dino-fps, rtdetr-dino-ap]. RT-DETR-R50 is +2.2 AP and "about 21 times" faster (108 vs 5 FPS) [rtdetr-vs-dino]. Other DETRs in Table 2 are not timed (the paper says they are not real-time).

Objects365 pre-training (Table C): 12 epochs on Objects365, then COCO fine-tuning (60 epochs for R18, 24 for R50/R101). R18 reaches 49.2 AP (+2.7) at 217 FPS, R50 55.3 AP (+2.2) at 108 FPS, and R101 56.2 AP (+1.9) at 74 FPS [rtdetr-o365-r18, rtdetr-o365-r50, rtdetr-o365-r101].

Encoder ablation (Table 3, AP with 1x config = 12 epochs) [rtdetr-t3-A ... rtdetr-t3-E]:

| Variant | AP (%) | #Params (M) | Latency (ms) |
|---|---|---|---|
| A | 43.0 | 31 | 7.2 |
| B | 44.9 | 32 | 11.1 |
| C | 45.6 | 32 | 13.3 |
| D | 46.4 | 35 | 12.2 |
| D_S5 | 46.8 | 35 | 7.9 |
| E | 47.9 | 42 | 9.3 |

Text deltas (Sec. 5.2):
- B vs A: +1.9 AP, latency +54%.
- C vs B: +0.7 AP, +20%.
- D vs C: +0.8 AP, -8%.
- D_S5 vs D: +0.4 AP, -35% latency.
- E vs D: +1.5 AP, params +20%, latency -24%.

[rtdetr-enc-deltas]. The latency hardware and precision for Table 3 are not stated in Sec. 5.2 [rtdetr-t3-latency-hw, uncertain].

Query selection ablation (Table 4, RT-DETR-R50, 1x config) [rtdetr-t4-vanilla, rtdetr-t4-um]:

| Query selection | AP (%) | Prop_cls (%) | Prop_both (%) |
|---|---|---|---|
| Vanilla | 47.9 | 0.35 | 0.30 |
| Uncertainty-minimal | 48.7 | 0.82 | 0.67 |

Prop_cls is the proportion of selected features with classification score > 0.5 (the paper writes these as 0.35% / 0.82%; only the ratio matters for the 138% / 120% comparison), and Prop_both the proportion with both classification and IoU score > 0.5. Figure 6 text: "138% more purple dots than green dots" (class score > 0.5) and "120% more" with both scores > 0.5 [rtdetr-fig6-138, rtdetr-fig6-120] (exact as stated). Note: the Table 4 ratios are 0.82/0.35 = 2.34 (134% more) and 0.67/0.30 = 2.23 (123% more), which do not exactly match 138% / 120%. That is probably rounding of the underlying counts, but it is not reconciled in the paper.

Decoder-depth ablation (Table 5, RT-DETR-R50, 6x config). Rows are the decoder layer index used for inference, columns Det^k are models trained with k layers:

| ID | Det^4 | Det^5 | Det^6 | Det^7 | Latency (ms) |
|---|---|---|---|---|---|
| 7 | - | - | - | 52.6 | 9.6 |
| 6 | - | - | 53.1 | 52.6 | 9.3 |
| 5 | - | 52.9 | 53.0 | 52.5 | 8.8 |
| 4 | 52.7 | 52.7 | 52.7 | 52.1 | 8.3 |
| 3 | 52.4 | 52.3 | 52.4 | 51.5 | 7.9 |
| 2 | 51.6 | 51.3 | 51.3 | 50.6 | 7.5 |
| 1 | 49.6 | 48.8 | 49.1 | 48.3 | 7.0 |

[rtdetr-t5-det6, rtdetr-t5-latency, rtdetr-dec-5of6]. Training with 6 layers is best (53.1). Training with 7 is worse (52.6).

Lighter scaled models (Table B, RepBlock channels halved; the Dec^k variants are one trained model per backbone truncated at inference; scaled R50/34/18 were trained with 6/4/3 decoder layers, Appendix Sec. 2):
- Scaled RT-DETR-R50-Dec^5: 52.1 AP, 132 FPS, vs YOLOv8-M at 50.2 AP, 97 FPS [rtdetr-tB-r50dec5, rtdetr-tB-yolov8m].
- Scaled RT-DETR-R18-Dec^2: 45.5 AP, 238 FPS, vs YOLOv6-S at 45.0 AP, 201 FPS [rtdetr-tB-r18dec2, rtdetr-tB-yolov6s].
- The text claims R50-Dec^5 is "+0.9% AP" and "+36%" FPS over YOLOv8-M. The FPS figure matches (132/97 = 1.36), but the table rows give 52.1 - 50.2 = 1.9 AP, so the text's 0.9 conflicts with the table [rtdetr-tB-text-mismatch, uncertain]. The R18-Dec^2 vs YOLOv6-S text (+0.5 AP, +18% FPS) is consistent with the table [rtdetr-tB-r18-text].

Training setup (Appendix Sec. 1, Table A) [rtdetr-train-setup]:
- AdamW, 4x V100, batch 16, EMA 0.9999.
- 1x = 12 epochs; reported results use 6x.
- Embedding dim 256, 8 heads, 300 queries, 6 decoder layers, decoder npoints 4.

## Limitations

- Stated by the paper (Sec. 6): small-object AP still trails the strongest real-time detectors. RT-DETR-R50's APS is 0.5 below the best L model (YOLOv8-L, 34.8 vs 35.3), and RT-DETR-R101's is 0.9 below the best X model (YOLOv7-X, 36.0 vs 36.9) [rtdetr-limit-aps].
- Observations (mine, from the text):
  - The NMS timing isolates only the `EfficientNMS` kernel on one GPU/runtime (T4, TRT FP16), and the box-count figure uses a single unspecified input.
  - YOLO FPS numbers are the authors' re-measurements under their own benchmark, not the YOLO papers' figures.
  - DINO is timed at (800, 1333) input vs 640x640 for RT-DETR, so the 21x speedup partly reflects input size.
  - The uncertainty loss is only given abstractly in v3.

## Facts useful for visuals (with claim ids)

- NMS slider (the core interactive):
  - Measured anchors (YOLOv8, IoU 0.7): conf 0.001 gives AP 52.9 / NMS 2.36 ms, conf 0.01 gives 52.4 / 1.73 ms, conf 0.05 gives 51.2 / 1.06 ms [rtdetr-t1-conf0001, rtdetr-t1-conf001, rtdetr-t1-conf005].
  - Box-count trend (derived from Figure 2 geometry, label as approximate): YOLOv5 falls from ~9.3k at 0.001 to ~0.6k at 0.25, and YOLOv8 from ~3.6k to ~0.4k [rtdetr-fig2-counts].
  - The honest framing is "lower threshold means more boxes, more NMS time, and slightly higher AP".
- IoU-threshold toggle (conf 0.001): IoU 0.5 gives 2.24 ms, 0.6 gives 2.29 ms, 0.8 gives 2.46 ms. AP is 52.1 / 52.6 / 52.8 [rtdetr-t1-iou05, rtdetr-t1-iou06, rtdetr-t1-iou08].
- RT-DETR counterpart: a score threshold only, "does not affect the inference speed" (Appendix Sec. 4, Figure B) [rtdetr-score-thr-speed].
- Speed-accuracy scatter (Figure 1 style): the Table 2 rows above.
- Encoder ladder bar chart: Table 3 AP vs latency. D to D_S5 is the dramatic drop, 12.2 to 7.9 ms with AP up 46.4 to 46.8 [rtdetr-t3-D, rtdetr-t3-DS5].
- Decoder-layer dial: Table 5 Det^6 column, latency 7.0 to 9.3 ms for layers 1 to 6, about 0.4-0.5 ms per layer [rtdetr-t5-det6, rtdetr-t5-latency].
- Query-selection scatter (Figure 6): class score vs IoU score for selected features, with purple (uncertainty-minimal) clustered top-right and green (vanilla) bottom-right. Only points with class score > 0.5 are plotted [rtdetr-fig6-138].
- Encoder cost motivation: 49% of GFLOPs for 11% of AP in Deformable-DETR (cited from Lin et al.) [rtdetr-deform-enc-cost].
