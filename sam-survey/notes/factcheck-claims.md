# Fact check of extracted claims (2026-10-09)

Scope: all five claims files, the equations in notes/papers/*.md, and notes/understanding.md.
Method: each value was compared against the cited location in the downloaded LaTeX, PDF, abs.html or README.
Every arxiv field matches the downloaded version (ID file). The v1/v2 dates match abs.html to the day:
SAM v1 2023-04-05; SAM 2 v1 2024-08-01, v2 2024-10-28; SAM 3 v1 2025-11-20, v2 2026-03-28; SAM 3D v1 2025-11-20, v2 2026-06-02.
An independent random sample of 30 non-uncertain claims was re-checked by hand; all 30 reproduce. One loc problem came out of it (sam3-window-tokens).

## (a) WRONG

None. No stored value differs from its source. Every reported problem is a location, a flag or a label.

## (b) DOUBTFUL / ambiguous loc

### Flag or label problems (fix first)

- sam2-readme-savtest-l (79.5): code_only flag is wrong. The value is in the paper: Table 17a, SAM 2 (Hiera-L) double-dagger row, SA-V test J&F (latex/tab/appendix_sa-v.tex:52).
- sam2-readme-lvosv2-l (80.6): code_only flag is wrong. The value is in Table 17c, SAM 2 (Hiera-L) double-dagger row, LVOSv2 J&F (latex/tab/appendix_LVOSv2.tex:42).
- sam-zs-inst-rating-coco-gt (7.6): value is right, label is wrong. COCO GT is 794 ratings of COCO ground-truth masks, collected while the review process was tested (App. E, segany.tex L1139). They are not ratings on LVIS v1 boxes. The same mislabel is in notes/papers/sam-data.md (Fig. 11 line).
- sam3-v2-date (2026-03-28): date is right. The parenthetical "adds Object Multiplex appendix" is not in abs.html, and the v1 source was not downloaded. It is an inference; label it so, or cite the README (SAM 3.1 Object Multiplex release dated 2026-03-27).
- unet-levels (5): value is right (four 2x2 poolings, Fig. 1). The "our reading" label is only in loc. Put it in the claim text or mark the claim uncertain.

### Wrong location, value correct

- sam3-window-tokens (24x24): the cited text (main.tex:1133, App. A.1 Model Ablations, backbone-ablation Implementation Details) describes the encoders in that ablation, not the default model spec. The default spec (main.tex:1414, App. C.2) says 3x3 non-overlapping windows of 336 px; 336/14 = 24 tokens is derived. Cite both, and label the 24x24 for the released model as derived.
- sam2-p1-fps (6 FPS): the data card is App. H.2, not App. G (App. G is the semi-supervised VOS comparison). Sec. 5.1 Phase 1 (sam2.1_arxiv.tex:337) is correct.
- sam2-fps-video-bplus (43.8), sam2-fps-video-l (30.2): the sentence is in paragraph 1 of Sec. 7 (sam2.1_arxiv.tex:612), not paragraph 2.
- sam-data.json, 45 claims: the PDF page in loc is wrong; the paragraph, table or figure name is right. Pages verified with pdftotext:
  - Semi-automatic and Fully automatic stage paras are on PDF p6, not p5: sam-engine-s2-masks, sam-engine-s2-images, sam-engine-s2-total, sam-engine-s2-retrains, sam-engine-s2-time, sam-engine-s2-mpi-end, sam-engine-s3-images, sam-engine-s3-masks
  - Fully automatic para is on p6 (not p5) and App. B Cropping is on p18 (not p16): sam-engine-s3-grid
  - App. B is on PDF p18, not p16: sam-engine-s3-crops, sam-engine-s3-crop-window-a, sam-engine-s3-crop-window-b, sam-engine-s3-crop-grid-2x2, sam-engine-s3-crop-grid-4x4, sam-engine-s3-nms, sam-engine-s3-iou-thresh, sam-engine-s3-stab-thresh, sam-engine-s3-stab-lo, sam-engine-s3-stab-hi, sam-engine-s3-cover-filter, sam-engine-s3-min-area, sam-engine-s3-spurious-pct, sam-engine-gen-iters, sam-engine-default-iters, sam-engine-gen-points, sam-engine-gen-decoder-layers
  - Fig. 6 legend is on PDF p7, not p6: sam-sa1b-masks-exact, sam-sa1b-lvis-images, sam-sa1b-lvis-masks, sam-sa1b-coco-images, sam-sa1b-coco-masks, sam-sa1b-ade-images, sam-sa1b-ade-masks, sam-sa1b-oi-images, sam-sa1b-oi-masks
  - Mask properties para and Fig. 6 caption are on PDF p7, not p6: sam-sa1b-vs-oi-images, sam-sa1b-vs-oi-masks, sam-sa1b-vs-oi-mpi, sam-sa1b-vs-ade-mpi
  - Sec. 7.6 Ablations text is on PDF p11, not p12 (only Fig. 13 is on p12): sam-abl-oversample, sam-abl-auto-only-gap, sam-abl-1m-share, sam-abl-1m-masks, sam-abl-subset-1m, sam-abl-subset-01m
  notes/papers/sam-data.md Sec. 1 page map has the same errors.

### Uncertain claims with no source in the download (information only)

- sam-vith-width (1280), sam-vith-depth (32), sam-vith-heads (16): correctly marked uncertain. They match ViT-Huge in the ViT paper, which is not downloaded; no file in sam, mae, vitdet or hiera states them. Download ViT (2010.11929) if the page shows them as fact.
- sam-model 636M (Fig. 13) vs MAE 632M (mae.tex L768) for ViT-H parameters: the stored 636M is right for its loc. Note the difference if both papers are cited.

## (c) Equation transcription problems

No transcription errors. Results per notes file:
- prehistory.md: FCN loss and mean IU, U-Net softmax, energy E (no minus sign in the source; the notes mark it sic) and w(x), Mask R-CNN L, MaskFormer pixel-cls loss, Eq. 1 and mask sigmoid, Mask2Former Eqs. 1 to 3: all verbatim.
- sam-model.md, sam-data.md: SAM has no display equations. The loss formula is labelled "our reading" and agrees with the prose. The stability, relative size, concavity and mIoU prose formulas match.
- sam2.md: no display equations in the paper. The two App. F.1.2 annotation-time formulas and constants (T_exam=30 s, T_loc=1 s, T_click=1.5 s, N_click=3, N_frame=1..8) match.
- sam3.md: E1 to E7 match the 7 of 8 LaTeX display blocks they quote. Omission in E4: the MoE gradient (dL_MoE/dtheta_j = p_j dL_j/dtheta_j) and the winner-takes-all piecewise gradient (main.tex:1456-1480) are given only in prose.
- sam3.md Sec. 5 step 6 says the presence product is an "Eq. in Sec. 6". It is in App. C.2 Presence Head (main.tex:1433-1436), with prose in Sec. 3.
- sam-model.md Sec. 5 step 1 "long side is 1024": App. A (segany.tex L625) says only "rescaling the image and padding the shorter side". Label it as our reading.
- sam-model.md Sec. 1 names Kirillov as project lead. The author block (segany.tex L65) marks Nikhila Ravi as project lead too.
- sam2.md: the README gives SAM 2.1 two dates (model table: September 29, 2024; news header: 09/30/2024). sam2-ckpt-2-1-date uses the first; mention the other.

## (d) understanding.md problems

All numbers checked hold: 21 VOC classes, 80 COCO classes, 133 / 80 / 150, SA-1B 11M / 1.1B / 99.1%, 20 -> 44 -> 72 masks per image, 11 rounds = 1 + 8 + 2, 1024 -> 256x64x64, ~50 ms in a browser on CPU, N = 6 recent memories with temporal encoding on recent frames only, the SAM 3 presence gradient rule (main.tex:1087, 1438), 927 claims, and every listed known conflict (each number was found at the stated place, including MaskFormer 874 in Sec. 4 text and 847 in the Table 2 header, and README 270K / 848M).
- Mechanism 1: "With more than one prompt the ambiguity is gone". The source says ambiguity is "much rarer" with multiple prompts (segany.tex L654). Soften the wording.
- Mechanism 4: "3 masks per point" is not stated for the fully automatic stage. Sec. 4 (L208) says "a set of masks" per point; 3 follows from the 3 output tokens. Sec. 7.2 edge detection states "3 per point" for its 16x16 grid. Mark it as derived or cite App. A.
- Mechanism 4 lists "crops" as a filter. Crops are an extra source of masks for small objects, not a filter (App. B Cropping). Minor wording.

## (e) Claims checked per file

| file | claims | non-uncertain | lines checked | wrong | doubtful |
|---|---|---|---|---|---|
| prehistory.json | 116 | 116 | 116 | 0 | 1 |
| sam-model.json | 98 | 89 | 98 | 0 | 0 |
| sam-data.json | 309 | 215 | 254 | 0 | 46 |
| sam2.json | 232 | 232 | 232 | 0 | 5 |
| sam3.json | 172 | 171 | 172 | 0 | 2 |

sam-model doubtful count excludes the 3 uncertain ViT-H claims (information only). sam3 count includes sam3-window-tokens from the manual sample.
sam-data: all 215 non-uncertain claims plus 39 uncertain ones. All 23 printed Fig. 9a deltas were checked against becnhmark_ritm_comparison.pgf and match (16 of 23 positive; mean 9.24 supports 9.2).
The reconstructed oracle deltas, the Fig. 13 bars and the Fig. 9c markers are plausible: the oracle deltas recomputed from pgf coordinates agree to 0.1, with mean 22.21 = 72.2 - 50.0.
The OVIS legend swap (SAM 7.2 vs single-output 7.7 printed) is supported by Table 8 and the text; both claims are correctly marked uncertain.
code_only claims: sam-model 1 and sam3 2 are correct (absent from the paper). sam2 has 2 wrong of 17 (see b).
