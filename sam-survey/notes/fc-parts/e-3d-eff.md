# Fact check: ch-sam3d and ch-efficient (with scene-sam3d, scene-efficient)

## (a) WRONG

None. Every checked value matches its source at the stated place (apart from the loc notes below).

## (b) MISLEADING / over-claimed

1. ch-efficient, figure viz-efficient, tab "One table, one setup" (parts/scene-efficient.js T1 array, label "COCO mAP where printed").
   The array sets COCO mAP to null for MobileSAM, EfficientSAM and EdgeSAM. EfficientViT-SAM Table 1 prints COCO mAP for every row:
   MobileSAM 38.7, EdgeSAM 42.1, EfficientSAM 44.4 (sources/efficientvitsam/latex/table/exp/efficiency.tex:10-12).
   "where printed" therefore says the paper does not print them, which is false. The plot also drops the XL0 row (278 img/s, 47.5; efficiency.tex:18) without saying so.
   Fix: show the three mAP values, or relabel "COCO mAP for EfficientViT-SAM and SAM only".
2. ch-efficient: "EfficientSAM3 (November 2025) proposes to distil SAM 1, 2 and 3 into nine small students".
   The title says "from SAM1, 2, and 3", but the method has one teacher, SAM 3: "transfers capability from SAM3 to lightweight students in three stages"
   (sources/efficientsam3/latex/main.tex:36; Stage 1 mimics "the SAM3 teacher", main.tex:118). SAM 1 and SAM 2 data (SA-1B, SA-V) are used, not SAM 1/2 as teachers.
   Correct: "proposes to distil SAM 3 into nine small students, in stages on SA-1B, SA-V and SAM 3's data". The "no results yet" part is right (main.tex:33, 224).
3. ch-efficient: "The same move works for the other members of the family. EdgeTAM ... shrinks SAM 2's memory attention".
   EdgeTAM's thesis is the opposite of "the same move": replacing the image encoder is "not sufficient for SAM 2" because memory attention dominates latency
   (sources/edgetam/latex/sec/1_intro.tex:13). Reword so EdgeTAM is the exception, not another instance.
4. ch-efficient, EdgeTAM sentence: "15.7 FPS ... against 0.7 for SAM 2-B+, at a cost in accuracy (SA-V val J&F 72.3 against 76.8)".
   76.8 is SAM 2.1-B+, not SAM 2-B+; SAM 2-B+ scores 73.6 on SA-V val (sources/edgetam/latex/table/2_tab_vos.tex:20-21). Both run at 0.7 FPS on iPhone.
   Name SAM 2.1-B+ for the 76.8, or use 73.6 against SAM 2-B+.
5. ch-efficient, MobileSAMv2: "cuts decoder time from 6464ms to 97ms".
   These are Table 2 totals of "prompt encoding (including pre-sampling)" plus "mask decoding (including post-filtering)"
   (sources/mobilesamv2/latex/main_mobileSAMv2_arxiv.tex:164, 177, 181, 183). The 97 ms includes 47 ms of object discovery; mask decoding alone is 6400 ms -> 50 ms.
   The table caption does call it the "(prompt-guided) mask decoder", so this is wording, not a wrong number. Suggest "prompt sampling plus decoding time".
6. ch-efficient: "ViT-H's 632M" (MobileSAM Table 1/3, sources/mobilesam/latex/main_MobileSAM_arxiv_v2.tex:120, 228) vs ch-sam-arch, ViT-H table in the
   page script (explainer.src.html:1679, par '636M', SAM Fig. 13). Two values for the same encoder on one page with no note. Add "as MobileSAM reports" or a footnote.
7. ch-efficient, HQ-SAM: "For inspection, where defects are often thin, this is the variant to know." This is the authors' recommendation, not a source statement. Label it "our reading".
   Also the FPS (4.8 vs 5.0) and 5.1M params are for ViT-L (sources/hqsam/latex/neurips_2023.tex:286, 296-297); the page names no backbone. Minor.
8. ch-sam3d: "pick the best of N=8 candidate shapes (80 seconds)". Sec. 3.2.1 says N = 8 (PDF p7), but the 80 s figure is from App. A.5, which says "6-10 candidate meshes"
   (App. A.5 Annotation Statistics, first bullets). Minor; mention "6 to 10 in practice" or drop N=8 next to the 80 s.
9. ch-sam3d: "mid-training on 61 million synthetic objects pasted into real images". The paper says "61 million samples with 2.8 million unique meshes" (Sec. 3.1.2, PDF p6);
   part of RP-3DO is object swap and ProcThor scenes (sources/sam3d/latex/tables/table_training_hyperparameters_geometry.tex:10-12). "61 million samples" is accurate; "objects pasted" is loose.

## Loc problems (value right)

- efficientvitsam-lvis-l2, efficientvitsam-lvis-sam, efficientvitsam-sam-map: LVIS (and COCO) with ViTDet boxes is Table 4 in the PDF, not Table 3 (Table 3 is GT boxes). PDF "Table 4. Zero-Shot Instance Segmentation Results, Prompted with ViTDet" (sources/efficientvitsam/latex/table/exp/box_vitdet.tex:25).
- viz-efficient figcaption cites "MobileSAM (v2, Table 3 and Sec. 4)". The 4 ms decoder time is in the Abstract and Sec. 1 (main_MobileSAM_arxiv_v2.tex:85, 131), not Sec. 4.

## (c) Honesty labels

- viz-sam3d: data-toy, caption "Toy model in 3D ... The placement values are ours", badge('toy') in all tabs. Correct.
- viz-efficient: no data-toy; caption says tabs 1 and 3 plot paper numbers and tab 2 is a toy; badges sourced/toy/sourced match. Correct, except item (b)1 (the sourced tab hides printed values).
- scene-efficient TIME array (452, 8, 4 ms) matches MobileSAM Table 3 and Abstract. T1 throughputs (11, 278, 183, 449, 762, 538, 182) and mAPs (46.5, 45.7, 46.6, 47.8) match efficiency.tex:8-19.

## (d) Equations

- eq-sam3d-fm: matches sources/sam3d/latex/sec/XX_supp.tex:240 term by term (the only display equation before it, so Eq. 1; it sits in App. C.2, PDF "C.2 ... (1)").
  The x_tau and v definitions in the eqno match XX_supp.tex:245. Gloss: M = {S, R, t, s} (XX_supp.tex:242); lambda R = 0.1 and s = 0.1, S = t = 1.0 (table_training_hyperparameters_geometry.tex:12-13). Correct.

## (f) Checked

ch-sam3d: all 32 sam3d.json spans (dates, 4 token sets, 1.2B, 600M, 4096 + 1 tokens, 64^3, 6D, lambda 0.1, 10/80/150 s, N=8, 80%, ~1M images, 3.14M meshes, 7M prefs, 2.7M, 61M, 1,000,
Table 4 six rows, Hi3DGen 0.1629 = best baseline F1 in Table 2, 5:1 and 6:1 from Abstract and Sec. 4.1); "the paper prints no per-benchmark percentages" (Fig. 8 is bars without printed values: true);
LLM-style recipe (sec/01_introduction.tex:15, 29); limits (App. F) and mask robustness (App. E.9, "fair robustness"); "SAM series as a fast way" (App. E.9); scene text strings; equation.
ch-efficient: all 36 spans against MobileSAM v2, EfficientSAM v1 (Table 5 rows, 75% mask ratio, ViTDet boxes), EfficientViT-SAM v2 (Table 1, Table 4, abstract 48.9x = 538/11),
EdgeTAM Table 2, EfficientSAM3 v1, MobileSAMv2 Table 2, HQ-SAM v2 (Table 1, Table 2 DIS mBIoU with GT-box prompts, Sec. 3.3); dates in parentheses (all match v1 months); scene arrays.
