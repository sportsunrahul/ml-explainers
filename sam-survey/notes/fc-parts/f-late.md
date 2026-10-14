# Fact check part: ch-family, ch-map, ch-inspection, ch-timeline, ch-summary, ch-deeper

Scope: explainer.src.html lines 2989-end, parts/scene-tree.js (inline), scene-map.js (inline), scene-inspect.js.
Method: every claim id used in these sections was checked at its loc in sources/; the scan numbers were recomputed
from notes/arxiv-scan-2026-10-09.json with the patterns in notes/scan/domains.json; every tree date was compared
with the [v1] line of sources/*/abs.html.

## (a) WRONG

1. ch-map, "[scan-pre-sam] 4 of them predate SAM and use the phrase in another sense".
   Wrong for 3 of the 4. Their v1 dates predate SAM, but their (revised) abstracts refer to SAM itself:
   EfficientViT 2205.14756 ("For Segment Anything, EfficientViT delivers 48.9x higher throughput"),
   CUTS 2209.11359v7 ("on par with Segment Anything Models (SAM, MedSAM, SAM-Med2D)"), and the skyrmion paper 2303.02876v2
   dated 2023-03-06 ("the newly developed Segment Anything Model (SAM)"). Only 1902.03715 "Towards Segmenting
   Anything That Moves" uses other words, and it matches only through arXiv stemming ("Segmenting Anything").
   Source: notes/scan/abs-0.xml and ti-0.xml <summary> of those entries. Correct: "4 records have a v1 date
   before SAM; 3 are older papers whose later revisions mention SAM, 1 is a stemming match. We drop them."
   The same error is in the scan-pre-sam claim text in notes/claims/scan.json.

2. ch-map, "Most papers ([scan-dom-other] 592) match none of the lists."
   592 of 1,443 is 41%, not a majority. Recomputed: other 592, medical 387, efficient 98, 3D 98, video 95,
   remote sensing 82, robustness 57, industrial 34 (sum 1,443). Correct: "The largest share (592, about 41%)".

3. ch-timeline list order: MobileSAM (2023-06-25) is listed before HQ-SAM (2023-06-02). Both dates are right
   (sources/mobilesam/abs.html [v1] 25 Jun 2023; sources/hqsam/abs.html [v1] 2 Jun 2023), but the list is
   chronological everywhere else. Swap the two items.

## (b) MISLEADING / over-claimed

1. ch-family, "SAM, SAM 2, SAM 3, SAM 3D, each built on its predecessor." SAM 3D does not build on SAM 3.
   SAM 3D v1 and SAM 3 v1 are the same day (2025-11-20). The SAM 3D paper builds on a two-stage latent flow
   model (sources/sam3d/latex/sec/02_method.tex:15, "We build upon ... xiang2025structured") and uses SAM v1
   only as an annotation tool (sec/XX_supp.tex:86) and "the SAM series" as a mask source (XX_supp.tex:647).
   Suggest "SAM 3D takes a mask from any SAM model as its input" and mark the sam3->sam3d tree edge as weak.

2. ch-family, "SAM-Adapter keeps SAM frozen and inserts small trainable adapters". Only the image encoder is
   frozen; the SAM mask decoder is fine-tuned (sources/samadapter/latex/neurips_2023.tex:135-137, "We keep the
   weight of pretrained image encoder frozen ... and tune the mask decoder during training").
   Also "its SAM baseline gets no prompt at all": the paper does not say how the SAM row of Table 1 was run.
   Line 137 says SAM-Adapter itself feeds no prompts to the decoder. Mark the baseline setup as our reading.

3. ch-family, SAM2Long "adding [sam2long-savtest-gain] 5.3 J&F on SA-V test without retraining". 5.3 is the
   best case, for SAM 2-L against the authors' reproduced SAM 2-L (75.5 -> 80.8). On SAM 2.1-L the gain is 1.6
   (79.6 -> 81.2), and the average over 12 comparisons is 3.7 (sources/sam2long/latex/main.tex:86, table at
   lines 214-235). Say "up to 5.3 (SAM 2-L; 1.6 on SAM 2.1-L)".

4. ch-family, Grounded SAM: "(it reaches [groundingdino-coco-zs-ap] 52.5 AP on COCO without COCO training)".
   52.5 is Grounding DINO L (Swin-L; sources/groundingdino/latex/tables/COCO.tex:41). The SGinW numbers in the
   same sentence use Grounding DINO-B (sources/groundedsam/latex/table_tex/seginw.tex:23,26). Name the L model
   or the reader assumes the chain uses it.

5. ch-family, "MobileSAM already cites it in June 2023". The downloaded MobileSAM is v2, dated 2023-07-01; the
   citation is at sources/mobilesam/latex/main_MobileSAM_arxiv_v2.tex:137. v1 (June 25) was not checked.
   Say "by July 2023" or verify v1.

6. ch-inspection, DefectSAM: "box-prompted zero-shot SAM". The paper never states the prompt type for its SAM
   baseline (sources/samdefect/latex/sec/4_Experiment.tex:59-61; only "thermal images with prompts", line 15).
   "zero-shot" is supported ("have not been exposed to the defect dataset", line 61). Drop "box-prompted".

7. ch-inspection, SAA: "Most of the gain comes from the rule filters, not from SAM". The ablation removes the
   property prompts (texture 53.79 -> 21.83, sources/saa/latex/arxiv.tex:496,499,538), but no row removes SAM,
   so "not from SAM" is not measured. Label "our reading" or cut "not from SAM".

8. ch-inspection, Industrial SAM: "five defect classes needed 1-3 clicks each". The source is one illustrative
   figure (one image per class), not a measured study (sources/industrialsam/latex/paper.tex:501, Fig. at 536).
   Say "in the example images". The next sentence ("No paper here measured the labelling time saved") is right;
   note the paper does assert, without measurement, that SAM "drastically reduces the time" (paper.tex:501).

9. ch-inspection, scene-inspect tab 0 readout states as fact: "a click on a defect that has no object-like
   boundary returns the object around it: the plate". No cited paper reports this. Not Always Perfect uses
   everything mode with oracle selection (sources/notperfect/latex/main.tex:664); DefectSAM does not show it.
   The caption ("to show the failure modes the papers below report") over-claims for this tab. Rephrase as
   "can return" and say it is our illustration. Tab 2 draws all student masks as "close"; Boxes2Pixels reaches
   defect mIoU 0.6523, so "closer" is fairer.

10. ch-map, "Part of the fall in 2026 comes from the method ... the last weeks are still being indexed".
    Plausible, but unmeasured. Label "our reading". Also the topic heuristic has small known noise: "edge" in
    the efficient pattern matches "knowledge" (5 titles, e.g. "SinkSAM-Net: Knowledge-Driven ..."), "organ"
    matches "Organizer"/"organoid" (3), and "anomal" puts "Dr-SAM ... Anomaly Detection on Angiography" and an
    Earth-observation change set into industrial, which is checked first. The page sentence "titles mention
    defects, anomalies, cracks or inspection" omits "industrial", "manufactur" and "surface", which are in the
    pattern (notes/scan/domains.json). The caption already says "heuristic", so this is minor.

11. ch-summary, "For inspection, SAM is a strong labelling tool". The only evidence on the page is qualitative
    (Industrial SAM example figure) and the page itself says no paper measured time saved. Soften to "a useful
    labelling tool" or label it our reading.

12. ch-deeper, "SAM 3D v2, Sec. 2 to 3, Eq. 1 (flow matching) and Eqs. 2 to 3 (DPO)". The equations are in the
    appendix "Details on Model Training" (sources/sam3d/latex/sec/XX_supp.tex:236-283, labels eq:fm and eq:dpo),
    not in Sec. 2 to 3. Eq. numbers 1 and 2-3 are right (the DPO align numbers its first line and an empty line).
    "Compare SAM 2 v2, App. D.2.1": D.2.1 is pre-training (IoU-loss changes); the simulated-click loop is in
    D.2.2 Full training (sources/sam2/latex/sam2.1_arxiv.tex:1074). Cite both.

13. scene-tree readout "Every paper on this page at its first arXiv date": the tree also holds papers the page
    never discusses (Concealed scenes, MedSAM2 appear only in the tree data). Minor. The edge "sam -> kairosad"
    should be "mobilesam -> kairosad" (sources/kairosad/latex/0_main.tex:72, built on MobileSAM).

## (c) Honesty labels

- viz-tree: no data-toy, badge 'sourced', caption says dates are v1 and lines are our reading. Correct.
  All 47 dates in the tree data match sources/*/abs.html [v1] (checked every one, including Cutie 2023-10-19,
  Hiera 2023-06-01, FocalClick 2022-04-06, EfficientSAM3 2025-11-19, MedSAM2 2025-04-04, Concealed 2023-04-12).
- viz-map: badge 'sourced'. MONTHS and DOMAINS in the JS equal the recomputation from the scan JSON exactly.
- viz-inspect: data-toy, badge 'toy', caption "Toy model". Label correct; see (b)9 on the caption wording.

## (d) Equations

None in these sections.

## (f) What was checked

- ch-family: groundingdino-coco-zs-ap (COCO.tex:41, intro:50), groundedsam-sginw-n (03_experiments.tex:6),
  groundedsam-sginw and -hq-sginw (seginw.tex:23,26), groundedsam-published (abs.html), seem-train-images and
  seem-sam-images (main_table1.tex:54-64), medsam-pairs and -modalities (main-medsam.tex:195), medsam-npc-dsc
  (main-medsam.tex:253, external validation paragraph), samadapter-cod10k(-sam) (neurips_2023.tex:196,198,
  COD10K S_alpha column), trackanything-davis17 (neurips_2022.tex:217,268), sam2long-savtest-gain (main.tex:230),
  samurai-(sam21-)lasot-auc (compare_wrt_sam.tex:17,19). All values hold.
- ch-map: all 18 scan-* values reproduce from the records (1,447 records, 1,447 unique ids; 1,443 kept; years
  302/490/426/225; peak 2024-08 = 63; domains as above).
- ch-inspection: notperfect-cds2k-* (main.tex:459, 616-625, 665), samdefect-* (4_Experiment.tex:51,74,76; F is
  F2 with recall weight, 3_Method.tex:93), saa-* (arxiv.tex:496-499), boxes2pixels-* (4_expirements.tex:49-51,
  106-107,167; trained on identical SAM pseudo-masks, caption line 38), industrialsam-borescope-clicks
  (paper.tex:501), kairosad-* (4_experiments.tex:45,65,80; supervised, image-level only: 1_intro.tex:27,
  2_related.tex:22). All values hold. Side note, not on the page: the Boxes2Pixels abstract says +6.97 anomaly
  mIoU, the text says +0.0642 (abstract vs 4_expirements.tex:57).
- ch-timeline: all 19 tl-* dates match abs.html [v1].
- ch-summary, ch-deeper: section, appendix and figure pointers checked (SAM App. A paragraphs segany.tex:652,658;
  SAM Fig. 14 = fig:model_diagram_zoomin; SAM 3 Sec. 5 holds cgF1 main.tex:543; App. C.2 and App. D exist).
