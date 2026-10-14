# Fact check of the finished pages (2026-10-11)

Scope: explainer.src.html (all sections, inline scripts, parts/scene-*.js and viz-*.js) and atlas/chapters.html (plus atlas hero, chips, dialogs).
Method: seven independent checkers, one per chapter group plus the atlas. Each claim used on the page was re-read at its loc in sources/ (LaTeX first, then PDF, abs.html or README). Each sentence around a number was checked for model, dataset, split, metric, setup and direction. Display equations were compared with the cited LaTeX. Figure badges, data-toy attributes and figcaptions were compared with each other.
Value-equals-claim was not re-checked (a script already did it). The sam-data PDF page errors from notes/factcheck-claims.md are not repeated here.

Tally (counted from this file):
- Long page: 6 WRONG, 62 MISLEADING or location items, 2 honesty-label problems (viz-stages 72->100 line; viz-efficient sourced tab nulls printed values, listed as ch-efficient (b)1), 0 equation errors.
- Atlas: 3 WRONG, 9 MISLEADING.
- Caveat: the (a) layer-wise-decay item rests on the standard definition (BEiT, ELECTRA). Those papers are not in sources/; SAM does not define the term (segany.tex L664).
- File state: atlas/chapters.html and atlas/scenes.js have mtime 2026-10-11 21:42, during this check. No checker was told to edit them; the change is probably a concurrent atlas rebuild. The atlas findings quote the current file.

Global checks:
- 531 data-claim spans, 491 unique ids. Every id exists in notes/claims/*.json and in notes/claims.json.
- No uncertain claim is shown on the page. The two code_only ids (sam3-bench-nps-readme 270K, sam3-params-readme 848M) are both attributed to the README in their sentence.
- backbones.json: 0 ids used on the page.
- ViT-H parameters appear as 636M (SAM Fig. 13, ch-sam-arch inline table, line ~1679) and as 632M (MobileSAM Tab. 1, ch-efficient). Both are right for their source; the page does not note the difference. Listed under (b).

## (a) WRONG

### hero/intro, ch-before, ch-clicks, ch-promptable

1. ch-clicks, "RITM (2021) added the two ideas SAM later reuses. ... Second, it trains *iteratively*: during training it runs the model, finds the error region, places the next click there, and repeats".
   RITM did not add either idea. It says: "We revive an iterative training procedure, and make a network aware of the mask from a previous step [Mahadevan et al., ITIS 2018]" (sources/ritm/latex/main.tex:98). It adopts the ITIS iterative sampling with changes (main.tex:184: eroded-region sampling with 4x less area, and a mix of random and iterative clicks). ITIS also proposed the previous-mask input as an optional extension (main.tex:191). RITM's contribution is to make that input work: the mask is a third input channel (main.tex:193), and RITM reports better stability.
   Correct statement: "RITM (2021) revived and refined two ideas from ITIS (2018) that SAM later reuses: iterative click simulation in training, and the previous mask as an extra input." (Merge note: one checker first called the previous-mask input RITM's own idea; main.tex:191 contradicts that.)

### ch-sam-arch, ch-training, ch-engine, ch-zeroshot

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

### ch-sam2

None found. Every number checked holds at its loc, and no sentence states something the source contradicts outright.

### ch-sam3

None found. Every checked value matches its loc, and no sentence states something the source contradicts outright.

### ch-sam3d, ch-efficient

None. Every checked value matches its source at the stated place (apart from the loc notes below).

### ch-family, ch-map, ch-inspection, ch-timeline, ch-summary, ch-deeper

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

## (b) MISLEADING / over-claimed (and location problems)

### hero/intro, ch-before, ch-clicks, ch-promptable

1. Intro (line ~265), "Every model on this page answers the same request. Someone points at an image with a prompt (a click, a box, a word), and the model returns a mask". FCN, U-Net, Mask R-CNN, MaskFormer and Mask2Former take no prompt; ch-before itself says they label a fixed class list. SAM 3D returns a 3D shape, not a mask. Suggest "Every model in the SAM line answers ...", or "Most models on this page".
2. ch-before, "Before 2023, segmentation models learned to draw masks for a fixed list of classes, from one labelled dataset." The next chapter (interactive segmentation, DIOS 2016 onward) contradicts this, and the page says so itself ("A second line of work never asked a model to know classes"). Suggest "Before 2023, most segmentation models ...".
3. ch-before, FCN "On PASCAL VOC the mean overlap (mean IU) went from 59.4 at stride 32 to 62.7 at stride 8." Both numbers are on "a subset of PASCAL VOC2011 validation" (sources/fcn/latex/paper.tex:590, Table 2 rows :598, :600), and 59.4 is with extra training data (paper.tex:483). The values are right; add "on a VOC2011 val subset".
4. ch-before, Mask2Former "Removing it changes COCO instance AP by -5.9. With it, one architecture matched or beat the specialised models on all three tasks: 57.8 PQ ..., 50.1 AP ... and 57.7 mIoU". The -5.9 is an R50 ablation (43.7 -> 37.8, sources/mask2former/latex/sections/expts.tex:154-156, caption :242). The three headline numbers use Swin-L with 200 queries and 100 epochs for COCO (expts.tex:25, :63), and Swin-L-FaPN, multi-scale, for ADE20K (expts.tex:135). The paragraph just before says "100 [queries] by default". This is not wrong, but name the setup ("best Swin-L models") so the reader does not tie the headline numbers to the R50/100-query default.
5. ch-before, U-Net "It is still the shape of most dense-prediction networks today." Unsourced opinion stated as fact. Mark it as our reading or soften.
6. ch-before table, MaskFormer ADE20K-Full 847. MaskFormer itself also says 874 (sources/maskformer/latex/sections/4.experiment_v2.tex:9) against 847 in Table 2 header (:102) and later text (:136, :246). The claim record notes the conflict; the page does not. Low priority (a footnote or "about 850" would do).
7. ch-clicks, DIOS "negative clicks sampled by 3 different rules around it". Strategy 2 samples negatives on other objects in the same image, not only around the target (DIOS paper.pdf p3, Sec. 3.2, Strategy 2; pdftotext line ~142). "around it" fits strategies 1 and 3 only. Suggest "in the background and on other objects".
8. ch-clicks, "DIOS needed 6.04 clicks on the GrabCut images; RITM, five years later, 1.56." Same metric and threshold (NoC@90, GrabCut; DIOS Table 1 p6; RITM main.tex:512), and RITM says it follows the DIOS click protocol (main.tex:320). But training data differ (DIOS: PASCAL; RITM: COCO+LVIS, 104k images). The comparison is fair as a trend; one clause on the different training data would make it exact. Low priority.
9. ch-promptable, "After training, other systems talk to SAM through the same request: a detector sends boxes, a person sends clicks, a text model sends words." In SAM v1 text prompts are a proof of concept with a modified training procedure (sources/sam/latex/segany.tex:513, "proof-of-concept"; "for this one SAM's training procedure is modified"). The released SAM takes no text. Suggest "and, in a proof of concept, a text model sends words".
10. ch-promptable, "Over many ambiguous clicks, each slot settles on one level of the nesting." The paper does not state that slots specialise to whole/part/subpart; it only says three layers are "often enough" (segany.tex:652) and that the model "will return the subpart, part, and whole object" (segany.tex:208). Label as our reading.

### ch-sam-arch, ch-training, ch-engine, ch-zeroshot

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

### ch-sam2

1. ch-sam2, archflow stage "7. Where the gradient goes": "In the 16-frame fine-tune, Hiera is frozen ... the FPN
   still learns." The JS readout repeats it ("the FPN and everything after it learn"). The paper says only "freeze the
   image encoder" (sam2.1_arxiv.tex:1078). App. D.1 describes the FPN inside the "Image encoder" paragraph
   (sam2.1_arxiv.tex:954), so the paper's "image encoder" most plausibly includes the FPN. The page states the FPN
   detail as fact with no "our reading" label. Fix: say "the image encoder is frozen" and either drop the FPN clause or
   label it our reading.
2. ch-sam2, "What memory buys", third bullet heading "The pointers matter, the recurrence does not." The paper says
   object pointers "do not enhance average performance across the 9 zero-shot datasets" and help on SA-V val and
   LVOSv2 (sam2.1_arxiv.tex:898). In Table 11 the no-pointer row is best on the 9 zero-shot average (70.9 vs 70.7)
   and on MOSE dev (73.1 vs 73.0) (sam2.1_arxiv.tex:909-911). The GRU row (65.3) is above the no-pointer row (64.5)
   on SA-V val, so "the recurrence does not [matter]" is also a simplification. Fix: "Pointers help on SA-V val and
   LVOSv2, not on the 9 zero-shot average."
3. Same bullet: 68.3 / 64.5 / 65.3 / 69.0 / 0.93 are from the ablation setup ("a smaller model setup with 512 input
   resolution", sam2.1_arxiv.tex:708), not the released 1024 model. The page does not say so; a reader can compare
   68.3 with the 77.0 / 76.8 Table 6 numbers in the previous bullet. Add "(ablation setup, 512 px)".
4. ch-sam2, SA-V paragraph: "That is 53x more masks than any earlier video segmentation dataset." The 53x counts the
   automatic masklets; without them it is 15x (sam2.1_arxiv.tex:399, 450, macro L154). The paper compares with VOS
   datasets in Table 3 ("the largest VOS dataset"), not all video segmentation datasets. Add "with automatic masklets
   (15x manual only)" and say "video object segmentation dataset".
5. ch-sam2, "Fewer clicks" bullet: "SAM 2 reaches 64.7 J&F". Table 4 reports J&F except on VOST, where it reports J
   (sam2.1_arxiv.tex:535). Minor; "average accuracy" is the paper's wording (Table 4 caption, L528).
6. ch-sam2, claim card "Claim": "without making them worse". The paper says "comparable quality" and "better
   alignment" than Phase 2 (sam2.1_arxiv.tex:230, 375); 10.9% of Phase 3 masks fall below IoU 0.75 vs Phase 1
   (Table 1, L362). The card tests the claim honestly, so this is wording only. Prefer "at comparable quality".
7. ch-sam2, "Two details carry a lot": "a correction click must become a memory that outranks them". The paper keeps
   prompted memories in a separate FIFO of up to M frames without temporal codes (sam2.1_arxiv.tex:319, 323); it
   says nothing about ranking. Label as our reading or say "is kept in the prompted-frame queue".

Checked and fine (no action): the occlusion head (L300, L968); prediction of multiple masks and propagation of the
highest-IoU one (L298); 6 recent memories with temporal codes, prompted without (L323); 64-dim memory, 256-dim
pointer split into 4 x 64 (L973); stride 16+32 to FPN, 4+8 skips (L954); 2d-RoPE in memory attention (L956);
learned occlusion embedding (L968); loss ratio and supervision rules (L1080); IoU on all masks, 7 correction
clicks (L988); AdamW, ~150k, 6e-5 / 3.0e-4, batch 256 (Table 12b, L1047-1062); 16-frame 50k at half lr
(L1078); 256 A100 x 108 h for "the released SAM 2" (L1782); up to 2 prompted frames of 8, GT mask / click / box
(L326); "The paper does not state how many prompted memories are kept" is true (M has no value, L319). The
limitations quotes are verbatim (App. C, L939, L941). Chapter cross-references (ch. 4, 7, 12) point to the right
chapters.

### ch-sam3

1. "The paper gives three human numbers for this split: 72.8 ..., 76.2 for an oracle on a three-reviewer subset (Table 29) and 70.8 on the internal version". Table 29 gives a fourth number on the same subset: Random Pair cgF1 55.5 (main.tex:2381). It is the one closest to SAM 3's 54.1, so leaving it out makes the human gap look larger. SAM 3's Gold score uses the oracle protocol (compared against all three annotations, main.tex:2362 and 549), so 76.2 (oracle) is the like-for-like human number. Say "four" and add 55.5 (Random Pair), or drop the count.
2. "With one exemplar and no text, SAM 3 beat the earlier exemplar model T-Rex2 by +18.3 AP on COCO (Table 3)." The metric is AP+ (evaluated only on positive examples), column I: 76.8 vs 58.5 (main.tex:676-680, caption main.tex:688, text main.tex:705-707). Write "AP+ (positives only)".
3. "on RF-100VL ... ten labelled examples (10-shot) lift it to 36.5 AP (Table 2)". This is box AP after fine-tuning (40 epochs, masks disabled) on the official FSOD 10-shot splits, averaged over 100 datasets (main.tex:2740, Table 36 main.tex:2814-2832). "Ten labelled examples" understates the setup (10-shot is per class, per dataset). Zero-shot SAM 3 scores 15.2 here, below gDino-T 15.7 (main.tex:649-651); the page omits that.
4. "On LVIS ... 48.5 zero-shot mask AP in Table 1, against 38.5 for the best earlier model". 38.5 is DINO-X, copied from its own paper (dagger, main.tex:615). Table 1 has higher LVIS mask AP numbers that are not zero-shot (APE-D* 53.0, OWLv2* 43.4, gray, trained on LVIS; main.tex:611-614). Say "best earlier zero-shot model (DINO-X, from its paper)".
5. "Phase 3: a Wikidata ontology of 22.4M nodes and 15 domains." The 15 is the number of SA-Co/HQ annotation domains (datasets), not a property of the ontology (main.tex:509; Table 19 main.tex:1679). The ontology has 17 top-level categories and 72 sub-categories (main.tex:511). Split into two facts.
6. Claim card: "The human-labelled set has 4M unique phrases on 5.2M images with 52M masks". SA-Co/HQ comes from data engine phases 1-4, in which AI verifiers auto-label a growing share (main.tex:494-498, 526). "Human-labelled" over-states it; use "high-quality (human and AI-verified) set".
7. Claim card, Phase 1: "people verify masks proposed by a captioner, a detector and SAM 2". The captioner and parser propose the phrases; SAM 2 prompted by an open-vocabulary detector proposes the masks (main.tex:492). Minor.
8. Table 10 sentence: "gave 54.0 on the paper's main benchmark (cgF1 ...) against 52.2 ...". Table 10 is an ablation run (stride-28 model, shorter schedule, internal Gold version; main.tex:1138). The best row in Table 10 is c (54.9, also supervise the total score; main.tex:1103), which the page does not mention. Add "ablation setting" and note row c.
9. Aside "Version 2 adds Object Multiplex" / "The version 2 appendix (released as SAM 3.1)". Only v2 was downloaded, so "v1 lacked it" is inferred (earlier claims check flagged the same for sam3-v2-date). The README dates SAM 3.1 Object Multiplex 03/27/2026 (sam3.README.md:60). Also the 5.2x and 7% use "optimized internal implementations" (multiplex.tex:81 footnote). Label "our reading" for the v1/v2 attribution.
10. Tracker figure, step 1: "Training uses single frames and short clips". Not stated in the source; App. C.4 Stage 4 gives only the 16- and 32-frame fine-tune (main.tex:1592). Step 3: "Past frames where the object was confidently present are stored". The paper says the memory bank holds past frames and conditioning frames, and that the occlusion score "may also be used" to select frames (main.tex:1519, 1521). Soften both.
11. Detector figure, step 7: "The vision encoder learns more slowly than the rest: 5e-4 ... against 1e-4 for the text encoder and 1e-3 for the rest." The text encoder is the slowest; the sentence implies the vision encoder is. Minor wording (values right, main.tex:1582).
12. Hard-negative paragraph: "Only the presence score moves ... The paper's ablation shows the same pattern." Table 9b reports IL_MCC and pmF1 of whole models (main.tex:917-931); it does not isolate the presence score. Mark the mechanism as our reading.

Checked and fine: sam3-bench-nps-readme (270K) and sam3-params-readme (848M) are both attributed to the README in the sentence ("the README says"); sources: sam3.README.md:49 ("270K unique concepts") and :199 ("848M parameters"). Paper 207K (main.tex:223, 537) and ~850M (main.tex:1392) are stated alongside.

### ch-sam3d, ch-efficient

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

Location problems (value right):

- efficientvitsam-lvis-l2, efficientvitsam-lvis-sam, efficientvitsam-sam-map: LVIS (and COCO) with ViTDet boxes is Table 4 in the PDF, not Table 3 (Table 3 is GT boxes). PDF "Table 4. Zero-Shot Instance Segmentation Results, Prompted with ViTDet" (sources/efficientvitsam/latex/table/exp/box_vitdet.tex:25).
- viz-efficient figcaption cites "MobileSAM (v2, Table 3 and Sec. 4)". The 4 ms decoder time is in the Abstract and Sec. 1 (main_MobileSAM_arxiv_v2.tex:85, 131), not Sec. 4.

### ch-family, ch-map, ch-inspection, ch-timeline, ch-summary, ch-deeper

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

## (c) Honesty-label problems

### hero/intro, ch-before, ch-clicks, ch-promptable

All five figures in scope carry data-toy and badge('toy', ...): viz-hero (scene-overview.js:49, "The scores are toy values"), viz-before (scene-before.js:121), viz-clicks (scene-clicks.js:60), viz-promptable (scene-promptable.js:123). Captions say "Toy model" (clicks, promptable) or "A toy desk scene ... grid sizes are illustrative" (before). No problems.
Note: the clicks scene claims "The simulated user in the figure follows the same rule" (RITM centre-of-largest-error, point farthest from the boundary). scene-clicks.js:27-31 implements it with a 4-neighbour BFS distance (Manhattan, not Euclidean) and counts the image border as boundary. Close enough for a toy; no change needed.

### ch-sam-arch, ch-training, ch-engine, ch-zeroshot

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

### ch-sam2

- viz-sam2: data-toy, figcaption starts "Toy model", SCENES.sam2 calls badge('toy', ...). Consistent. The staged
  drift after the occlusion is disclosed in the caption.
- viz-sam2flow (ArchFlow): data-computed, badge "Computed", figcaption "Computed: ..." with the 64x64 memory size and
  1/4-resolution logits labelled our reading. Hiera stage widths and depths in the JS (T [96..768] [1-2-7-2], S
  [1-2-11-2], B+ [112..896] [2-3-16-3], L [144..1152] [2-6-36-4]) match hiera_arxiv.tex:391-395. Spatial sizes
  256/128/64/32 follow from a 1024 input and strides 4-32. Consistent.
- Toy readout on frame 1: "segmented exactly as SAM would" is an approximation inside a toy-labelled figure; no
  action needed.

### ch-sam3

- viz-sam3: data-toy, figcaption begins "Toy model", scene uses badge('toy', ...). Correct.
- viz-sam3flow and viz-sam3trk: badge "Computed", data-computed. Values shown are paper config values plus the 72 x 72 grid, which is labelled "our arithmetic". Consistent with the page-authoring rule for ArchFlow readouts. No problem.
- 72 x 72 detector grid: supported by main.tex:1138 (the ablation stride-28 variant uses 4x fewer detector tokens than SAM 3, so the default is stride 14, 1008/14 = 72).

### ch-sam3d, ch-efficient

- viz-sam3d: data-toy, caption "Toy model in 3D ... The placement values are ours", badge('toy') in all tabs. Correct.
- viz-efficient: no data-toy; caption says tabs 1 and 3 plot paper numbers and tab 2 is a toy; badges sourced/toy/sourced match. Correct, except item (b)1 (the sourced tab hides printed values).
- scene-efficient TIME array (452, 8, 4 ms) matches MobileSAM Table 3 and Abstract. T1 throughputs (11, 278, 183, 449, 762, 538, 182) and mAPs (46.5, 45.7, 46.6, 47.8) match efficiency.tex:8-19.

### ch-family, ch-map, ch-inspection, ch-timeline, ch-summary, ch-deeper

- viz-tree: no data-toy, badge 'sourced', caption says dates are v1 and lines are our reading. Correct.
  All 47 dates in the tree data match sources/*/abs.html [v1] (checked every one, including Cutie 2023-10-19,
  Hiera 2023-06-01, FocalClick 2022-04-06, EfficientSAM3 2025-11-19, MedSAM2 2025-04-04, Concealed 2023-04-12).
- viz-map: badge 'sourced'. MONTHS and DOMAINS in the JS equal the recomputation from the scan JSON exactly.
- viz-inspect: data-toy, badge 'toy', caption "Toy model". Label correct; see (b)9 on the caption wording.

## (d) Equation problems

### hero/intro, ch-before, ch-clicks, ch-promptable

- eq-maskdot (ch-before), "MaskFormer v2, Sec. 3.3 (unnumbered)": matches sources/maskformer/latex/sections/3.method_v2.tex:56 term for term ($m_i[h,w] = \text{sigmoid}(\mathcal{E}_\text{mask}[:,i]^T \cdot \mathcal{E}_\text{pixel}[:,h,w])$). It is inline in the paper, in Sec. 3.3 "MaskFormer" (subsection at :37). Gloss: 256 channels (4.experiment_v2.tex:39), stride-4 pixel decoder (4.experiment_v2.tex:35). OK.
- eq-noc (ch-clicks), "our notation" for DIOS Sec. 4.1: faithful. DIOS caps at 20 and thresholds at 20 when not reached (DIOS paper.pdf p6, Sec. 4.1). RITM uses the same NoC@85/90 (ritm main.tex:320). OK.
- eq-samloss (ch-promptable), "our transcription" of SAM App. A: faithful to segany.tex:652 (min over three, backprop lowest loss) and :656 (focal:dice 20:1, IoU head MSE, scale 1.0). The mean over all three IoU terms is labelled our reading. One more unstated choice: whether the min is over the mask loss only or mask+IoU loss. The gloss says mask loss; the paper says "compute the loss (described shortly) ... only backpropagate from the lowest loss", which does not settle it. Already inside the "our transcription" label; optional note.

### ch-sam-arch, ch-training, ch-engine, ch-zeroshot

- ch-sam-arch, M_k[h,w] = MLP_k(t_k)^T U[:,h,w], labelled "Our notation for SAM v1, App. A". Faithful to L646
  ("a small 3-layer MLP ... spatially point-wise product between the upscaled image embedding and the MLP's output").
  Note: the paper speaks of one MLP; the per-token subscript k (a separate MLP per mask token) comes from the code,
  not the paper. Acceptable under "our notation"; U = 32 channels at 1/4 resolution matches L646, L650.
- No other display equations in these four sections.

### ch-sam2

- eq-memread: labelled "Standard scaled dot-product cross-attention (our notation)". The paper prints no equation
  and describes the block in words (Sec. 4, sam2.1_arxiv.tex:293). The transcription is faithful. It omits the
  residual, the output projection and the self-attention / MLP sublayers, which the eqno text names. "Sec. 4" is the
  correct section number (Model is \section 4, L279). Fine.

### ch-sam3

- eq-sam3-presence: matches main.tex:1434-1435 term by term. Label "App. C.2, Presence Head (unnumbered display)" is right.
- Video propagate/detect/match_and_update: matches main.tex:434-441. "Sec. 3 (unnumbered)" is right (Sec. 3 Model, paragraph Tracker and Video Architecture).
- cgF1 = 100 * pmF1 * IL_MCC: matches main.tex:2352-2354. "App. E.3, Eq. 3" is right (Eq. 1 pmF1 main.tex:2340, Eq. 2 IL_MCC main.tex:2346). Gloss descriptions of Eq. 1 and Eq. 2 are faithful.

### ch-sam3d, ch-efficient

- eq-sam3d-fm: matches sources/sam3d/latex/sec/XX_supp.tex:240 term by term (the only display equation before it, so Eq. 1; it sits in App. C.2, PDF "C.2 ... (1)").
  The x_tau and v definitions in the eqno match XX_supp.tex:245. Gloss: M = {S, R, t, s} (XX_supp.tex:242); lambda R = 0.1 and s = 0.1, S = t = 1.0 (table_training_hyperparameters_geometry.tex:12-13). Correct.

### ch-family, ch-map, ch-inspection, ch-timeline, ch-summary, ch-deeper

None in these sections.

## (e) Atlas problems (atlas/chapters.html and atlas hero/chips/dialogs)

Method: every number and factual sentence was matched to a claim in notes/claims.json and then to the
source line. Paths below are relative to sources/.

### WRONG

1. Panel 8 (SAM for inspection): "even with the best of its masks picked by hand, SAM ViT-H has a mean
   absolute error of 0.265". The masks were not picked by hand. The paper runs SAM in automatic
   (everything) mode to get N masks and selects the mask with the highest IoU against the ground truth
   (notperfect/latex/main.tex:664). Correct: "even when the best of its automatic masks is chosen by
   comparing each with the ground truth (an oracle upper bound)". The long page states this correctly
   (explainer.src.html:3151). The preceding "Clicked once, SAM tends to return the object around the
   defect" also does not match this protocol: the 0.265 row uses everything mode, not one click.

2. Panel 4 (Any prompt, a valid mask): "With two clicks or a box, the ambiguity is much rarer, and a
   fourth output returns one mask." The paper ties the fourth token to "more than one prompt"; the
   fourth mask "is never returned for a single prompt" (sam/latex/segany.tex:654). In SAM's training a
   box is one prompt: the first prompt is "either a foreground point or bounding box"
   (segany.tex:658). The paper does not say a box alone uses the fourth token. The long page avoids
   this: its Box mode is labelled toy (explainer.src.html:1414). Correct: "With two or more prompts
   (for example two clicks), ambiguity is much rarer, and a fourth output returns one mask."

3. Panel 3 (A human in the loop): "RITM (2021) added the two ideas SAM reuses: it feeds the previous
   mask back in, and it trains by simulating a user who clicks on the errors." RITM did not add either
   idea. Its title is "Reviving Iterative Training with Mask Guidance", and it says "We revive an
   iterative training procedure, and make a network aware of the mask from a previous step
   [Mahadevan et al. 2018]" (ritm/latex/main.tex:98). Both the iterative click simulation and the
   previous-mask input are from Mahadevan et al. (ITIS, 2018) (ritm/latex/main.tex:184, 191).
   Correct: "RITM (2021) revived and refined two ideas from ITIS (2018) that SAM reuses: ...". SAM
   itself cites RITM and Forte et al. for the simulated setup (segany.tex:658).

### MISLEADING / over-claimed

4. Panel 2, Mask2Former paragraph: "A fixed set of queries (100 by default) ... 57.8 PQ on COCO
   panoptic." The paragraph covers MaskFormer and Mask2Former, but 57.8 PQ is Mask2Former with a
   Swin-L backbone, 200 queries and 100 epochs (mask2former/latex/sections/expts.tex:25; abs.tex:11).
   MaskFormer Swin-L gets 52.7 PQ in the same table (expts.tex:23). Name the model and backbone, and
   do not set the number next to "100 by default".

5. Panel 2 takeaway: "Every model before SAM learned a fixed list of classes from one labelled
   dataset, and was retrained for each new task." The next panel contradicts it: interactive
   segmentation before SAM needs no class list (RITM 2021, SimpleClick 2022). The long page limits the
   claim to "segmentation models" of that family (explainer.src.html:1057, 3261). Correct: "Every
   model in this panel ...".

6. Panel 6 (SAM 2): "a memory bank of the 6 most recent frames and of every frame you prompted". The
   paper keeps "a FIFO queue of up to M prompted frames" (sam2/latex/sam2.1_arxiv.tex:319). "Every"
   over-claims. (Note for the page fork: explainer.src.html:2239 says "The paper does not state how
   many prompted memories are kept"; the paper names the bound M but gives no value.)
   Merge note: the long page is right. The paper keeps a FIFO of up to M prompted frames (sam2.1_arxiv.tex:319) and gives no value for M, so "does not state how many" (explainer.src.html:2239) holds. Only the atlas over-claims; do not change the long page.

7. Panel 7 (SAM 3): "Splitting 'is it here?' from 'where is each one?' is what lets SAM 3 say 'no
   yellow mug'". The paper credits both the presence head and training with hard negatives for
   rejecting absent concepts: the presence head is "especially effective when training with
   challenging negative phrases" (sam3/latex/main.tex:353), and hard negatives separately improve
   IL_MCC (main.tex:983). "Is what lets" is our reading; soften to "helps SAM 3 say".

8. Panel 8: "For thin defects, HQ-SAM's fine-detail token helps." No source in the download tests
   HQ-SAM on defects. HQ-SAM evaluates on general fine-structure sets; the only inspection paper that
   mentions HQ-SAM (industrialsam/latex/paper.tex:132) only cites it in related work. Label as our
   reading, or cite Boxes2Pixels' finding that SAM misses thin cracks (boxes2pixels/latex/sec/
   3_method.tex:7) as the motivation only.

9. Panel 8: "Clicked once, SAM tends to return the object around the defect." The source for this
   behaviour (Concealed, concealed/latex/main.tex:297, "SAM's behaviour appears to segment the main
   part of the object, such as the screw ... and bottle") used the online demo with no stated prompt
   mode (main.tex:293). "Clicked once" is not stated. Minor.

10. Panel 8: "the industrial defect set CDS2K ... 0.089 for a dedicated model". DGNet is a camouflaged
    (concealed) object detector, the best of the listed SOTA rows on CDS2K (notperfect/latex/
    main.tex:619), not a defect-specific model. CDS2K is a "concealed defect segmentation dataset"
    (main.tex:459). Minor; "the best dedicated model in the table (DGNet)" is safer.

11. Panel 3: "RITM needs 5.34 on DAVIS, SimpleClick 4.78." Both are correct and comparable (both
    trained on COCO+LVIS, DAVIS NoC@90). The RITM row is H32 IT-M (ritm/latex/main.tex:512, DAVIS
    NoC@90 column) and SimpleClick is ViT-H (simpleclick/latex/main.tex:266). The variants are not
    named; the ViT-H vs HRNet-32 size gap is large. Minor.

12. Panel 6: "SAM 2 is about 6 times faster than SAM". The 6x compares SAM 2 Hiera-B+ with SAM ViT-H
    (sam2/latex/sam2.1_arxiv.tex:1442; abstract). Hiera-L is 3.4x. Variant not named. Minor.

### Checked and correct

- Panel 1: three masks with predicted scores (segany.tex:652); prompts click / box / rough mask.
- Panel 2: FCN 59.4 -> 62.7 mean IU, FCN-32s vs FCN-8s on a VOC2011 val subset (fcn/latex/paper.tex:598-600;
  59.4 uses extra training data, paper.tex:483). U-Net 30 EM training images (unet/latex/paper459.tex:163)
  and 2 output classes (Fig. 1). Mask R-CNN 28x28 mask, 80 COCO classes. Queries 100 by default.
  Class table 21 / 80 / 133 / 150 matches claims fcn-voc-classes, maskrcnn-mask-classes,
  m2f-coco-pan-classes, m2f-ade-classes. Years 2014 / 2015 / 2017 / 2021 match the timeline dates.
- Panel 3: NoC@90 definition and the click rule "centre of the largest error region" (ritm/latex/main.tex:320).
- Panel 4: loss on all three masks, backprop from the lowest (segany.tex:652); ~50 ms in a browser on CPU
  (claim sam-decoder-runtime); ViT-H encoder runs once per image.
- Panel 5: 1.1B masks, 11M images, 99.1% automatic, 20 -> 44 -> 72 masks per image, 32x32 grid,
  confident / stable / NMS (segany.tex:208), 95% whole-image filter (segany.tex:675), holes and
  components < 100 px (segany.tex:677). "Up to three masks" per point is derived from the 3 output
  tokens (Sec. 4 says "a set of masks"; already noted in factcheck-claims.md).
- Panel 6: N = 6 recent memories (claim sam2-n-recent); occlusion head (sam2.1_arxiv.tex:300, 968).
- Panel 7: concept prompt = noun phrase and/or exemplar, all instances (sam3/latex/main.tex:290); DETR-based
  detector with object queries (main.tex:353, 407-409); final score = p(match | present) * p(present)
  (main.tex:1433-1436).
- Panel 8: Boxes2Pixels mIoU_anom 0.6523 vs SegFormer-B2 0.5881 with "identical box-only supervision via
  SAM pseudo-masks" (boxes2pixels/latex/sec/4_expirements.tex:39-51, 57); 6.20 ms on an H100
  (4_expirements.tex:167, 174); SAM used offline only (3_method.tex:51).
- hero.html: 1,447 records, scan date 2026-10-09 (notes/arxiv-scan-2026-10-09.json: 1447 records).
- dialogs.html: arXiv versions match sources/*/ID (SAM v1, SAM 2 v2, SAM 3 v2, SAM 3D v2, etc.); image encoder
  64x64x256 at 1024x1024; stability score definition (segany.tex:675); occlusion head; presence product.
- chips.html: labels only, no facts.

Note from the ch-sam-arch..ch-zeroshot checker:

Not in scope for this fork.

## (f) What was checked, per chapter

### hero/intro, ch-before, ch-clicks, ch-promptable

Scope: explainer.src.html lines 216-1540 (hero, intro, colour key, badge note, ch-before, ch-clicks, ch-promptable) and parts/scene-overview.js, scene-before.js, scene-clicks.js, scene-promptable.js.

- Hero/intro: arXiv versions in the paper list against sources/*/ID (all 9 match). Colour key and badge note: no factual content.
- ch-before: all 17 claims (fcn-*, unet-*, maskrcnn-*, maskformer-*, m2f-*) against loc: FCN paper.tex:483, 530, 567, 590-600; U-Net paper459.tex:163 (30 EM images), Fig. 1 (2 classes); Mask R-CNN maskrcnn_arxiv.tex:105, 287-290 (28^2 x 80); MaskFormer 4.experiment_v2.tex:9, 35, 39, 102; Mask2Former expts.tex:15-25, 63, 93 (100 queries), 135, 156, 242, 450-453 (Sec. 4.6 quote verbatim). All values correct. Prose qualifiers checked (above).
- ch-clicks: all 12 interactive claims (dios-*, ritm-*, simpleclick-*) against DIOS PDF pp2-6 (2 maps, truncate 255, 3 strategies, cap 20, 6.04 at 90% IU GrabCut), RITM main.tex:184 (4x less area), :186, :193 (third channel), :267 (104k, 1.6M), :455 (N_iters=3), :512 (H32 IT-M C+L: GrabCut 1.56, DAVIS 5.34); SimpleClick main.tex:168 (symmetric patch embedding for clicks + previous mask), :174 (MAE), :198/:266 (ViT-H DAVIS NoC90 4.78); FocalClick focalclick.tex:72, :125 (local crops). RITM quote at main.tex:320 verbatim. Dates: DIOS 1603 (2016), RITM 2102 (2021), SimpleClick 2210 (2022), FocalClick 2204 (2022).
- ch-promptable: all 8 claims (sam-ambiguity-masks, sam-nest-depth, sam-focal-dice, sam-iou-loss, sam-iou-weight, sam-fourth-token, f1-sam-no-deep-sup, f1-sam-rarer) against segany.tex:652-656, all correct. Quotes "even when a prompt is ambiguous ... reasonable mask for at least one of those objects" (segany.tex:153, Sec. 2), "since we observe that three layers ...", "only backpropagate from the lowest loss" (:652), "is never returned for a single prompt and is the only mask returned for multiple prompts" (:654): all verbatim. Prompt list (points, box, mask, free-form text, :153) correct.
- Scene readout strings in scene-before/clicks/promptable/overview: no sourced numbers; statements consistent with the papers (average over masks: segany.tex:652; fourth token: :654).

### ch-sam-arch, ch-training, ch-engine, ch-zeroshot

Sources: sources/sam/latex/segany.tex (SAM v1), sources/sam2/latex/sam2.1_arxiv.tex, sources/vit/latex/04_experiments.tex,
sources/sam/latex/figs/*.pdf|pgf. Line numbers are segany.tex unless stated.

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

### ch-sam2

Sources: sources/sam2/latex/sam2.1_arxiv.tex (SAM 2 v2), tab/*.tex; stm/latex/final.tex; xmem/latex/*.tex;
cutie/latex/*.tex; hiera/latex/hiera_arxiv.tex; abs.html dates.

All 68 claim spans in ch-sam2 (64 unique ids) against their locs:
- video-memory.json: stm-every-n (final.tex:560-561), xmem-stores (introduction.tex:37-38), xmem-34k
  (methods.tex:185). The prose around them holds: STM matches every query pixel with every memory pixel
  (final.tex:158, 397); XMem has sensory (every frame), working and long-term prototype memory (methods.tex:11, 24,
  133); prior methods "struggle to handle videos longer than a minute" (introduction.tex:10); Cutie reads object-level
  memory (00-abstract.tex:2); all three start from a first-frame annotation. Dates: STM 2019-04-01, XMem 2022-07-14,
  Cutie 2023-10-19 (abs.html).
- sam2-*: architecture (L284-326, L950-976), Table 5 (L566-586), Sec. 7 FPS (L612), Table 1 (L355-362), controlled
  experiment (L1234), SA-V macros and text (L139-168, L399-450, L1110), Table 4 (L520-528), Table 6
  (tab/tab-mask2masklet-results-oss.tex), Table 9c (L773-774), Table 11 (L907-911), Table 12 and App. D.2
  (L984-1082), model card (L1782); sam2-hiera-bplus-channels (hiera_arxiv.tex:394).
- No extra-f* ids are used in this section. No uncertain or code_only claims are used in this section.
- Unsourced prose: Hiera is MAE-pretrained (L291), streaming encoder run once per frame (L291), prompt encoder
  identical to SAM (L296), image segmentation as a one-frame video (task section). No errors beyond (b).

### ch-sam3

Sources: sources/sam3/latex/main.tex (v2), sources/sam3/latex/multiplex.tex (App. H), sources/repos/sam3.README.md, /tmp pdftotext of sources/sam3/paper.pdf for table numbers.
Appendix letters confirmed: A Ablations, B Limitations, C Model Details (C.1 arch, C.2 image impl, C.3 video impl, C.4 training), D Data engine, E Dataset and metrics (E.3 Metrics), F Additional experiments, G Agent, H Object Multiplex (multiplex.tex, \input at main.tex:3420), I Cards.
No extra-f* claim ids are used in this chapter.

All 92 claim ids in ch-sam3 against main.tex / multiplex.tex / README / abs.html: v1 date; 30 secs (379); Q=200 (1428); 6 fusion, 6 decoder blocks (1424-1426); 4 of 32 global layers, 1008, 3x3 windows of 336, context 32, 5.4B pairs (1414); 850M/450M/300M (1392, 1574); patch 14 (Table 11, 1123); 1296 tokens (1133); presence BCE 20, LR x0.025, 5k iters, batch 512 (1590); stage 2 LRs, LLD 0.9, 95k, 896, loss weights 5/2/100/200/10 (1582); stage 4 512, 190k, 5.0e-4, 20:1:1:1, 60k (1592); K=2, 15% (1452, 1511); T=15, iou 0.1, N=16, 0.8 (1544, 1548, 1559); Table 9a/9b (909-931); Table 10 (1101-1102); Table 7 (821-823); Table 3 (676-680, 707); Table 2 (651); Table 1 (609-618, 633); 48.8 / 38.5 intro (364); 70.8 internal (1138); Table 29 oracle (2380); Table 23 (1942-1943); Table 15 (1302); Table 19 (1677-1678); 6 retrains (496); 22.4M ontology (511); Table 24 HQ 4.0M / 5.2M / 52.3M / 88.5%, SYN 1.4B (2126-2128); M=16, 5.2x, 7%, H100 (multiplex.tex:18, 81); 172k A100 / 86k H200 (3603); confidence 0.5 (549).
Prose around each number, all unsourced statements, the limitation quote (1377, verbatim), the App. C.2 and C.4 quotes (1430, 1592, verbatim), the three display equations, the inline scene code comments and readouts, and both ArchFlow specs (shapes, sub, params, readouts).

### ch-sam3d, ch-efficient

ch-sam3d: all 32 sam3d.json spans (dates, 4 token sets, 1.2B, 600M, 4096 + 1 tokens, 64^3, 6D, lambda 0.1, 10/80/150 s, N=8, 80%, ~1M images, 3.14M meshes, 7M prefs, 2.7M, 61M, 1,000,
Table 4 six rows, Hi3DGen 0.1629 = best baseline F1 in Table 2, 5:1 and 6:1 from Abstract and Sec. 4.1); "the paper prints no per-benchmark percentages" (Fig. 8 is bars without printed values: true);
LLM-style recipe (sec/01_introduction.tex:15, 29); limits (App. F) and mask robustness (App. E.9, "fair robustness"); "SAM series as a fast way" (App. E.9); scene text strings; equation.
ch-efficient: all 36 spans against MobileSAM v2, EfficientSAM v1 (Table 5 rows, 75% mask ratio, ViTDet boxes), EfficientViT-SAM v2 (Table 1, Table 4, abstract 48.9x = 538/11),
EdgeTAM Table 2, EfficientSAM3 v1, MobileSAMv2 Table 2, HQ-SAM v2 (Table 1, Table 2 DIS mBIoU with GT-box prompts, Sec. 3.3); dates in parentheses (all match v1 months); scene arrays.

### ch-family, ch-map, ch-inspection, ch-timeline, ch-summary, ch-deeper

Scope: explainer.src.html lines 2989-end, parts/scene-tree.js (inline), scene-map.js (inline), scene-inspect.js.
Method: every claim id used in these sections was checked at its loc in sources/; the scan numbers were recomputed
from notes/arxiv-scan-2026-10-09.json with the patterns in notes/scan/domains.json; every tree date was compared
with the [v1] line of sources/*/abs.html.

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
