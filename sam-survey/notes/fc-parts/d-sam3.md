# Fact check: ch-sam3 (explainer.src.html lines 2449-2801, inline scene-sam3 and viz-sam3flow / viz-sam3trk)

Sources: sources/sam3/latex/main.tex (v2), sources/sam3/latex/multiplex.tex (App. H), sources/repos/sam3.README.md, /tmp pdftotext of sources/sam3/paper.pdf for table numbers.
Appendix letters confirmed: A Ablations, B Limitations, C Model Details (C.1 arch, C.2 image impl, C.3 video impl, C.4 training), D Data engine, E Dataset and metrics (E.3 Metrics), F Additional experiments, G Agent, H Object Multiplex (multiplex.tex, \input at main.tex:3420), I Cards.
No extra-f* claim ids are used in this chapter.

## (a) WRONG

None found. Every checked value matches its loc, and no sentence states something the source contradicts outright.

## (b) MISLEADING / over-claimed

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

## (c) Honesty labels

- viz-sam3: data-toy, figcaption begins "Toy model", scene uses badge('toy', ...). Correct.
- viz-sam3flow and viz-sam3trk: badge "Computed", data-computed. Values shown are paper config values plus the 72 x 72 grid, which is labelled "our arithmetic". Consistent with the page-authoring rule for ArchFlow readouts. No problem.
- 72 x 72 detector grid: supported by main.tex:1138 (the ablation stride-28 variant uses 4x fewer detector tokens than SAM 3, so the default is stride 14, 1008/14 = 72).

## (d) Equations

- eq-sam3-presence: matches main.tex:1434-1435 term by term. Label "App. C.2, Presence Head (unnumbered display)" is right.
- Video propagate/detect/match_and_update: matches main.tex:434-441. "Sec. 3 (unnumbered)" is right (Sec. 3 Model, paragraph Tracker and Video Architecture).
- cgF1 = 100 * pmF1 * IL_MCC: matches main.tex:2352-2354. "App. E.3, Eq. 3" is right (Eq. 1 pmF1 main.tex:2340, Eq. 2 IL_MCC main.tex:2346). Gloss descriptions of Eq. 1 and Eq. 2 are faithful.

## (f) Checked

All 92 claim ids in ch-sam3 against main.tex / multiplex.tex / README / abs.html: v1 date; 30 secs (379); Q=200 (1428); 6 fusion, 6 decoder blocks (1424-1426); 4 of 32 global layers, 1008, 3x3 windows of 336, context 32, 5.4B pairs (1414); 850M/450M/300M (1392, 1574); patch 14 (Table 11, 1123); 1296 tokens (1133); presence BCE 20, LR x0.025, 5k iters, batch 512 (1590); stage 2 LRs, LLD 0.9, 95k, 896, loss weights 5/2/100/200/10 (1582); stage 4 512, 190k, 5.0e-4, 20:1:1:1, 60k (1592); K=2, 15% (1452, 1511); T=15, iou 0.1, N=16, 0.8 (1544, 1548, 1559); Table 9a/9b (909-931); Table 10 (1101-1102); Table 7 (821-823); Table 3 (676-680, 707); Table 2 (651); Table 1 (609-618, 633); 48.8 / 38.5 intro (364); 70.8 internal (1138); Table 29 oracle (2380); Table 23 (1942-1943); Table 15 (1302); Table 19 (1677-1678); 6 retrains (496); 22.4M ontology (511); Table 24 HQ 4.0M / 5.2M / 52.3M / 88.5%, SYN 1.4B (2126-2128); M=16, 5.2x, 7%, H100 (multiplex.tex:18, 81); 172k A100 / 86k H200 (3603); confidence 0.5 (549).
Prose around each number, all unsourced statements, the limitation quote (1377, verbatim), the App. C.2 and C.4 quotes (1430, 1592, verbatim), the three display equations, the inline scene code comments and readouts, and both ArchFlow specs (shapes, sub, params, readouts).
