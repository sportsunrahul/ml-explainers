# Page fact-check 2: ch-sphere, ch-negatives, ch-beyond, ch-supcon, ch-clip, ch-guide, ch-closing (+ kits 60, 65, 75, 78, 81, 85, 90)

Date: 2026-10-13. Checker: fresh agent, adversarial pass on prose, eqno lines, kit readouts, badges and the closing reading list.

Note on sources: `sources/` was not on disk. I re-fetched every paper in notes/papers.tsv from arXiv (abs, e-print LaTeX, PDF) into /tmp/cs-src/<name>/ and read them as data (pdftotext, grep). Hadsell 2006 (no arXiv) was checked against notes/papers/hadsell2006.md only. I also fetched SupCon v1-v4 to test the "earlier versions" claim.

## Counts

- Files read: 7 chapter files, 7 kit files, 5 sim notes (sphere, negatives, beyond, supcon, clipsig).
- data-claim spans in scope: about 190 unique ids. Script check: every id exists in notes/claims.json and every span contains its claim value (0 mismatches). I read each claim description against its sentence (model, protocol, epochs, dataset): 0 mismatches found.
- eqno lines checked against source: 17. Wrong: 1.
- "Where to go deeper" references checked: 13 items. Wrong section/equation pointers: 3.
- Kit readouts and canvas labels: checked against notes/sim-*.txt and the kit code. No readout prints a paper number. Readouts match the sims. Prose that describes a readout: 1 overstated.
- Badges and captions: 7 figures. All "Computed" figures run the stated loss or algorithm (InfoNCE/L_align with Adam; Wang-Liu Eq. 5 shares, Robinson weights, Chuang Eq. 7-8; SwAV App. A.1 Sinkhorn; Barlow Eq. 1-2 with exact gradient; Jing Eq. 2; SupCon Eq. 1/Eq. 2; real softmax/sigmoid). Guide badge "Our reading" is honest. 0 problems.
- Equations displayed vs source: all match or are labelled as our notation (SigLIP sign of b handled and explained; verified: Sec. 3.2 prints +b, Algorithm 1 adds b).
- Statements verified true that a reader might doubt: SupCon v1-v4 print L_out <= L_in and v5 corrects it; Wang-Liu main vs supp ImageNet100 tau=1.0 disagree (63.57 vs 65.49); DINOv3 88.4 is at 448 px vs DINOv2 86.5 at 224; Web-DINO crossover at 5B; DetCon 5x; Barlow prints "SwAV w/o multi-crop 71.8" = SimSiam repro; Wang-Isola Theorem 1 rate O(M^-1/2) (changelog 11/6/2020); SwAV "fast and sufficient", 35 ms, Table 14 "fail"; Chuang App. B fallback to the biased loss.
- Problems: 15 (WRONG 5, MISLEADING 4, OVERSTATED 4, UNSOURCED 2).

## Problems

1. parts/ch-beyond.html, eqno under eq-jingx: "Lemma 2, Sec. 4.2 in Jing et al., 2110.09348v3 (unnumbered)." -- WRONG. Jing et al. number it: "X = Sigma_0 - Sigma_1 (7)" (paper.pdf p.4, Lemma 2). Fix: "Eq. 7 (Lemma 2), Sec. 4.2 in Jing et al., 2110.09348v3."

2. parts/ch-closing.html, Negatives item: "Chuang et al. (2007.00224v3), Sec. 3 (Eq. 7)" -- WRONG. Sec. 3 is "Setup and Sampling Bias"; Eq. 7 and Eq. 8 are in Sec. 4 "Debiased Contrastive Loss" (paper.pdf p.4-5). Fix: "Sec. 4 (Eq. 7 and 8)".

3. parts/ch-closing.html, memory-bank item: "MoCo (1911.05722v3), Sec. 3.2 (Eq. 1 and 2) and Fig. 3" -- WRONG (partly). Eq. 1 (the InfoNCE loss) is in Sec. 3.1 "Contrastive Learning as Dictionary Look-up"; Eq. 2 (momentum update) is in Sec. 3.2. Fix: "Sec. 3.1 to 3.2 (Eq. 1 and 2) and Fig. 3".

4. parts/ch-closing.html, Without negatives item: "SwAV (2006.09882v5), Sec. 3.1 (Eq. 1 to 5)" -- WRONG (minor). Eq. 1 and Eq. 2 (swapped-prediction loss) are in the Sec. 3 intro, before Sec. 3.1 "Online clustering", which holds Eq. 3 to 5. Fix: "Sec. 3 and 3.1 (Eq. 1 to 5)".

5. parts/ch-negatives.html: "Press it. The near negatives get almost all of it. The bars on the far side of the circle are so short you can hardly see them." -- OVERSTATED. The figure's own readout (notes/sim-negatives.txt, plain) gives the 18 far negatives 30% of the push (false 50%, 2 nearest true 19%). Each far bar is short, but together they take almost a third. Fix: "The six near negatives get about 70% of it; each far negative gets only a sliver, though the 18 of them still add up to about 30%."

6. parts/ch-guide.html, "search" recipe step 2: "If you have labels, fine-tune with a supervised contrastive loss ... On ImageNet, SupCon reached 78.7 top-1 with a ResNet-50, against 78.2 for the published cross-entropy baseline." -- MISLEADING. The recipe is for "only a few labelled examples per class" and for fine-tuning a pretrained encoder; the cited number is training from scratch on all 1.28M ImageNet labels, scored by top-1, not retrieval. SupCon's own transfer table shows no gain over cross-entropy features (Table 4 mean: SupCon-50 84.27 vs Xent-50 84.67; SupCon-200 85.67 vs Xent-200 85.77). Fix: state the evidence scope, e.g. "SupCon's evidence is full-label ImageNet training (78.7 vs 78.2 top-1) and better corruption robustness; it was not tested in the few-label fine-tuning setting, and its transfer features were not better than cross-entropy's (SupCon Table 4). Treat this step as our reading."

7. parts/ch-supcon.html: "A real encoder must map thousands of different images through the same weights, so it cannot squeeze a class to one point. You get tight clusters instead" -- MISLEADING / UNSOURCED. Nothing in SupCon supports "cannot"; the L_out optimum with labels is all same-class views on one point (class collapse), and an overparameterised network can approach it on training data. Fix (hedged, no in-project source states the optimum): "In this loss, nothing stops a class from shrinking to one point, and the toy shows it. Real encoders trained on finite data usually stop short of that, so you see tight clusters (our reading)."

8. parts/ch-beyond.html: "Barlow Twins prints 71.8 for "SwAV without multi-crop"; SwAV never prints that number" -- WRONG (literal, minor). SwAV Fig. 3 (left table) does print 71.8, for SeLa-v2 with multi-crop (2x160+4x96). The point (SwAV never reports 71.8 for itself without multi-crop) holds. Fix: "SwAV never reports 71.8 for itself".

9. parts/ch-negatives.html, In practice: "Turn hardness up gently: on CIFAR100 the best was beta = 0.5, and much larger values mostly find false negatives." -- OVERSTATED. Robinson Sec. 6.1 offers this as a suggested explanation ("This observation suggests an explanation ... many of whom will have the same label as x"), not a measurement of a majority. Fix: "... and the paper suggests that much larger values pick many false negatives (Sec. 6.1)."

10. parts/ch-beyond.html: "SimCLR needs a batch of 4096 images (chapter 5)." -- OVERSTATED. 4096 is SimCLR's default and best-result batch (claim simclr-batch, Sec. 2.3); SimCLR trains at 256 with lower accuracy (e.g. 61.8% at batch 256, 200 epochs, DCL Sec. 4.2, cited in ch-negatives). Fix: "SimCLR's best results use a batch of 4096 images."

11. parts/ch-sphere.html: "HaoChen et al. ... prove that a linear probe on its features works well whenever augmentations rarely connect images of different classes." -- MISLEADING (minor). The guarantee (augmentation-graph analysis, abstract and Sec. 3) also needs each class to be well connected inside by augmentations ("ground-truth classes naturally form connected sub-graphs"); few cross-class edges alone is not enough. Fix: "... works well when augmentations link images within a class well and rarely link images of different classes."

12. parts/ch-sphere.html, claim card "How": "They trained many encoders per dataset (304 on STL-10, 45 on ImageNet-100) ... and report the best of each family with a linear probe." -- MISLEADING (minor). Table 1 picks its rows by 5-fold cross-validation "among all 150 encoders trained from scratch with 128-dimensional output and 768 batch size", not among all 304. Fix: "... and report the best of each family (picked by 5-fold cross-validation; for STL-10 among the 150 encoders with 128-d output and batch 768)."

13. parts/ch-sphere.html, In practice: "A L_uniform that drifts toward 0 warns of collapse long before a downstream test does." -- UNSOURCED. No paper in scope measures how early L_uniform signals collapse; "long before" is untested. Fix: "A L_uniform that drifts toward 0 is a cheap early sign of collapse (our reading)."

14. parts/ch-supcon.html, In practice: "And when a class has very few examples, it adds little over the plain two-views game, so pair it with self-supervised pretraining." -- UNSOURCED. SupCon never tests rare classes or the pairing; the in-batch fallback is derived, the recommendation is not. Also "Start from a pretrained encoder and fine-tune with the supervised contrastive loss" is untested in the paper (all SupCon runs train from scratch). Fix: append "(our reading)" to both recommendations.

15. parts/ch-supcon.html, check-yourself: "which is one reason it needs large batches: the paper reports batch 6144 for ResNet-50", and In practice: "It needs batches that hold several samples per class" -- OVERSTATED. SupCon Sec. 4.1: with a memory of 8192 stored vectors and batch 256, ResNet-50 reaches 79.1%, "slightly better than the 78.7% accuracy with 6144 batch size (and no memory)". Fix: "it needs many same-class views per anchor: a large batch, or a memory bank of stored embeddings (79.1% at batch 256 with a memory of 8192, Sec. 4.1)."

## Notes (checked, not counted as problems)

- parts/ch-clip.html: "Past about 32 k, a bigger batch stopped helping in the paper's runs." In App. B Table 5 (9B examples) the softmax still gains 0.3 at 98 k (72.9 to 73.2), but the paper's Fig. 2 caption says both losses "saturate" around 32 k. Acceptable as written; optional precision: "the sigmoid peaked at 32 k; the softmax gained 0.3 more at 98 k".
- Dates and venues confirmed: CLIP submitted 26 Feb 2021, ALIGN 11 Feb 2021 (same month), SupCon 23 Apr 2020 and NeurIPS 2020 (PDF footer), STL-10 encoders AlexNet-based (Wang-Isola Sec. 5 list).

## Checked and found sound (no action)

- Wang-Isola: L_align/L_uniform displays (Sec. 4.1.1/4.1.2, unnumbered), Eq. 2 Theorem 1 (Sec. 4.2), proof in App. A.2, code alpha = 2, t = 2 with pdist (Fig. 6), Fig. 8 finetuning, Tables 1/3/4/5 rows, 240-epoch unofficial MoCo, official MoCo v2 200 epochs, ICML 2020.
- Wang-Liu: Eq. 4 gradient and Eq. 5 share (match LaTeX), Sec. 4.2 dilemma quote verbatim, Sec. 5.3 "0.2 or 0.3", memory bank, ResNet-18/50, 200 epochs, batch 128, Table 1 values, supp. sweep 0.05 to 1.0.
- Chuang Eq. 7 (max with e^{-1/t}) and Eq. 8, CMC at tau = 0.07, App. B fallback, Table 1, "4.26%" quote.
- Robinson Sec. 3.1 reweighting display, Fig. 4 right values, Sec. 6.1, Table 1 tinyImageNet, App. C.1 MoCo-v2 88.08 to 88.47.
- DCL 61.8 to 65.9 (Sec. 4.2 text), ECCV 2022.
- SwAV Eq. 1 to 5, App. A.1 pseudocode matches kit 75, high-eps quote, "not a target" quote, Table 14, Fig. 2/3 numbers.
- Barlow Eq. 1/2, lambda search, Fig. 5 range and "not very sensitive", Table 5 ablation at 300 epochs.
- Jing Eq. 1, Eq. 2 (kit 81 matches, negatives from one branch), Lemma 1/Eq. 5, Theorem 1 quote, Fig. 3 16x16 and k range, Fig. 7b, Table 1, Sec. 5 implicit regularization.
- SupCon Eq. 2/3, Jensen direction, v1-v4 erratum claim, "remove bias" quote, Table 1/3/5, Fig. 3 mCE, Sec. 4.3 variance, Sec. 8 margin 2 tau, N-pairs 57.4, tau = 0.1 for all results, posted 23 Apr 2020.
- CLIP Fig. 3 pseudocode, 0.07 init, clip at 100, N-pair/InfoNCE lineage (Sec. 2.3); SigLIP softmax display (Sec. 3.1), sigmoid display (Sec. 3.2, +b), Algorithm 1, t' = log 10, b = -10, chunked b^2 memory, no all-gathers, App. B Table 5, SigLiT g/14 84.5 on 4 chips in 2 days.
- Guide dense numbers (DINOv2 Table 10; DINOv3 Table 3), CLIP chapter "where it stands" bullets, cross-links (vlm-survey #ch-sigmoid exists and has the bias-start buttons and ring schedule; jepa-survey #ch-collapse exists).
- All worked arithmetic in the seven chapters recomputed (sphere -1.10 and -6; shares 0.62/0.28/0.10 and 0.982/0.018/0.0001; Chuang g = 2.78; SwAV 0.527 and the 2x2 Sinkhorn; Barlow 0.05305 and 67,100,672; SupCon 1.11, 0.82 and gaps 0.35/0.02; CLIP 0.95/0.05/0.001, sigmoid rows, puppy 0.70 and 0.88): all correct.
