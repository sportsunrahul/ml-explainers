# Fact-check: claims A.json + E.json, and 22 paper notes

Scope: 237 claims (A.json 96, E.json 141), checked against sources/<name>/ (tex, PDF, abs.html).

| Verdict | Count |
|---|---|
| OK | 234 |
| WRONG | 0 |
| WRONG-LOC | 2 |
| MISLEADING | 1 |
| UNVERIFIABLE | 0 |

Uncertain claims (2, both in A.json): both resolve as OK.
- cpc-img-negatives: neither CPC Sec 3.2 nor the appendix states an image negative count or source. "not stated" is correct.
- hadsell2006-margin-value: the rendered Fig. 1 (PDF p.3) shows the 1.25 tick under "margin: m". No experiment states m. The value is illustrative only, as the claim says.

Code-only claim wu2018-bank-momentum: OK, but the source is weak. src/readme.md is a lab log, and its momentum-0.5 runs are 120-epoch runs. CMC Sec 4.1 Implementation ("use a momentum 0.5") is a better citation.

## Non-OK claims

- cpc-mi-bound: WRONG-LOC. The value is right, but the unnumbered bound display after Eq. (5) is on PDF p.4, not p.3. Eq. 4 is on p.3. Fix: loc "Sec 2.3, unnumbered display after Eq. (5), PDF p.4; App. A.1 Eqs. (6)-(11), PDF p.13".
- chopra2005-venue: WRONG-LOC. The Chopra PDF title page prints no venue. CVPR 2005 is confirmed by DrLIM reference [5] ("CVPR-05, 1:539-546, 2005"). Fix: loc "not printed in PDF; hadsell2006 PDF, reference [5]".
- wu2018-bank-storage: MISLEADING. Sec 4.2 item 4 says "The encoded features of all 1.28M images ... only take about 600 MB of storage". That sentence is about kNN search cost, not the memory bank. The bank holds the same n x 128 vectors (Sec 3.1), so the size transfers. The paper does not call it bank storage. Fix: claim text "storage of all 1.28M 128-d encoded features (same size as the memory bank)".

## Notes problems

- cpc.md Sec 6: gives the MI bound display as PDF p.3. It is on p.4.
- cpc.md Sec 9: "Bound saturates at log N (stated in A.1)". A.1 says only that larger N helps. Saturation follows from L_N >= 0, but it is our reading. Label it so.
- wu2018.md Sec 7: "memory bank ... about 600 MB" has the same problem as wu2018-bank-storage.
- chopra2005.md Sec 1: "Venue: CVPR 2005" has no loc. Cite DrLIM ref [5].
- hadsell2006.md Sec 7/10: Fig. 1 does not draw the dissimilar branch to Eq. 4 scale. At m = 1.25 its intercept is about 1.56 = m^2, not m^2/2 = 0.78 (our reading of the rendered plot). A toy must use Eq. 4 with the 1/2 factor and must not copy the plot.
- tschannen.md Sec 4 item 5: "CPC v2 draws negatives from the same image, which violates independence" reads as Tschannen's claim. Tschannen does not say it. CPC v2 Sec 2.1 does say negatives come "from other locations in the image". Mark it "(our reading)".
- arora.md Identity: "source is the ICML 2019 template" is wrong. main.tex loads `\usepackage[accepted]{arxiv}`, and the icml2019 line is commented out. No venue is printed in the PDF or abs.html. Mark ICML 2019 as external knowledge.
- supcon.md Sec 6: "Supp. Sec. 7, Eqs. 20-24". Supp. Sec. 7 is Eqs. (16)-(24); the magnitude analysis is Eqs. (19)-(24).
- supcon.md Sec 10 (toy anchor, Supp. Table 5): the cap ablation also removes the dropped positives from the denominator (positives.tex). So 69.3 is neither SupCon with fewer positives nor SimCLR. State this if the toy uses 69.3 -> 78.5.
- supcon.md (information only): the SupCon main text points to "Supplementary (Sec. 6)" for the positives ablation, but in v5 it is Sec. 11. The note cites Sec. 11 correctly. Do not copy the paper's cross-reference.
- clip.md Sec 9: attributes "needs big batches and compute" to Sec. 6 Limitations. Sec. 6 discusses compute only. Drop "big batches" or mark it as our reading. Keep the SigLIP cross-link line out of any "stated limitations" list.
- metaclip.md: "114k of 500k entries have no match (Sec. 3.2)". The 114k figure is in Sec. 3.3 (Inverted Indexing).
- metaclip.md: takes 72.4 (ViT-B/16, 1B data) from SigLIP 2 Table 1. MetaCLIP's own Table 5 (ViT-B/16, row MetaCLIP(1B)) prints 72.4. Cite it directly.
- pe.md: "avg robustness 86.6" mislabels Table 5 column "Avg Class.". That column is the mean over six ImageNet-family zero-shot sets (val, v2, ObjectNet, A, R, Sketch), and it includes ImageNet val. Call it "avg zero-shot classification".
- siglip2.md: the model sizes (B 86M, L 303M, So400m 400M, g 1B) come from the abstract, not Sec. 2.1. Add loc "abstract".

## Confirmed (no action)

- All quoted equations in the 22 notes match the sources. This includes CPC Eqs. 1-5 and A.1, Wu Eqs. 1-10, DrLIM Eqs. 1-9, Chopra Eqs. 8-9, Gutmann Eqs. 3-5, Sohn Eqs. 3/6/7, Tschannen Eqs. 2-5, Poole Eq. 10, CMC Eq. 2, Arora Eq. 5 and Thm 4.1, and FaceNet Eqs. 1/3/4. It also includes SupCon Eqs. 1-6, the CLIP Fig. 3 pseudocode, ALIGN, DenseCL, DetCon and the OpenCLIP power law.
- siglip-bias-sign holds. The printed Sec. 3.2 equation reduces to sigmoid(z(t x.y - b)), and Algorithm 1 line 9 uses t x.y + b.
- SigLiT B/8 prints 79.7 in the caption, Sec. 1 and Sec. 4.4, and 79.8 in the Table 1 row.
- densecl.md correctly flags the missing /tau on the positive term in the DenseCL denominator.
- All arXiv v1 and version-read dates match abs.html.
