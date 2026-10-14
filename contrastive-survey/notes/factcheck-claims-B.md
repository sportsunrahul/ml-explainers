# Fact-check: notes/claims/B.json

Checker: fresh pass against sources/<name>/ (src/*.tex, figure PDFs, paper.pdf, abs.html).
Scope: 340 claims (SimCLR 189, MoCo 54, MoCo v3 47, MoCo v2 37, SimCLRv2 9, InfoMin 4), plus the
equations and paper-claim statements in notes/papers/{moco,mocov2,mocov3,simclr,simclrv2,infomin}.md.

## Summary

| verdict | count |
|---|---|
| OK | 322 |
| WRONG | 0 |
| WRONG-LOC | 0 |
| MISLEADING | 18 |
| UNVERIFIABLE | 0 |

- Every numeric value matches its source. No value is wrong.
- All 18 MISLEADING verdicts are claim-text problems: the value is right, but the text describes it wrongly.
- Versions: the arXiv stamp on page 1 of each paper.pdf matches the version in the claims (1911.05722v3,
  2003.04297v1, 2104.02057v4, 2002.05709v3, 2006.10029v2, 2005.10243v3). The dates in abs.html match all
  9 date claims. All 340 claim ids are unique.
- Figure data labels:
  - MoCo Fig. 3 (src/fig/results_mechanisms.pdf): I checked all 15 values with pdftotext and a 200-dpi render.
    The curve for each value is correct (at K=256: end-to-end 54.9, MoCo 54.7, memory bank 50.0).
  - SimCLR Fig. 5 (src/figures/heatmap_da_pairwise.pdf): a script checked all 56 values against the pdftotext
    grid, with 0 mismatches. A render confirms the row/column assignment (rows = 1st, columns = 2nd).
  - SimCLR Table B.1: a script checked all 48 values against main.tex, with 0 mismatches.
- "Our reading" claims are sound:
  - simclr-views: 2N = 8192 at N = 4096.
  - simclr-neg-4096: 2(N-1) = 8190.
  - mocov3-vitb-linear: Table 1 ViT-B 76.7 equals the 600-ep cell of the Sec. 6.2 training-length table.
  - mocov3-vitl/vith-linear: Sec. 6.2 says "we train the bigger ViT-L/H for 300 epochs".

## Non-OK claims

- simclr-fig5-crop-crop, simclr-fig5-cutout-cutout, simclr-fig5-color-color, simclr-fig5-sobel-sobel,
  simclr-fig5-noise-noise, simclr-fig5-blur-blur, simclr-fig5-rotate-rotate | MISLEADING |
  The values are right (33.1, 25.6, 18.8, 4.0, 9.8, 2.6, 2.6). The Fig. 5 caption (main.tex l.297) says
  "diagonal entries correspond to single transformation". The text "1st=X, 2nd=X" implies X is applied twice. |
  Fix: "single transformation X (diagonal)".
- simclr-fig5-crop-average, simclr-fig5-cutout-average, simclr-fig5-color-average, simclr-fig5-sobel-average,
  simclr-fig5-noise-average, simclr-fig5-blur-average, simclr-fig5-rotate-average | MISLEADING |
  The values are right (39.2, 29.4, 25.7, 18.8, 15.5, 14.5, 13.8). The caption says "The last column reflects
  the average over the row". "Average" is not a 2nd transformation. |
  Fix: "row average over 2nd transformations, 1st=X".
- simclr-t5-tau-0p05, simclr-t5-tau-0p1, simclr-t5-tau-0p5, simclr-t5-tau-1 | MISLEADING |
  The values are right (59.7, 64.4, 60.7, 58.0). The claim text says "batch 4096". The Table 5 caption
  (main.tex l.455) says only "The contrastive distribution is over 4096 examples". At batch 4096 with both
  views, the distribution would be over 8191 examples. So either the batch was 2048, or the negatives came
  from one view only (the Sec. 5.1 footnote says this for the Table 4 losses). The source does not state the
  batch. | Fix: drop "batch 4096" and quote the caption wording. The simclr.md Config tau row has the same
  problem (see notes-level item 7).

## Caveats on OK claims (the value and loc are right, but the context needs care)

- simclr-linear-r50, simclr-linear-r50-2x, simclr-linear-r50-4x: "batch 4096" is not in Sec. 6, which states
  only "1000 epochs". The batch comes from App. C Table C.1, which gives the setup for the best (R50-4x)
  result, and from the Sec. 2.3 default. Cite App. C if the page prints the batch.
- mocov3-vitb-ft, mocov3-vitl-ft: 83.2 is a 150-epoch fine-tune and 84.1 is a 100-epoch fine-tune with drop
  path 0.5. Both use the DeiT codebase (Sec. 6.3). Give the schedule if the page compares them with other
  fine-tune numbers.
- infomin-linear-800, infomin-linear-200: the Table 1 caption defines "InfoMin Aug." as RandomResizedCrop,
  color jitter, blur, RandAugment, color dropping, "and a JigSaw branch as in PIRL". Do not present 73.0 or
  70.1 as a pure-augmentation result next to SimCLR or MoCo.
- simclr-t4-*: the Table 4 losses use negatives from one augmentation view only (Sec. 5.1 footnote). The
  values are right.
- moco-gpus: the GPU type is not stated for the MoCo v1 IN-1M run. The loc already says so.

## Notes-level problems

All quoted equations match the LaTeX character for character:
- MoCo Eq. 1 (eq:infonce) and Eq. 2 (eq:moco).
- MoCo v2 Eq. 1.
- MoCo v3 Eq. 1 and Alg. 1 ctr() with the 2 tau scale.
- SimCLR Eq. 1 (eq:loss), the \one macro on l.48, and the Alg. 1 total loss.
- SimCLRv2 Eq. 2 (eq:distill).
- InfoMin Prop. 3.1 (theory:infomin).

The statements about what each paper claims are faithful. Smaller items:

1. infomin.md, Key number: the note does not say that "InfoMin Aug." includes a PIRL-style JigSaw branch
   (Table 1 caption). Add this. The parenthetical "MoCo framework (65536 negatives, batch 256, 8 GPUs)" comes
   from App. "Details" (sota.tex l.23). That paragraph covers the 100-epoch crop and color plots, not the
   Table 1 models. Also, the note leaves out the 100 epochs. Move the parenthetical to the crop sentence, or
   label it "ablation plots only, 100 ep".
2. moco.md, Config "GPU type": "MoCo v2 Table 3 gives 8 V100 16G, 53 hrs for this config" is an inference.
   MoCo v2 Table 3 gives the cost of the MoCo mechanism at batch 256 for 200 epochs. It does not say this is
   the v1 recipe. The 53 h agrees with MoCo v1 Sec. 4 ("~53 hours"). Label the GPU type "inferred from
   MoCo v2".
3. mocov3.md, Sec. 8, Fine-tune: add "ViT-B 150-ep, ViT-L 100-ep with drop path 0.5, DeiT codebase".
4. simclr.md, Sec. 4: "larger batch and longer training help contrastive learning more than supervised" follows
   the Sec. 5.2 heading ("benefits (more)"). The paper has no supervised batch-size sweep. The supervised
   comparison is only for longer training (App. B.3, Table B.3). Soften the wording, or cite B.3 for the
   "longer training" half.
5. simclr.md, Table 6 line, "(linear, 1000 ep, batch 4096)": this has the same batch-source caveat as the
   simclr-linear-* claims above.
6. Fig. 5 in simclr.md is described correctly: asymmetric one-branch protocol, rows = 1st transformation, and
   the paper says it "hurts the performance". Only the claim texts in B.json have the diagonal and average
   problem.
7. simclr.md, Config table, tau row: "Table 5 sweep (100 ep, batch 4096)" has no source for "batch 4096".
   Drop it. The Table 5 caption says only "The contrastive distribution is over 4096 examples".
   (The Results line in simclr.md quotes the caption correctly.)
