# Fork D: ch-prune, ch-lottery, kits 60/61/62, atlas sparsity

All data-claim spans in ch-prune.html and ch-lottery.html (and the spans in 62-imp.js) match their claim values;
spot checks against han2015, han2016dc, frankle2019, frankle2020lmc, li2017filters, liu2017slim, liu2019rethink,
blalock2020 and mishra2021nm sources agree. Badges are honest: viz-sparse and viz-dc are Computed (toy weights,
exact algorithms; sourced numbers only in captions/tables), viz-imp is Toy model (the mask is computed, the
"training" is a seeded drift; only text-stated LeNet findings are printed). Kit readouts: 60-sparse prints only
computed counts plus the spanned 2x; the 2:4 storage figure (18 bits per kept FP16 value, 43.8% less) matches
Mishra's 36-vs-64-bit example. 61-dc prints only computed values. 62-imp round-to-P_m mapping (0.8^r) matches the
paper's stated P_m points (51.3, 21, 13.5, 3.6, 2.9).

## Must-fix

1. parts/ch-prune.html, eq. note: "Deep Compression (arXiv 1510.00149v5, Sec. 2) uses the same rule."
   Problem: DC Sec. 2 only says "all connections with weights below a threshold are removed" and builds on Han 2015;
   it never states the q x std rule (grep: no "standard deviation" in han2016dc). FIXED: now says DC builds on the
   method, quotes its sentence, and notes it does not restate the rule.
2. parts/ch-lottery.html, claim card "At scale": "73.7% ... no better than random reinitialization at 73.4%".
   Problem: 73.7 is 0.3 points above 73.4 (LMC Table 2), so "no better" overstates. FIXED: "only 0.3 points above".
3. parts/ch-lottery.html: "Matrix sizes must be multiples of 16 for FP16 and 32 for INT8".
   Problem: Mishra (software.tex l.71, results.tex l.48) says layers whose inner (GEMM-K) dimension is not a multiple
   are left unpruned; it is the summed dimension, not "matrix sizes", and it is a pruning rule, not a hard requirement.
   FIXED: "The paper prunes only layers whose summed dimension is a multiple of 16 ... 32 ...".

## Nits

- atlas/parts/sparsity.dialogs.html: "measured no whole network" was ambiguous (the paper measures whole-network
  accuracy, not speed). FIXED: "timed no whole network".
- ch-lottery "Their conclusion is that operation and parameter counts are a loose proxy for latency": Blalock et al.
  state this in Sec. 2 while citing others; their main conclusion is that results are hard to compare. Acceptable.
- ch-lottery Li et al. timing: "inference over the test set" is right for CIFAR-10 (10,000 images, Li appendix).

## Open doubts

- ch-lottery "Blalock et al. measure no hardware": ShrinkBench reports parameter and FLOP counts only, as far as the
  source shows; not verified exhaustively.
- ch-prune "Most weights in a trained network are close to zero" is a general statement with no source span.
- The new "0.3 points" in the LMC card is computed from two spanned values and is not itself spanned.

Test build: bash build.sh parts/ORDER build/fc-D -> PASS.
