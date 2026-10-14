# Fork E: outliers, GPTQ/AWQ, outliers atlas

Scope: parts/ch-outliers.html, parts/ch-gptq-awq.html, parts/kit/70-actsurf.js, 71-gptq.js, 72-awq.js,
atlas/parts/outliers.article.html, atlas/parts/outliers.dialogs.html. All spanned claims were looked up in
notes/claims.json, and the prose around them was checked against sources/{dettmers2022,xiao2023,
yuan2022ptq4vit,frantar2023,lin2024,frantar2022obc,sun2024wanda,frantar2023sparsegpt}. Kit readouts were
checked against notes/sim-actsurf.txt, sim-gptq.txt and sim-awq.txt. Test build into build/fc-E: PASS.

## Must-fix

1. parts/ch-gptq-awq.html, GPTQ figure caption: "the paper shows that block size does not change the result".
   Problem: GPTQ runs no block-size experiment. It argues (Sec. 4, Step 2) that "the final rounding decisions
   for column i are only affected by updates performed on this very column". Changed to "the paper explains
   why blocking does not change the rounding decisions". FIXED.
2. parts/ch-gptq-awq.html, speed paragraph: "the paper notes that its kernels do not apply directly to large
   batches". Problem: GPTQ Sec. 6 says only that the study "does not consider ... speedups in batched
   execution". Changed to "the paper lists speedups in batched execution as outside its study". FIXED.
3. parts/ch-gptq-awq.html, pruning section: "Wanda ... is AWQ's idea applied to pruning". Problem: this is
   our framing, stated as fact. Wanda's own motivation is Dettmers et al.'s outlier features (main.tex l.129).
   It cites AWQ only in related work. Changed to "brings the same signal to pruning (our framing; the paper
   builds on LLM.int8()'s outlier features, not on AWQ)". FIXED.
4. atlas/parts/outliers.article.html: "the extra 16-bit multiply makes it slower than plain FP16 for smaller
   models". Problem: SmoothQuant Table 11 shows LLM.int8() slower than FP16 at OPT-13B (237.1 against
   152.6 ms) and at OPT-30B/256 (387.9 against 343.0 ms). The caption says "mostly slower". Dettmers Table 5
   shows a slowdown at 6.7B-sized layers and below. "For smaller models" understates this. Changed to
   "often makes it slower than plain FP16". FIXED.
5. atlas/parts/outliers.dialogs.html: "Dettmers et al. report up to 20x in models from 6.7B parameters up;
   Xiao et al. report about 100x in OPT-13B." Problem: Dettmers's "up to 20x" describes features that appear
   before the 6.7B phase shift too (intro l.128; Table 4 covers 125M-13B). Xiao's "~100x" is a general
   Sec. 3 statement and is not tied to OPT-13B; their OPT-13B Fig. 4 shows channels "greater than 70".
   Rewritten to say this. FIXED.

## Nits

- ch-outliers, PTQ4ViT paragraph: "With it, DeiT-S ... keeps 79.47%" credits the whole result to twin
  uniform quantization. The 79.47 is full PTQ4ViT, which also uses a Hessian-guided scaling metric. Not edited.
- ch-outliers: "That last step is memory-bound" (about decoding) is not in SmoothQuant. It is general
  background and is consistent with GPTQ Sec. 5. Not edited.
- outliers dialog: "so they cannot simply be clipped" is our inference from Dettmers's zeroing ablation.
  It is reasonable, but it is not stated in the paper. Not edited.
- atlas article: "SmoothQuant measured 2-3 levels". The paper computes the number from a formula; it does
  not measure it. Not edited.
- ch-gptq-awq: "Unlike GPTQ, AWQ does not fit the weights to the calibration data; it only measures
  activation sizes" follows the paper's own claim. AWQ's alpha search does evaluate L(s) on calibration
  inputs, and it also uses MSE-based weight clipping. The text says "The paper argues", which is fair.

## Verified, no change

- All 60+ spanned values in both chapters match claims.json and the primary sources. This includes Dettmers
  Tables 1/4/5, Sec. 4.1-4.2, SmoothQuant Tables 1/3/8/11 and Eq. 2-4, GPTQ Tables 1/2/5/6 and Eq. 1-2,
  Alg. 1, AWQ Tables 1/2/4/6/7/10 and Figs. 8/9, OBC Table 1, SparseGPT Table 1, and Wanda Table 3.
- Kit readouts (actsurfText, gptqText, awqText) print toy values only. They match the sim notes, and the
  chapter prose matches them: about 3 levels, 45% -> 1.6%, 4.7%/2.3% at alpha 0.5, output error 1.41 -> 0.40,
  GPTQ 5.4x below RTN, AWQ optimum at 0.42 with about 2x lower error. The only sourced number is 6.0, which
  is spanned.
- The badges are honest. actsurf is "Toy model" (invented data, papers' quantizers). GPTQ and AWQ are
  "Computed" (paper algorithms on toy data, and the captions list our conventions).
- Chapter cross-references (chapters 1, 5, 6, 9) match the ORDER numbering.

## Open doubts

- AWQ caption: "20 grid points spaced evenly from 0 to 1, including both ends" is labelled as our reading.
  The paper gives only "grid size of 20" over [0, 1].
- Dettmers caveat in the claim card ("the step to 6.7B also switches model family, GPT-J to fairseq") is
  labelled as our reading. Table 4 supports it (GPT-J 6.0B, FSEQ 6.7B), but the paper does not raise it.
