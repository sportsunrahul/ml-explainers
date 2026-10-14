# Fact check: explainer.src.html against primary sources

Checked the visible prose, captions, glosses, tables, every .eqno line, the family tree and the "Where to go deeper" list against sources/<slug>/tex, paper.pdf and abs.html. All 143 data-claim ids resolve in claims.json. All dates match abs.html v1 submission times. All PDF version stamps match the versions named in the header. Equation numbers were checked against the PDF equation tags.

Findings are listed most severe first.

## MUST-FIX

1. **ch-fast (EDM), records paired in the wrong order.**
   Quote: "CIFAR-10 FID 1.97 at 35 NFE unconditionally and 1.79 at 35 NFE class-conditionally, beating the previous records (1.85 and 2.10)".
   Problem: the page lists unconditional first, so the reader pairs 1.97 with 1.85. Read that way, 1.97 does not beat 1.85. In the paper, 1.85 is the conditional record and 2.10 the unconditional one.
   Source: edm/tex/paper.tex line 449: "FIDs of 1.79 and 1.97 for conditional and unconditional CIFAR-10, beating the previous records of 1.85~\cite{Sauer2022} and 2.10~\cite{Vahdat2021}."
   Fix: "(previous records 2.10 unconditional and 1.85 conditional)".

2. **ch-score, exact-likelihood comparison uses the wrong DDPM bound.**
   Quote: "the same DDPM model scores 3.28 bits per dimension exactly, against its bound of 3.70".
   Problem: 3.70 is the bound of DDPM's L-trained model (FID 13.51). The Score SDE row "DDPM 3.28" has ODE FID 3.37, which can only be the L_simple model, whose bound is <= 3.75. Score SDE also flags the DDPM ELBOs with an asterisk as "reported on discrete data", while its own exact NLLs are on uniformly dequantized data, so the two numbers are not strictly comparable.
   Source: score-sde/tex/sampling.tex, Table 2: "DDPM ($L$) ... $\leq$ 3.70* & 13.51", "DDPM ($L_{\text{simple}}$) ... $\leq$ 3.75* & 3.17", "DDPM & 3.28 & 3.37". Also line 60: "except for DDPM ($L$/$L_\text{simple}$) whose ELBO values (annotated with *) are reported on discrete data". The source does not name the checkpoint outright. The L_simple reading is inferred from FID 3.37.
   Fix: "3.28 bits per dimension, against the <= 3.75 bound DDPM reported for it (on discretized data)". Point the span at the existing id page-ddpm-simple-nll (3.75), not page-ddpm-bound-nll.

3. **ch-latent, "at equal compute" is wrong.**
   Quote: "at equal compute, f = 8 beat pixel-space diffusion by an FID gap of 38".
   Problem: the comparison is at equal training steps. LDM-8 does much less work per step than LDM-1, so the compute was not equal.
   Source: ldm/tex/ms.tex line 1006: "a significant FID gap of 38 between pixel-based diffusion (LDM-1) and LDM-8 after 2M training steps."
   Fix: "after the same 2M training steps, f = 8 beat pixel-space diffusion ...".

4. **viz-latent, f = 32 grid shape contradicts the table cited in the caption.**
   JS: `32: ['8x8x32', 'page-ldm-rfid-32', 31.83]`. Caption: "Grid shapes and reconstruction scores are sourced from LDM v2 Table 8".
   Problem: Table 8 has no z-shape column. It gives channel counts, and its f = 32 VQ row (the row the R-FID 31.83 comes from) has c = 16, so the latent is 8x8x16. The 8x8x32 figure comes from a different table, the hyperparameter tables (Tables 13/14). There it is printed as "88 x 8 x 32", which is a typo, and it conflicts with Table 8. The claims.json loc for page-ldm-z-8 also names a "Table 8 z-shape column" that does not exist.
   Source: ldm/tex/ms_tables.tex line 398: "32 & 16384 & 16 & 31.83 & ..." (columns f, |Z|, c, R-FID). ms_tables_supp.tex line 162: "$z$-shape & ... & $88 \times 8 \times 32$".
   Fix: use 8x8x16 to match the Table 8 row whose R-FID is shown, or cite Tables 13/14 for shapes and note the inconsistency. Also correct the claims.json loc.

## NIT

5. **viz-cfg-sweep / ch-guidance, "better than ADM-G" without the paper's own cost caveat.**
   Quote: "FID goes from 7.27 unguided down to 2.43 vs 2.97 for classifier-guided ADM-G". A later paragraph says "The cost is two network evaluations per step instead of one."
   Severity note: this mirrors how CFG's own Table 2 is framed, and the page does state the 2x cost two paragraphs later, so this is an improvement rather than an error. Problem: the reader is left to conclude that CFG beat ADM-G. CFG's authors say the fair speed comparison is T = 128, and at T = 128 CFG loses to ADM-G.
   Source: cfg/tex/main.tex, Sec. 4.3: "each sampling step for our method requires evaluating the denoising model twice ... the fair comparison in terms of sampling speed would be our $T=128$ setting, which underperforms compared to ADM-G in terms of FID score." Table 2, w = 0.3, T = 128: 3.04 (vs ADM-G 2.97).
   Fix: add "(at equal network evaluations, T = 128, CFG's best is 3.04, slightly worse than ADM-G, as the paper itself notes)".

6. **viz-timesteps caption and ch-sd3 text, the shift value is presented as if computed from pixel counts.**
   Caption: "here with the shift factor \sqrt{m/n} = 3.0 used at 1024x1024". Text: "\alpha = \sqrt{m/n}, the square root of the ratio of pixel counts (Eq. 23)".
   Severity note: SD3 itself defines alpha := sqrt(m/n) and then uses 3.0, so the page follows the paper's own framing. Problem: the reader can infer that 3.0 follows from the resolution ratio. It does not: from 256^2 to 1024^2 the ratio is sqrt(16) = 4. SD3 chose alpha = 3.0 from a human-preference study, and it notes that the constant-image assumption behind the formula "is not realistic".
   Source: sd3 PDF p.10 (Sec. 5.3.2): "Note that the assumption of constant images is not realistic. To find good values for the shift value alpha := sqrt(m/n) during inference, we apply them to the sampling steps of a model trained at resolution 1024 x 1024 and run a human preference study ... we thus use a shift value of alpha = 3.0".
   Fix: "(alpha = 3.0, picked by a human-preference study at 1024x1024; the constant-image argument alone would give 4 relative to 256x256)".

7. **ch-latent, date vs. CFG-guided numbers.** The chapter introduces "Rombach and colleagues (20 Dec 2021)" and then reports FID 3.60 with classifier-free guidance and the 1.45B text-to-image model's MS-COCO 12.63. Both results were added in the v2 revision (13 Apr 2022). The Dec 2021 v1 had neither.
   Source: ldm/tex/ms.tex lines 1539-1541 (Changelog): "We updated the results on text-to-image synthesis ... obtained by training a new, larger model (1.45B parameters) ... Both the updated text-to-image and the class-conditional model now use classifier-free guidance".
   Fix: "(numbers from the Apr 2022 v2 revision, which added CFG)".

8. **ch-fast vs ch-consistency, two different EDM numbers.** ch-fast gives EDM's unconditional CIFAR-10 FID as 1.97 at 35 NFE. ch-consistency gives "2.04 at 35 NFE for the EDM teacher itself". Each is correct for its source (EDM Table 2, and CM's own re-measurement in CM Table 1), but a reader sees an unexplained contradiction, and the page itself warns to compare numbers only within one table.
   Fix: "2.04 at 35 NFE (CM's own measurement; EDM reported 1.97) for the EDM teacher".

9. **ch-dit, "adaLN-Zero was also the cheapest".** adaLN-Zero ties plain adaLN. The main text gives both as 118.6 Gflops, and the appendix table has adaLN slightly lower (118.56 vs 118.64). The paper only claims it is more efficient than cross-attention and in-context.
   Source: dit/tex/main.tex line 215, "adaptive layer norm (adaLN, 118.6 Gflops) or adaLN-zero (118.6 Gflops) ... lower FID than both cross-attention and in-context conditioning while being the most compute-efficient". Appendix: "adaLN & 25.21" at 118.56 Gflops.
   Fix: "and, with plain adaLN, the cheapest".

10. **ch-fast (DDIM), "a simple (Euler) discretization of the probability flow ODE (its Proposition 1)".** Two things are conflated. DDIM Eq. 13 is Euler on DDIM's own ODE (Eq. 14), which steps in sigma. Proposition 1 says only that this ODE is equivalent to the VE probability flow ODE, and the paper stresses that the samplers differ.
   Source: ddim PDF Sec. 4.3: "While the ODEs are equivalent, the sampling procedures are not, since the Euler method for the probability flow ODE will make the following update ... we take Euler steps with respect to d sigma(t) ... whereas Song et al. (2020) take Euler steps with respect to dt."
   Fix: "an Euler discretization, in the noise level rather than in time, of an ODE equivalent to the probability flow ODE (Proposition 1)".

11. **ch-score, headline Score SDE numbers come from different models.** Quote: "With a better architecture the paper reached CIFAR-10 FID 2.20, IS 9.89, 2.99 bits per dimension". FID 2.20 and IS 9.89 are NCSN++ cont. (deep, VE), sampled with a PC sampler. The 2.99 bpd is DDPM++ cont. (deep, sub-VP).
   Source: score-sde/tex/sampling.tex, Table 2 "DDPM++ cont. (deep, sub-VP) & 2.99"; Table 3 "NCSN++ cont. (deep, VE) & 2.20 & 9.89".
   Fix: "... FID 2.20 and IS 9.89 with one model, and 2.99 bits per dimension with another".

12. **ch-score, "DDPM's sampler and NCSN's annealed Langevin are two particular ways to discretize it [the reverse SDE]".** DDPM's ancestral sampler is a discretization of the reverse VP SDE. Score SDE casts NCSN's annealed Langevin as a corrector (Langevin MCMC) paired with an identity predictor, not as a discretization of the reverse SDE.
    Source: score-sde Sec. 4.2 (predictor-corrector samplers; SMLD = identity predictor + annealed Langevin corrector).
    Fix: "... DDPM's sampler discretizes it, and NCSN's annealed Langevin is a Langevin 'corrector' run at each noise level".

13. **ch-denoise (Improved DDPM), 2.94 bpd reads as if it came from L_hybrid.** The sentence introduces L_hybrid, then says "With the cosine schedule and importance-sampled bound this reached 2.94". The 2.94 comes from pure L_vlb with importance sampling, and that model has FID 11.47. L_hybrid gives 3.17.
    Source: improved-ddpm/tex/main.tex, CIFAR-10 ablation table: "500K & 4K & cosine & $L_{\text{vlb}}$ & 2.94 & 11.47" and "cosine & $L_{\text{hybrid}}$ & 3.17 & 3.19".
    Fix: "Training on the importance-sampled bound alone, with the cosine schedule, reached 2.94 (at a much worse FID, 11.47); the hybrid gave 3.17".

14. **ch-fast (EDM), "3.2x for DDIM".** This is the ADM "dropout" ImageNet-64 model sampled with its original DDIM sampler, not a separate DDIM-trained model.
    Source: edm/tex/paper.tex Sec. 3: "We also evaluate the 'ADM (dropout)' model ... on class-conditional ImageNet at 64x64"; Discussion: "7.3x for VP, 300x for VE, and 3.2x for DDIM".
    Fix: "3.2x for the ImageNet-64 ADM model's DDIM sampler".

15. **ch-fast (EDM), "every earlier method is just a row of choices in one table (Table 1)".** In EDM Table 1 each method is a column and each design choice is a row.
    Fix: "a column of choices".

16. **ch-latent, "271 vs 916 V100-days, without guidance".** The meaning is unclear. 916 is ADM's diffusion model alone; ADM-G, the model actually compared, costs 962 including its classifier. LDM-4-G's 271 already includes CFG.
    Source: ldm/tex/ms_tables_supp.tex Table 18: "ADM-G ... (250 steps) & 916 & 46 & 962 & 0.07 & 608M & 4.59".
    Fix: "(271 vs 962 V100-days including ADM-G's classifier; 916 without it)".

17. **ch-sd3, "the largest model scored 0.74" on GenEval.** That score is the depth-38 model at 1024^2 after DPO fine-tuning. Without DPO it scores 0.68 at 512^2, which still beats 0.67, so the conclusion stands.
    Source: sd3/tex/tables.tex: "Ours (depth=38), $1024^2$ w/DPO & 0.74", "Ours (depth=38), $512^2$ & 0.68".
    Fix: "the largest model, DPO-finetuned at 1024x1024, scored 0.74".

18. **ch-guidance (CFG), "its gradients can be adversarial noise rather than meaningful direction".** CFG raises this as an open question and a motivation. It does not report it as a finding.
    Source: cfg/tex/main.tex intro: "classifier-guided diffusion sampling can be interpreted as attempting to confuse an image classifier with a gradient-based adversarial attack. This raises the question of whether ..."
    Fix: "and the CFG authors worried its gradients might act like an adversarial attack on the classifier".

19. **ch-deeper, DDIM, "Sec. 3 (Eqs. 7, 9 and 12)".** Eq. 12 is in Sec. 4.1, not Sec. 3 (ddim PDF: "4.1 DENOISING DIFFUSION IMPLICIT MODELS ... (12)").
    Fix: "Sec. 3 (Eqs. 7, 9), Sec. 4.1 (Eq. 12)".

20. **ch-deeper, DDPM, "Sec. 3.2 end to end, Eqs. 4, 6-7, 11, 12 and 14".** This reads as though all these equations are in Sec. 3.2. Eqs. 4 and 6-7 are in Sec. 2 (Background) and Eq. 14 is in Sec. 3.4 (Simplified training objective) (ddpm/tex/main.tex lines 135-177, 287-293).
    Fix: "Sec. 2 (Eqs. 4, 6-7), Sec. 3.2 (Eqs. 11-12), Sec. 3.4 (Eq. 14)".

21. **ch-denoise eqno, "Rearranged from DDPM v2 Sec. 3.2 (the argument inside Eq. 11)".** This is correct, but DDPM states the same formula directly as Eq. 15 (Sec. 4.3): "x0 ~ x^0 = (xt - sqrt(1 - abar_t) eps_theta(xt)) / sqrt(abar_t)   (15)".
    Fix: "Eq. 15 in DDPM v2 (also the argument inside Eq. 11)".

22. **Badge definition vs two Computed figures.** The key says "Computed means a real formula from a paper, run on a small two-dimensional toy dataset". viz-timesteps (SD3 densities) and the left panel of viz-schedule plot 1-D paper formulas and involve no 2-D toy. viz-reflow's 2-rectified flow is a kernel-regression stand-in, which the caption discloses, but the badge alone says Computed.
    Fix: broaden the key ("run on a toy dataset or evaluated directly"), or add "partly approximated" to the viz-reflow caption lead.

23. **viz-latent, Toy model badge on a figure that also shows sourced numbers.** The R-FID values come from paper tables. The caption discloses this. A second, smaller "Sourced" tag on the bar readout would make the mix obvious.

24. **viz-cfg-sweep, "The white marker follows the toy's w slider".** Both use CFG's w convention, but the same w value has no quantitative correspondence between a two-class 2-D toy and ImageNet 128. A reader may read the marker position as a prediction.
    Fix: add "(same knob, not the same effect size)".

## UNVERIFIED (not contradicted, but not checkable against the sources, or toy behaviour I did not run)

25. ch-generate: "The results on real images lagged far behind GANs". Sohl-Dickstein reports no FID or IS. The only number is DDPM Table 1's "Diffusion (original) NLL <= 5.40". The claim is plausible but has no source here.
26. ch-latent: "The openly released Stable Diffusion models were built on this recipe". Not stated in any source paper. DiT does confirm that it uses "an off-the-shelf pre-trained VAE ... from Stable Diffusion".
27. ch-sample toy: "With few, large steps ... the spiral shrinks inward: the toy version of the blurry, washed-out samples a naively shortened DDPM produces." I did not run the JS. In DDIM Table 1 the naive short DDPM with the larger variance fails by excess noise (FID 367.43 at 10 steps), not only by blur.
28. ch-fast toy: "at three steps its [Heun's] correction overshoots badly ... from about ten steps on it tracks the exact curve several times more closely than Euler for the same number of network calls". This is a toy measurement and was not rerun.
29. ch-fast toy: "switching DDIM's spacing to EDM's ... is what rescues the few-step samples here". This holds for the toy only. For the real ImageNet-64 DDIM model, EDM reports that the schedule change is a no-op (red curve identical to green) and the gains come from spacing plus Heun. The sentence says "here", so it is not wrong, but it was not checked.

## Checked and correct (summary, no action)

- Displayed equations match their papers, modulo the notation changes the page states: DDPM Eqs. 2, 4, 6-7, 12, 14 and Alg. 2 line 4; Improved DDPM Eqs. 15, 16, 17, 19 (Eq. 17 tag confirmed in the PDF); NCSN Eq. 4; Score SDE Eqs. 5, 6, 9, 11, 13; DDIM Eq. 12; EDM Eqs. 1, 2-3, 5, 7; ADM Alg. 1; CFG Eq. 6; LDM Eq. 2 and the Sec. 3.3 attention; RF Eqs. 1-2 and Theorems 3.3/3.5/3.7; FM Eqs. 9, 20 and Theorem 2; CM Eqs. 5, 7 and 10; SD3 Eqs. 13, 19, 23 (Eq. 19 tag confirmed in the PDF).
- Time conventions are stated correctly: RF t = 0 noise / t = 1 data; SD3 t = 0 data / t = 1 noise; DDIM alpha = DDPM alpha-bar.
- Sourced JS arrays match the sources exactly: the viz-progressive Q array equals cifar10_sample_quality.csv FID (rounded to 2 dp, x-axis "T - t" as in the paper's figure); the viz-cfg-sweep FID/IS arrays equal the CFG Table 2 T = 256 column for all 14 w values; the viz-dit T object equals the DiT appendix table (Gflops, params, FID) for all 12 adaLN-Zero 400K models.
- Other items checked and found correct: DDIM Table 1 (all 15 cells), DiT block-design FIDs, ADM Table 4 guidance numbers, FM Table 1 ImageNet-64, RF Table 1(a), CM Tables 1-2, the LDM Table 8 R-FIDs and Tables 2/3/18, SD3 AE FIDs, the r = -0.982 correlation (read off img/scale_val_squeeze/00_coco_val_loss_elo.png), DDPM 17 s per 256 images on TPU v3-8, and DDIM's 20 h / 1000 h estimates on a 2080 Ti.
