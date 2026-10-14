# Fact check A: sohl-dickstein, ncsn, ddpm, ddim, score-sde, improved-ddpm, beat-gans, cfg

Checked against sources/<slug>/tex (LaTeX) and paper.pdf (pdftotext) for the version in each claim (arXiv versions confirmed from abs.html: sohl-dickstein v8, ncsn v3, ddpm v2, ddim v4, score-sde v2, improved-ddpm v1, beat-gans v4, cfg v1). Table, figure and equation numbers were checked against the PDF.

Counts: IMPRECISE 6, LOC-WRONG 2, OK 145, UNVERIFIABLE 1 (total 154).

| id | verdict | correction / evidence |
|---|---|---|
| ncsn-toy-sigma-range | IMPRECISE | Source is internally inconsistent; set uncertain:true. Sec. 4.3 (method.tex) says the Fig. 3 demo uses "$L=10$, $\sigma_1 = 10$ and $\sigma_{10} = 0.1$", but App. B.1 (app.tex) says for Fig. 3: "For annealed Langevin dynamics, we use $T=100$, $L=10$ and $\epsilon = 0.1$. We choose ... $L=10$, $\sigma_1 = 20$ and $\sigma_{10} = 1$." Suggested value: "L = 10; sigma_1 = 10, sigma_10 = 0.1 (Sec. 4.3) vs sigma_1 = 20, sigma_10 = 1 (App. B.1); T = 100, eps = 0.1 per App. B.1". |
| score-sde-cifar-pc-settings | IMPRECISE | Loc should be App. H.1 "Settings for architecture exploration". The settings quoted are for the architecture search: "For sampling, we use the PC sampler discretized at 1000 time steps. We choose reverse diffusion ... as the predictor. We use one corrector step ... for VE SDEs with a signal-to-noise ratio of 0.16, but save the corrector step for VP SDEs". For the continuously trained VP/sub-VP models in Table 3, App. H.2 says "we used the Euler-Maruyama solver as the predictor for continuously-trained models, instead of the ancestral sampling predictor or the reverse diffusion predictor". Correct to: "Architecture-exploration sampler (App. H.1): PC, N=1000, reverse diffusion predictor, 1 corrector step with SNR 0.16 for VE, no corrector for VP; DDPM++ cont. uses Euler-Maruyama predictor (App. H.2)". |
| ncsn-cifar10-fid | IMPRECISE | 25.32 is correct, but "MoLM 18.9 is best in table" holds only within the unconditional block. The same Table 1 lists conditional BigGAN at "$\mathbf{9.22}$ & $\mathbf{14.73}$". Correct to "MoLM 18.9 is the best unconditional FID in the table (conditional BigGAN 14.73)". |
| beat-gans-vs-iddpm-in64 | IMPRECISE | Values 2.92 / 2.61 / 2.07 match Table 5 (ImageNet 64x64 block: "IDDPM & 2.92", "ADM & 2.61", "ADM (dropout) & \bf{2.07}"). The label "unconditional-architecture gain" is wrong or at least unsupported: the IDDPM 2.92 baseline is the class-conditional ImageNet 64 model (Improved DDPM Table 4, "class-conditional ImageNet $64 \times 64$"), and beat-gans never states the conditioning of its IN64 model (Sec. 5 names conditional models only at 128/256/512). Reword to "architecture-only gain on ImageNet 64, no classifier guidance". |
| cfg-two-passes | IMPRECISE | The conclusion holds, but w=0.3 is not the best T=128 point. Table 2 T=128 column: "$w=0.3$ & 3.04 / ..." and "$w=0.4$ & 3.02 / ...". Both are above ADM-G "2.97". Suggested value: "T=128 best FID 3.02 (w=0.4); 3.04 at w=0.3; vs ADM-G 2.97". |
| beat-gans-arch-final | IMPRECISE | The Sec. 3.1 last paragraph matches the claim text. But Table 11, also cited in loc, departs from it for two models. ImageNet 64 uses "Depth & 2 & 3 & ..." (3 res blocks). ImageNet 128 uses "Heads & & & 4" with no Heads Channels value (4 fixed heads, not 64 ch/head). Either cite only Sec. 3.1, or add "(Table 11: IN64 depth 3; IN128 uses 4 heads)". |
| ddim-ddpm-sampling-cost | LOC-WRONG | The values are correct but the sentence is in Sec. 1, paragraph 3, not 2. Paragraph 1 starts "Deep generative models have demonstrated...", paragraph 2 "Recent works on iterative generative models...", paragraph 3 "A critical drawback of these models ... For example, it takes around 20 hours to sample 50k images of size $32 \times 32$ from a DDPM ... nearly $1000$ hours". Corrected loc: Sec. 1, paragraph 3. |
| cfg-implicit-classifier | LOC-WRONG | The content is correct. The paragraph right after Eq. 6 is "Eq. (6) has no classifier gradient present..."; the implicit-classifier derivation is in the next one ("Despite the fact that there in general may not exist a classifier ... $\nabla_{\bz_\lambda} \log p^{i}(\bc \| \bz_\lambda) = -\frac{1}{\sigma_{\lambda}}[\bepsilon^*(\bz_\lambda, \bc) - \bepsilon^*(\bz_\lambda)]$"). Corrected loc: Sec. 3.2, second paragraph after Eq. 6. |
| ddim-code | UNVERIFIABLE | Not in any provided source. grep for "ermongroup" in sources/ddim/ (tex, abs.html, PDF text) finds nothing. The only GitHub links in the DDIM tex point to BigGAN-PyTorch, hojonathanho/diffusion and NVlabs/stylegan. uncertain:true is correct. |
| beat-gans-fid-in128 | OK |  |
| beat-gans-fid-in256 | OK |  |
| beat-gans-fid-in512 | OK |  |
| beat-gans-recall-in256 | OK |  |
| beat-gans-ddim25-fids | OK |  |
| beat-gans-upsample-fids | OK |  |
| beat-gans-guidance-cond-unguided | OK |  |
| beat-gans-guidance-cond-s1 | OK |  |
| beat-gans-guidance-cond-s10 | OK |  |
| beat-gans-guidance-uncond | OK |  |
| beat-gans-tradeoff-statement | OK |  |
| beat-gans-guidance-scales-used | OK |  |
| beat-gans-arch-ablation | OK |  |
| beat-gans-adagn | OK |  |
| beat-gans-lsun-bedroom | OK |  |
| beat-gans-sampling-steps | OK |  |
| beat-gans-model-size-in256 | OK | Table 11: "Model size ... 554M", "Iterations ... 1980K", batch 256. |
| cfg-guided-eps-formula | OK |  |
| cfg-label-dropout | OK |  |
| cfg-puncond-values | OK |  |
| cfg-puncond-finding | OK |  |
| cfg-in64-best-fid | OK |  |
| cfg-in64-unguided | OK |  |
| cfg-in64-w1 | OK |  |
| cfg-in64-w4 | OK |  |
| cfg-in128-best-fid | OK |  |
| cfg-in128-unguided | OK |  |
| cfg-in128-w4-vs-biggan | OK |  |
| cfg-in128-w-sweep | OK |  |
| cfg-in128-is-sweep | OK |  |
| cfg-tradeoff | OK | The claim is fine. Note that Sec. 4.1 itself mis-states "FID monotonically decreasing and IS monotonically increasing with $w$"; FID actually rises for w > 0.1-0.3. Do not quote that sentence. |
| cfg-training-setup | OK |  |
| cfg-w-range | OK |  |
| cfg-fig5-legend-typo | OK | Confirmed from tex: \legend{$T=128$,$T=256$,$T=512$} in fig:i128_fid_is_plot (Figure 5), while Table 2 header reads "$T=128/256/1024$". This can be uncertain:false; only the "typo" reading is interpretation. |
| ddim-loss-depends-on-marginals | OK |  |
| ddim-theorem1-same-objective | OK |  |
| ddim-alpha-is-alphabar | OK |  |
| ddim-update-three-terms | OK |  |
| ddim-eta-interpolates | OK |  |
| ddim-no-retraining | OK |  |
| ddim-tau-selection | OK |  |
| ddim-cifar10-fid-steps | OK |  |
| ddim-cifar10-ddpm-fid-steps | OK |  |
| ddim-cifar10-sigmahat-fid-steps | OK |  |
| ddim-celeba-fid-steps | OK |  |
| ddim-speedup-abstract | OK |  |
| ddim-speedup-sec51 | OK |  |
| ddim-speedup-sec5-intro | OK |  |
| ddim-lsun-50-step | OK |  |
| ddim-consistency | OK |  |
| ddim-reconstruction-error | OK |  |
| ddim-ode-ve-equivalence | OK |  |
| ddpm-venue | OK |  |
| ddpm-T | OK |  |
| ddpm-beta-schedule | OK |  |
| ddpm-LT-bits | OK |  |
| ddpm-cifar-fid | OK |  |
| ddpm-cifar-fid-test | OK |  |
| ddpm-cifar-is | OK |  |
| ddpm-cifar-nll | OK |  |
| ddpm-cifar-true-bound | OK |  |
| ddpm-ablation-mu-vs-eps | OK |  |
| ddpm-baseline-fids | OK |  |
| ddpm-sohl-cited-nll | OK | Confirmed: DDPM Table 1 "Diffusion (original) & & & $\leq 5.40$"; Sohl-Dickstein Table 1 gives K = 5.4 +/- 0.2 bits/pixel with the variance-1 scaling note. uncertain:true is appropriate. |
| ddpm-lsun-fid | OK |  |
| ddpm-rate-distortion | OK |  |
| ddpm-params | OK |  |
| ddpm-train-speed | OK |  |
| ddpm-sample-speed | OK |  |
| ddpm-optim | OK |  |
| ddpm-code | OK |  |
| improved-ddpm-venue | OK | Line 37 = \usepackage[accepted]{icml2021} (verified). |
| improved-ddpm-T | OK |  |
| improved-ddpm-cosine-s | OK | PDF eq. (17) is the cosine schedule; s=0.008 and 0.999 clip in the following paragraphs. |
| improved-ddpm-lambda | OK |  |
| improved-ddpm-linear-T4000 | OK |  |
| improved-ddpm-imagenet-baseline | OK |  |
| improved-ddpm-imagenet-hybrid | OK |  |
| improved-ddpm-imagenet-vlb | OK |  |
| improved-ddpm-cifar-ablation | OK |  |
| improved-ddpm-best-nll | OK |  |
| improved-ddpm-ensemble | OK |  |
| improved-ddpm-few-steps-50 | OK |  |
| improved-ddpm-few-steps-100 | OK |  |
| improved-ddpm-ddim-crossover | OK |  |
| improved-ddpm-step-sweep | OK |  |
| improved-ddpm-vs-biggan | OK |  |
| improved-ddpm-imagenet256 | OK |  |
| improved-ddpm-fid-10k-bias | OK |  |
| improved-ddpm-model-size | OK |  |
| improved-ddpm-code | OK |  |
| ncsn-score-definition | OK |  |
| ncsn-manifold-loss-fluctuates | OK | The paper writes N(0, 0.0001) without saying variance or std, but its convention elsewhere puts the variance second ("$\mcal{N}(\tilde{\bfx} \| \bfx, \sigma^2 I)$" in Sec. 4.2; "Draw $\tilde{\bfz} \sim \mcal{N}(0, \sigma_i^2)$" in Alg. 1). So the reading as variance 1e-4 (std 0.01) holds. |
| ncsn-toy-mixture | OK |  |
| ncsn-langevin-ignores-mode-weight | OK |  |
| ncsn-geometric-sigma | OK |  |
| ncsn-image-num-levels | OK |  |
| ncsn-image-sigma-range | OK |  |
| ncsn-lambda-sigma-squared | OK | Eq. 6 only introduces lambda(sigma_i); the choice lambda(sigma) = sigma^2 is stated in the prose after Eq. 6 in Sec. 4.2. |
| ncsn-step-size-schedule | OK |  |
| ncsn-langevin-steps-per-level | OK |  |
| ncsn-base-step-size | OK |  |
| ncsn-init-uniform | OK |  |
| ncsn-cifar10-is | OK |  |
| ncsn-fid-50k | OK |  |
| ncsn-single-noise-baseline-fails | OK |  |
| ncsn-ssm-4x-cost | OK | Source: "requires around four times more computations due to the forward mode auto-differentiation", contrasted with DSM in the same sentence. OK. |
| ncsn-code | OK |  |
| score-sde-forward-sde | OK |  |
| score-sde-reverse-sde | OK |  |
| score-sde-prob-flow-ode | OK |  |
| score-sde-ve-is-smld | OK |  |
| score-sde-vp-is-ddpm | OK |  |
| score-sde-pc-generalizes | OK |  |
| score-sde-pc-vs-predictor | OK |  |
| score-sde-corrector-only-worst | OK |  |
| score-sde-cifar10-fid | OK |  |
| score-sde-cifar10-is | OK |  |
| score-sde-cifar10-nll | OK |  |
| score-sde-exact-vs-elbo | OK | The values match. Caveat: Sec. 4.3 says the DDPM ELBOs "(annotated with *) are reported on discrete data" while 3.28 is on uniformly dequantized data, so the two numbers are not the same quantity. The paper makes the comparison itself. |
| score-sde-ode-nfe-reduction | OK | Figure 3 in PDF is fig:prob_flow; text: "reduced by over $90\%$ without affecting the visual quality". |
| score-sde-1024-celebahq | OK |  |
| score-sde-1024-sampler | OK |  |
| score-sde-vp-beta-range | OK |  |
| score-sde-ve-sigma-min | OK |  |
| score-sde-code | OK |  |
| sohl-dickstein-venue | OK | Line 39 = \usepackage[accepted]{icml2015} (verified). |
| sohl-dickstein-core-idea | OK |  |
| sohl-dickstein-physics | OK |  |
| sohl-dickstein-same-form | OK |  |
| sohl-dickstein-bound | OK |  |
| sohl-dickstein-T-images | OK |  |
| sohl-dickstein-T-toys | OK |  |
| sohl-dickstein-beta-learned | OK |  |
| sohl-dickstein-beta-binomial | OK |  |
| sohl-dickstein-datasets | OK |  |
| sohl-dickstein-cifar-K | OK |  |
| sohl-dickstein-cifar-K-vs-ddpm | OK | Confirmed as stated; uncertain:true is appropriate. |
| sohl-dickstein-earlier-version | OK |  |
| sohl-dickstein-heartbeat | OK |  |
| sohl-dickstein-dead-leaves | OK |  |
| sohl-dickstein-mnist | OK |  |
| sohl-dickstein-inpainting | OK |  |
| sohl-dickstein-no-sample-metric | OK |  |
| sohl-dickstein-code | OK |  |
