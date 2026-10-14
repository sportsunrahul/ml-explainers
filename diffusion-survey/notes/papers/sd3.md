# Stable Diffusion 3 (Esser, Kulal, Blattmann, et al. 2024) -- arXiv 2403.03206v1

"Scaling Rectified Flow Transformers for High-Resolution Image Synthesis". Stability AI. ICML 2024.
Sources: tex/main.tex, tex/sec/{1_intro,2_background,3_method,4_experiments,5_conclusion,X_suppl}.tex, tex/tables.tex and tex/figures.tex (tables/figures are macros defined there), tex/tikz/mm_block2.tex (architecture figure).
PDF section map: 2 = Simulation-Free Training of Flows; 3 = Flow Trajectories (3.1 Tailored SNR Samplers); 4 = Text-to-Image Architecture; 5.1 = Improving Rectified Flows; 5.2.1-5.2.3 = autoencoders / captions / backbones; 5.3.2 = high-resolution finetuning (QK-norm, timestep shift); 5.3.3 = scaling results; Appendix A = Background (sec/2_background.tex).

**Time convention is REVERSED relative to Rectified Flow and Flow Matching: here x_0 is data, epsilon is noise, t=0 is data and t=1 is noise.** The RF path is z_t = (1-t) x_0 + t epsilon and the velocity target is epsilon - x_0.

## Core problem
Rectified flow (straight noise-data paths) has better theory and fewer-step sampling than diffusion, but had only been shown in small class-conditional settings. Can it beat established diffusion formulations for large-scale, high-resolution text-to-image synthesis, and what backbone should carry it?

## What prior methods did and why that hurt
- LDM / Stable Diffusion 1.x-2.x, SDXL: diffusion in the latent space of a 4-channel autoencoder, U-Net backbone, text injected by cross-attention from a *fixed* text representation (text never updated by the image). Diffusion ODE trajectories are curved, so many steps are needed.
- DiT: transformer backbone with adaLN modulation, but only class-conditional; CrossDiT/PixArt add text by cross-attention. UViT is a U-Net/transformer hybrid.
- Earlier rectified flow work sampled t uniformly, which spends as much training on the easy endpoints as on the hard middle.
- No scaling laws for text-to-image diffusion transformers had been established.

## The one key idea
Two things combined and then scaled: (1) rectified flow trained with timesteps drawn from a logit-normal distribution that concentrates on intermediate noise levels, chosen from a controlled comparison of 61 diffusion/flow formulations; (2) MM-DiT, a transformer where text tokens and image tokens each have their own weights but attend jointly in one attention operation, so information flows both ways.

## Delta from predecessors
- vs LDM/SD: same latent-diffusion recipe (frozen autoencoder, frozen text encoders), but rectified flow instead of eps-prediction on the LDM-linear schedule; 16-channel latents instead of 4; transformer instead of U-Net; three text encoders; 50/50 synthetic (CogVLM) and original captions.
- vs DiT: keeps patchify (2x2), adaLN-style modulation by timestep and pooled text vector c_vec, and the transformer block; adds the text sequence as a second token stream with separate weights (DiT with concatenated text is the special case of one shared weight set). Adds QK RMSNorm for stability at high resolution.
- vs Rectified Flow / Flow Matching: same straight-path CFM loss (the RF paper's objective, the FM paper's sigma_min = 0 OT path), but non-uniform t sampling (equivalently a loss weighting), and a resolution-dependent timestep shift. No reflow or distillation is used.

## Mechanism step by step
1. Encode image X to latents x = E(X) with a 16-channel, factor-8 autoencoder; normalize latents by global mean/std.
2. Encode the caption with CLIP L/14, OpenCLIP bigG/14 and T5-v1.1-XXL (all frozen, embeddings precomputed). Pooled CLIP outputs concatenated to c_vec (2048-d). Penultimate CLIP token features concatenated channel-wise (77 x 2048), zero-padded to 4096 channels, then concatenated along the sequence with T5 features (77 x 4096) to c_ctxt (154 x 4096). Each encoder's output is independently zeroed during training for classifier-free guidance.
3. Sample t: draw u ~ N(m, s), t = logistic(u) (logit-normal, default m=0, s=1). At 1024^2 the timesteps are additionally shifted toward noise (shift alpha = 3.0).
4. Make the noisy latent z_t = (1-t) x_0 + t epsilon; target velocity is epsilon - x_0.
5. MM-DiT (wiring confirmed from tikz/mm_block2.tex, Figure 2b): patchify z_t into 2x2 patches plus positional embeddings; embed c_ctxt to the same width. Conditioning y (timestep embedding + c_vec) goes through SiLU + Linear to produce per-stream modulation parameters (alpha, beta, gamma, delta, epsilon, zeta separately for the text stream c and image stream x). In each block, each stream does its own LayerNorm -> modulation (alpha * . + beta) -> Linear to Q/K/V, with optional RMSNorm on Q and K; the Q, K, V of both streams are concatenated and ONE joint attention is computed; the output is split back, and each stream has its own output Linear, gate (* gamma), residual add, LayerNorm -> modulation (delta, epsilon) -> MLP -> gate (* zeta) -> residual. Paper prose: "equivalent to having two independent transformers for each modality, but joining the sequences of the two modalities for the attention operation."
6. Model size is set by depth d: hidden size 64 d, MLP 4 x 64 d, d attention heads. Largest: d = 38, 8B parameters.
7. Train at 256^2, then finetune at higher resolution with mixed aspect ratios (bucketed batches, extended+interpolated position grids), timestep shift, and QK-normalization; then DPO preference finetuning with LoRA (rank 128).
8. Sample by Euler integration of the ODE from t=1 (noise) to t=0 (data) with classifier-free guidance; decode with the autoencoder.

## Equations (TeX copied verbatim from source)

Macro key: `\mathcal{L}_{CFM}` is written out literally; `\text{logit}(t) = \log \frac{t}{1-t}`; `\RR` = real numbers.

### Eq. (13) -- rectified-flow forward process (sec/3_method.tex, unlabeled, Sec 3 "Rectified Flow")
```tex
z_t = (1-t) x_0 + t \epsilon\;,
```
Gloss: x_0 = a clean image latent (data); epsilon = standard Gaussian noise; t in [0,1] with t=0 clean and t=1 pure noise; z_t = the noisy latent on the straight line between them. The network v_Theta(z_t, t) predicts the constant velocity epsilon - x_0. The paper states this "uses L_CFM which then corresponds to w_t^RF = t/(1-t)" in its unified weighted-eps-loss notation.

### Eq. (8) -- Conditional Flow Matching loss as used here (sec/3_method.tex, label eq:condflowmatch, Sec 2)
```tex
   \mathcal{L}_{CFM} =  \mathbb{E}_{t, p_t(z | \epsilon), p(\epsilon) }|| v_{\Theta}(z, t) - u_t(z | \epsilon)  ||_2^2\;, \label{eq:condflowmatch}
```
Gloss: note SD3 conditions on the noise epsilon rather than on the data point (FM conditions on x_1); v_Theta = network velocity; u_t(z|epsilon) = the per-sample target velocity; the expectation averages over t, over noise, and over noisy latents. Same objective family as Flow Matching Eq. 9.

### Eq. (18) -- a timestep density is a loss weighting (sec/3_method.tex, unlabeled, Sec 3.1)
```tex
  w_t^\pi = \frac{t}{1-t}\pi(t)
```
Gloss: pi(t) = the density from which training timesteps are drawn; w_t^pi = the equivalent per-timestep loss weight in the unified notation. Sampling some t more often is the same as weighting their loss more.

### Eq. (19) -- logit-normal timestep density (sec/3_method.tex, label eq:logitnormal, Sec 3.1)
```tex
\pi_{\text{ln}}(t; m, s) = \frac{1}{s\sqrt{2\pi}} \frac{1}{t(1-t)}\exp\Bigl(-\frac{(\text{logit}(t)-m)^2}{2s^2}\Bigr),
```
Gloss: t = timestep; logit(t) = log(t/(1-t)); m = location (negative biases training toward data/low noise, positive toward noise); s = scale (width). In practice u ~ N(m, s) then t = 1/(1+e^{-u}). With m=0, s=1 the density peaks at t=0.5 and vanishes at the endpoints, because the middle is where predicting epsilon - x_0 is hardest (at t=0 or t=1 the optimal prediction is just a mean).

### Eq. (23) -- resolution-dependent timestep shift (sec/4_experiments.tex, label eq:timeshift, Sec 5.3.2)
```tex
t_m = \frac{\sqrt{\frac{m}{n}}t_n}{1+(\sqrt{\frac{m}{n}}-1)t_n}
```
Gloss: n = pixel count of the base resolution, m = pixel count of the target resolution; t_n = a timestep at the base resolution; t_m = the timestep at the new resolution with the same uncertainty about the image. Derivation: for a constant image, averaging n noisy pixels estimates the value with std t/((1-t) sqrt(n)), so more pixels leak more signal at the same t and you need to push t toward noise. alpha = sqrt(m/n) is the shift; equivalently the log-SNR shifts by -log(m/n) (Eq. 24-25). Used alpha = 3.0 for training and sampling at 1024^2. Figure 6 (img/timeshift_v1.png) human-preference Elo, read approximately: shift 3.0 ~1062, 6.0 ~1062, 2.0 ~1038, 4.0 ~1024, 1.5 ~914, 1.0 (no shift) ~898.

## Key hyperparameters
- Formulation study (Appendix B.3): batch 1024, AdamW lr 1e-4, 1000 warmup steps, EMA decay 0.99 every 100 batches; ImageNet (as "a photo of a <class>") and CC12M; eval on COCO-2014 val with CLIP score and CLIP-feature FID; 6 Euler sampler settings (50 steps at CFG 1.0/2.5/5.0; 5/10/25 steps at CFG 5.0).
- Scaling study (Sec 5.3.3): 500k steps at 256^2, batch 4096, 2x2 patches; validation loss on COCO every 50k steps, averaged over equally spaced t levels (excluding t=1).
- Winner timestep sampler: rf/lognorm(0.00, 1.00), i.e. m = 0, s = 1.
- Text-encoder dropout: 46.3% per encoder in Sec 5.3.3 text, 46.4% in Appendix B.3 ("roughly train an unconditional model in 10% of all steps"; 0.464^3 = 0.0999, so 46.4% matches the 10% rationale). Inconsistent in the source.
- Autoencoder: 16 latent channels, downsampling f = 8.
- High-res: QK RMSNorm with learnable scale in both streams; bf16 mixed precision with AdamW eps = 1e-15; shift alpha = 3.0 at 1024^2.
- DPO: LoRA rank 128 on all linear layers; 4k iterations (2B), 2k (8B).
- Caption mix: 50% CogVLM synthetic, 50% original.

## Headline results
1. Formulation comparison (61 formulations, 2 datasets, 24 control settings, non-dominated sorting on CLIP and FID): rf/lognorm(0.00, 1.00) has best average rank 1.54 overall (1.25 at 5 steps, 1.50 at 50 steps). Plain uniform rf ranks 5.67; LDM's eps/linear 2.88; EDM with its original parameters (-1.2, 1.2) 15.58. Only rectified flow with modified timestep sampling beats eps/linear. (Table 1, Sec 5.1.) rf/lognorm(0.50, 0.60) is best at 50 steps (rank 1.00) but poor at 5 steps (8.50).
2. Autoencoder channels: reconstruction FID 2.41 (4 ch) / 1.56 (8 ch) / 1.06 (16 ch), PSNR 25.12 / 26.40 / 28.62 (Table 3). The 16-channel latent needs more model capacity; its generation-FID gap to 8-channel becomes negligible at depth 22 (Figure 10, Appendix), so they pick 16 for scaling.
3. MM-DiT beats DiT (concatenation), CrossDiT and UViT on CC12M training curves; three weight sets (CLIP and T5 separate) give only a small gain over two, so two are used (Figure 4, Sec 5.2.3). Qualitative from plots.
4. Scaling: validation loss decreases smoothly with model size and training steps/FLOPs for images and videos, with no saturation; validation loss correlates strongly with GenEval, T2I-CompBench and human preference (Figure 8; no correlation coefficient given). Scaled to 8B parameters and 5e22 training FLOPs (Conclusion).
5. GenEval: depth-38 (8B) at 1024^2 with DPO scores 0.74 overall vs DALL-E 3 0.67, SDXL 0.55, PixArt-alpha 0.48 (Table 5). Without DPO at 512^2: 0.68.
6. Human preference on PartiPrompts: the 8B model "compares favorabl[y]" to DALL-E 3 and Ideogram v1 (proprietary) and SDXL-Turbo, Playground v2.5, PixArt-alpha, Wuerstchen (open) in visual aesthetics, prompt following and typography (Figure 7). Win-rate numbers are not in the tex; they are only in the bar chart img/baseline_comp.jpg. Approximate readings of the full 8B model's win rate (aesthetics / prompt following / typography, %): PixArt-alpha ~70/75/99, SDXL ~72/75/93, SDXL-Turbo ~71/72/90, Stable Cascade ~63/69/80, Playground v2.5 ~55/63/98, MJ-V6 ~54/60/88, Ideogram v1.0 ~49/57/54, DALL-E 3 ~53/54/67, Ours w/o T5 ~50/54/62. Note: Ideogram v1 aesthetics is slightly BELOW 50% (SD3 loses narrowly there), and the chart includes Stable Cascade and Midjourney v6, which the prose does not name. The chart plots the full model's win rate, so the "w/o T5" bar (~54, ~62) is the complement of the text's 46% / 38% (win rates of the w/o-T5 model). Dropping T5 at inference: 50% win rate on aesthetics, 46% prompt adherence, 38% typography (Sec 5.3.3).
7. Bigger models need fewer steps: relative CLIP-score drop at 5 vs 50 steps is 4.30% (depth 15), 3.59% (30), 2.71% (38); path length 191.13 / 187.96 / 185.96 (Table 6).

## Limitations and later fixes
- Stated: depth 38 needed learning-rate adjustments at 3e5 steps to avoid divergence (Figure 8 caption); QK-norm "is not a universal recipe".
- Stated: T5-XXL is 4.7B parameters and dominates memory (19.05 GB, Table 7); dropping it hurts typography most.
- Stated: the constant-image argument behind the timestep shift is not realistic; the shift value is chosen by a human preference study.
- Stated in background: rectified flows "do not result in optimal transport solutions"; they cite curvature-reduction work but do not use reflow.
- The formulation study is at small scale (ImageNet/CC12M, COCO metrics); its transfer to 8B is assumed, not re-verified per formulation.
- Later work (external knowledge, not in this paper): SD3 Medium (2B) and SD3.5 releases; Flux (Black Forest Labs, founded by several SD3 authors) uses a related rectified-flow MM-DiT design; distilled few-step variants (e.g. SD3-Turbo via latent adversarial diffusion distillation).

## Official code
Not given as a URL in v1. The abstract/intro say "we will make our experimental data, code, and model weights publicly available".
