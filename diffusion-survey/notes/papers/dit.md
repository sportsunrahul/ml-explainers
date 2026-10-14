# DiT -- Scalable Diffusion Models with Transformers

Peebles (UC Berkeley), Xie (NYU), ICCV 2023. arXiv 2212.09748v2.
Sources read: sources/dit/tex/main.tex (whole body, Tables 1-6, appendix).

## Core problem

Every strong image diffusion model (DDPM, ADM, LDM, DALL-E 2, Imagen) used a convolutional
U-Net as its denoising network. It was unknown whether the U-Net's inductive bias was
actually necessary, and diffusion had not joined the "one architecture everywhere" trend
(transformers in NLP, vision, RL) that brings reusable recipes and predictable scaling.

## What prior methods did and why that hurt

- U-Net backbones inherited from PixelCNN++ (ResNet blocks, self-attention only at low
  resolutions), ablated by ADM but with the high-level design unchanged since DDPM (Sec. 1).
- Pixel-space U-Nets are very expensive: ADM 1120 Gflops at 256x256, ADM-U 742 Gflops,
  ADM 1983 Gflops at 512x512 (Table 6).
- Complexity was usually reported in parameter counts, which ignore resolution / token
  count and so are a poor proxy for model compute (Sec. 2, Architecture complexity).
- No clean scaling story: it was not clear how to spend more compute to get better samples.

## The one key idea

Replace the U-Net inside a latent diffusion model with a plain Vision Transformer that
operates on patches of the VAE latent, and inject the timestep and class label through
adaptive LayerNorm whose residual gates are zero-initialised (adaLN-Zero). Then image
quality scales smoothly with transformer Gflops.

## Delta from predecessor (vs U-Net backbones like ADM and LDM)

- Same diffusion recipe as ADM: t_max = 1000 linear variance schedule from 1e-4 to 2e-2,
  epsilon-prediction with L_simple plus learned covariance Sigma_theta (iDDPM style), ADM's
  timestep/label embeddings, classifier-free guidance, 250 DDPM sampling steps (Sec. 4).
- Same latent space as LDM: off-the-shelf Stable Diffusion VAE, downsample factor 8, so a
  256x256x3 image becomes a 32x32x4 latent z; 512x512 becomes 64x64x4 (Sec. 3.1, Sec. 4, Sec. 5.1).
- Only the backbone changes: convolutional U-Net -> ViT-style transformer over latent
  patches. Hybrid pipeline: convolutional VAE + transformer DDPM.
- Compute: DiT-XL/2 is 118.6 Gflops vs LDM-4 103.6 (Table 6 says 104) and pixel-space ADM
  1120 / ADM-U 742 at 256x256; at 512x512, 524.6 vs ADM 1983 / ADM-U 2813 (Sec. 5.1, Table 6).

## Mechanism, step by step

1. Encode image x with the frozen VAE encoder: z = E(x), shape I x I x C (32x32x4 at 256px).
2. Add diffusion noise to z at timestep t (standard DDPM forward process).
3. Patchify: cut the latent into p x p patches, linearly embed each to a d-dim token.
   Sequence length T = (I/p)^2 (Fig. 4 caption). p in {2, 4, 8}. At I = 32: p=2 -> 256
   tokens, p=4 -> 64, p=8 -> 16 (arithmetic from the formula). Halving p quadruples T and at
   least quadruples Gflops, with almost no change in parameters.
4. Add fixed sine-cosine (frequency) positional embeddings (ViT style).
5. Run N DiT blocks of width d. Conditioning = sum of timestep embedding and class
   embedding. Four options were tried (Sec. 3.2):
   - In-context: append t and c as two extra tokens (like a cls token). ~0 extra Gflops.
   - Cross-attention: extra cross-attention layer over the length-2 [t, c] sequence;
     ~15% Gflops overhead.
   - adaLN: regress LayerNorm scale gamma and shift beta from the conditioning vector,
     instead of learning them as fixed parameters. Cheapest; same function for all tokens.
   - adaLN-Zero: also regress per-dimension gates alpha applied just before each residual
     connection, with the regressing MLP initialised to output alpha = 0, so every block
     starts as the identity function (like zero-init of the last BN in ResNets, or of the
     last conv in diffusion U-Net blocks).
   The conditioning MLP is SiLU + linear with 6d outputs for adaLN-Zero vs 4d for adaLN
   (App. A). Reading 6d as (gamma, beta, alpha) for each of the attention and MLP
   sub-blocks is my inference; the paper states only the 4x / 6x counts.
6. Final (adaptive) LayerNorm, then a linear decoder maps each token to p x p x 2C values
   (noise prediction plus diagonal covariance), rearranged back to I x I x 2C.
7. Sample with 250 DDPM steps (optionally classifier-free guidance), then decode with the
   VAE decoder x = D(z).

Why adaLN-Zero beat the alternatives: at 400K steps (Table 4, all DiT-XL/2): adaLN-Zero
FID 19.47 (118.64 Gflops) vs adaLN 25.21 (118.56), cross-attention 26.14 (137.62),
in-context 35.24 (119.37). So adaLN-Zero is both the best and among the cheapest; the paper
says its FID is "nearly half" the in-context FID, and that identity initialisation is
what separates it from vanilla adaLN (Sec. 5, DiT block design). The paper gives the
empirical result plus the identity-init analogy; it does not give a deeper theoretical
reason.

## Equations for the explainer

DiT has NO numbered display equations; everything is inline in main.tex Sec. 3.1
("Preliminaries") or in figure captions. Verbatim inline TeX:

### Forward noising process (inline, Sec. 3.1, Diffusion formulation)

```
q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})
```
```
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon_t
```
Gloss: x_0 = clean latent; x_t = noisy latent at step t; alpha-bar_t = fraction of signal
variance remaining (set by the fixed linear beta schedule); epsilon_t = standard Gaussian
noise. (In DiT, "x" here is the VAE latent z.)

### Training loss (inline, Sec. 3.1)

```
\mathcal{L}_{simple}(\theta) = ||\epsilon_\theta(x_t) - \epsilon_t||_2^2
```
Gloss: epsilon_theta = the DiT's noise prediction; the loss is mean squared error against
the noise that was actually added. Sigma_theta (covariance) is trained with the full
variational bound, following iDDPM.

### Classifier-free guidance (inline, Sec. 3.1)

```
\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t,\emptyset) + s \cdot \nabla_x \log p(x|c) \propto \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))
```
Gloss: c = class label; the empty-set symbol = learned "null" label (labels dropped at
random in training); s = guidance scale, s = 1 is ordinary sampling, s > 1 pushes samples
toward images the model thinks strongly match class c. DiT-XL/2-G uses s = 1.5 for the
headline result.

### Patchify token count (Fig. 4 caption)

```
T = (I/p)^2
```
Gloss: I = latent side length (32 for 256px images, 64 for 512px); p = patch side; T =
number of tokens the transformer processes. Attention cost grows with T^2, MLP cost with T.

### Training compute estimate (inline, Sec. 5, "Larger DiT models are more compute-efficient")

Plain text in source (not math mode): training compute = model Gflops x batch size x
training steps x 3 (the 3 approximates backward = 2x forward).

## Key hyperparameters

- Configs (Table 1): DiT-S 12 layers / d=384 / 6 heads / 1.4 Gflops; DiT-B 12 / 768 / 12 /
  5.6; DiT-L 24 / 1024 / 16 / 19.7; DiT-XL 28 / 1152 / 16 / 29.1 (Gflops at I=32, p=4).
- Parameters (Table 4): S ~33M, B ~130M, L ~458M, XL ~675M; excludes the 84M-param VAE.
- Naming: DiT-XL/2 = XL config, patch size 2.
- Training (Sec. 4): AdamW, constant lr 1e-4, no weight decay, batch 256, horizontal flips
  only, no warmup, no regularization, EMA decay 0.9999, final linear layer zero-init;
  hyperparameters "almost entirely retained from ADM" and not tuned.
- Diffusion: t_max = 1000, linear variance schedule 1e-4 to 2e-2, learned Sigma (ADM).
- Hardware: JAX on TPU v3; DiT-XL/2 ~5.7 it/s on a TPU v3-256 pod.
- Guidance applied to only the first 3 of 4 latent channels (App. A).

## Headline results

1. ImageNet 256x256 class-conditional: DiT-XL/2-G (cfg=1.50) FID 2.27, sFID 4.60, IS
   278.24, Prec 0.83, Rec 0.57, beating LDM-4-G 3.60 and StyleGAN-XL 2.30 (Table 2).
   Without guidance: FID 9.62 (Table 2). Trained 7M steps.
2. ImageNet 512x512: DiT-XL/2-G (cfg=1.50) FID 3.04 vs previous best ADM-G,ADM-U 3.85
   (Table 3); 3M steps, 1024 tokens from a 64x64x4 latent, 524.6 Gflops (Sec. 5.1).
3. FID tracks Gflops, not parameters: DiT-S/2 (6.06 Gflops, 33M params) FID 68.40 vs
   DiT-B/4 (5.56 Gflops, 130M params) FID 68.38 at 400K steps (Table 4; Sec. 5 text names
   this pair). Shrinking p at fixed config holds params ~fixed (slightly decreasing) while
   Gflops and quality rise (Fig. 6, Fig. 8).
4. Larger models are more compute-efficient: small models trained longer lose to large
   models trained shorter; XL/4 is overtaken by XL/2 after ~1e10 Gflops of training (Fig. 9).
5. Sampling compute cannot substitute for model compute: L/2 at 1000 steps (80.7 Tflops per
   image) FID-10K 25.9 vs XL/2 at 128 steps (15.2 Tflops) FID-10K 23.7 (Sec. 5.2, Fig. 10).
6. At only 2.35M steps (similar to ADM), XL/2 still reaches FID 2.55 (Sec. 5.1).

## Limitations and later fixes

- Class-conditional ImageNet only; text-to-image is left as future work ("drop-in backbone
  for text-to-image models like DALL-E 2 and Stable Diffusion", Conclusion).
- Relies on a frozen, off-the-shelf convolutional VAE; quality depends slightly on the
  decoder (FID 2.46 original / 2.30 ft-MSE / 2.27 ft-EMA, Table 5).
- Uses the 250-step DDPM sampler, so sampling is still slow; no study of fast samplers.
- FID never saturated for the XL/2 models; they were trained "as long as possible" (Table 4 caption).
- Later (general knowledge, not verified here): DiT-style transformers became the backbone of
  many later systems (e.g. SD3's MMDiT, video models); U-ViT (concurrent) and SiT are relatives.

## Official code

Paper states only the project page: https://www.wpeebles.com/DiT.html (main.tex footnote,
Sec. 1). No GitHub URL appears in the tex.
