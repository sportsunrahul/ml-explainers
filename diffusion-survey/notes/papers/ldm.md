# High-Resolution Image Synthesis with Latent Diffusion Models (LDM)

Rombach, Blattmann, Lorenz, Esser, Ommer. arXiv 2112.10752v2 (v1 submitted 20 Dec 2021; v2 is the CVPR 2022 version). Source: `sources/ldm/tex/ms.tex` (body), `ms_tables.tex`, `ms_tables_supp.tex`, `ms_figures*.tex` (table/figure macros), `commands.tex` (math macros). Code: https://github.com/CompVis/latent-diffusion (`\projectpath` in `commands.tex`, cited in contributions (vi)). This is the method behind Stable Diffusion, but Stable Diffusion itself is not in this paper.

v2 changelog (ms.tex `\section{Changelog}`): text-to-image results replaced by a new 1.45B model with new baselines (GLIDE, LAFITE, Make-A-Scene); class-conditional ImageNet results retrained with larger batch; both now use classifier-free guidance; user study added. So text-to-image and ImageNet numbers below are v2-specific.

## Core problem

Pixel-space diffusion models (ADM) give state-of-the-art quality but are extremely expensive: every training step and every one of the many sampling steps runs a big UNet at full image resolution. ADM needed 150-1000 V100 days to train and about 5 days on one A100 to draw 50k samples (Sec 1).

## What prior methods did and why that hurt

- Pixel-space DMs (DDPM, ADM): spend capacity and compute on imperceptible high-frequency detail ("mode-covering behavior ... spend excessive amounts of capacity ... on modeling imperceptible details", Sec 1). The reweighted loss only partially fixes this.
- Two-stage autoregressive models (VQGAN + transformer, DALL-E): compress to a discrete latent, then model it with a transformer over a 1D token sequence. Because transformers scale badly with sequence length they need aggressive compression (f=16 or more), which loses detail, and the 1D ordering ignores 2D structure (Sec 3.1).
- LSGM: trains encoder, decoder and a score-based prior jointly, which needs delicate balancing of reconstruction vs generation (contribution (iii)).

## The one key idea

Separate perceptual compression from generative learning. First train an autoencoder once that maps images to a smaller latent grid that is "perceptually equivalent" to pixels (mild compression, keeps 2D structure). Then train the diffusion model in that latent space with a normal convolutional UNet. Decode once at the end. Add cross-attention so the UNet can be conditioned on text, layouts, class labels etc.

## Delta from its predecessor (pixel-space diffusion)

- Same denoising objective, but on z = E(x) instead of x (Eq. 1 -> Eq. 2).
- Compute: the UNet runs on an h x w x c latent, e.g. 64x64x3 (f=4) or 32x32x4 (f=8) for 256x256 images, instead of 256x256x3.
- Two-stage training: the autoencoder is trained once and reused across many diffusion models and tasks.
- New general conditioning path (cross-attention with a domain encoder tau_theta), vs ADM's class-embedding-only conditioning.
- Measured: at fixed compute (single A100, same steps and parameters) LDM-8 beats pixel LDM-1 by an FID gap of 38 after 2M steps (Sec 4.1); class-conditional ImageNet LDM-4-G beats ADM-G (FID 3.60 vs 4.59) with 271 vs 962 V100 days (Table 18) and 400M vs 608M params (Table 3).

## Mechanism step by step

1. Stage 1, autoencoder (Sec 3.1, App. G): encoder E maps x (H x W x 3) to z (h x w x c), decoder D maps back. Downsampling factor f = H/h = W/w, f = 2^m.
2. Train it with a perceptual (LPIPS) reconstruction loss plus a patch-based adversarial loss (VQGAN-style discriminator), so reconstructions stay sharp and on the image manifold (Eq. 25).
3. Keep the latent tame with a small regularizer: either KL-reg (KL to a standard normal weighted ~1e-6, VAE-like) or VQ-reg (a vector-quantization layer absorbed into the decoder, codebook size |Z|).
4. For KL latents, rescale z by 1/sigma_hat (component-wise std estimated from the first batch) so the latent has unit variance; VQ latents are already near unit variance (App. G, App. D.1).
5. Stage 2, diffusion (Sec 3.2): freeze E and D. Train a time-conditional UNet eps_theta(z_t, t) with the standard noise-prediction loss on noised latents z_t (Eq. 2).
6. Conditioning (Sec 3.3): a domain-specific encoder tau_theta turns y (text tokens, layout, class) into an M x d_tau sequence; it enters the UNet through cross-attention layers (Q from UNet features, K and V from tau_theta(y)). tau_theta and eps_theta are trained jointly (Eq. 3). Spatially aligned conditions (low-res image, mask, semantic map) are concatenated to the input instead.
7. Sampling: start from Gaussian noise in latent space, denoise with DDIM (e.g. 100-250 steps), optionally with classifier-free guidance, then decode once: x = D(z_0).

## Equations (TeX copied verbatim from the LaTeX source)

Macro notes for rendering (`commands.tex`): `\lsimple` -> `L_{DM}`, `\lsimpleldm` and `\lsimplelcm` -> `L_{LDM}` (both), `\expec` -> `\mathbb{E}`, `\model` -> `\epsilon_\theta`, `\conditioner` -> `\tau_\theta`, `\encoder` -> `\mathcal{E}`, `\decoder` -> `\mathcal{D}`, `\zt{t}` -> `z_{t}`, `\cond` -> `y`, `\lrec` -> `L_{rec}`, `\ladv` -> `L_{adv}`, `\lreg` -> `L_{reg}`, `\disc` -> `D_{\psi}`, `\R` -> `\mathbb{R}`. Equation numbers from the compiled PDF.

### Eq. 1 (Sec 3.2, ms.tex) - pixel-space DM objective (the baseline)
```
\lsimple = \expec_{x, \epsilon \sim \mathcal{N}(0, 1),  t }\Big[ \Vert \epsilon - \model(x_{t},t) \Vert_{2}^{2}\Big] \, ,
```
- x: image; x_t: noised image at step t (t uniform in {1..T}); eps: the true noise; eps_theta: the UNet's noise prediction.

### Eq. 2 (Sec 3.2, ms.tex) - the LDM objective
```
\lsimpleldm := \expec_{\encoder(x), \epsilon \sim \mathcal{N}(0, 1),  t}\Big[ \Vert \epsilon - \model(z_{t},t) \Vert_{2}^{2}\Big] \, .
```
- E(x): latent code of the image from the frozen encoder; z_t: that latent after t steps of noising.
- Plain words: identical to Eq. 1, except everything lives in the small latent grid.

### Cross-attention (Sec 3.3, ms.tex, unnumbered `equation*`)
Inline before it: `$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$`, with
```
Q = W^{(i)}_Q \cdot  \varphi_i(z_t), \; K = W^{(i)}_K \cdot \conditioner(y),
  \; V = W^{(i)}_V \cdot \conditioner(y) . \nonumber
```
- varphi_i(z_t): flattened intermediate UNet features at layer i, N x d^i_eps (N = number of spatial positions).
- tau_theta(y): encoded conditioning, M x d_tau (e.g. M text tokens).
- W_Q, W_K, W_V: learned projections into a shared width d. Queries come from the image, keys/values from the condition, so each latent pixel "looks up" the relevant words.
- d: attention width (softmax temperature sqrt(d)).
- Source quirk: the text gives W_V in R^{d x d^i_eps} and W_Q in R^{d x d_tau}, which is dimensionally swapped relative to the formula (Q multiplies image features, V multiplies tau(y)). Treat as a typo; do not copy the dims.

### Eq. 3 (Sec 3.3, ms.tex) - conditional LDM objective
```
\lsimplelcm := \expec_{\encoder(x), y, \epsilon \sim \mathcal{N}(0, 1), t }\Big[ \Vert \epsilon - \model(z_{t},t, \conditioner(y)) \Vert_{2}^{2}\Big] \, ,
```
- y: the conditioning input (caption, layout, class). tau_theta and eps_theta are optimized jointly through this loss.

### Eq. 25 (App. G "Details on Autoencoder Models", ms.tex) - stage-1 autoencoder loss
```
L_{\text{Autoencoder}} =  \min_{\encoder, \decoder} \max_\psi \Big( \lrec(x, \decoder(\encoder(x))) - \ladv(\decoder(\encoder(x))) + \log \disc(x) + \lreg(x; \encoder, \decoder) \Big)
```
- L_rec: reconstruction loss (perceptual, LPIPS-based per Sec 3.1).
- L_adv, log D_psi(x): adversarial terms with a patch-based discriminator D_psi that tries to tell real images from reconstructions.
- L_reg: the KL or VQ regularizer on the latent.
- min over E, D / max over psi: autoencoder and discriminator play the usual GAN game.

## Key hyperparameters

- Downsampling factors studied: f in {1, 2, 4, 8, 16, 32} (LDM-1 = pixel space) (Sec 4.1).
- Latent (z) shapes for 256x256 inputs (Table 13): LDM-1 256x256x3, LDM-2 128x128x2, LDM-4 64x64x3, LDM-8 32x32x4, LDM-16 16x16x8, LDM-32 "88x8x32" as printed (surely 8x8x32, a typo in `ms_tables_supp.tex`).
- The "64x64x4 latent for 512x512 images" figure is Stable Diffusion's configuration; it is NOT stated in this paper. The paper's text-to-image model is f=8, z 32x32x4 at 256x256 (Table 15).
- Text-to-image model (Table 15): LAION-400M, f=8, KL-reg, 1.45B params, 320 channels, channel mult 1,2,4,4, 8 heads, cross-attention at 32,16,8, embedding dim 1280, batch 680, 390K iterations, lr 1e-4. tau_theta is a transformer on BERT-tokenized text (Sec 4.3.1).
- Class-conditional ImageNet LDM-4 (Table 15): f=4, z 64x64x3, 395M, batch 1200, 178K iterations.
- 1000 diffusion steps, linear schedule, for all models (Tables 13, 15).
- KL weight ~1e-6 (App. G).
- Classifier-free guidance scale s = 1.5 for LDM-KL-8-G and LDM-4-G (Tables 2, 3).

## Headline results

1. f sweep (Sec 4.1, Fig 6, figure only): LDM-{1,2} train slowly; LDM-32 stagnates early from information loss; "LDM-{4-16} strike a good balance"; FID gap of 38 between LDM-1 and LDM-8 after 2M steps. Speed/quality (Fig 7): "LDM-4 and -8 offer the best conditions". Autoencoder reconstruction (Table 8, R-FID on ImageNet-Val, VQ): f=2 0.16, f=4 0.58, f=8 1.14, f=16 5.15, f=32 31.83.
2. Class-conditional ImageNet 256 (Table 3): LDM-4-G FID 3.60, IS 247.67, 400M params vs ADM-G 4.59 / 186.7 / 608M and BigGAN-deep 6.95. Unguided LDM-4 FID 10.56 vs ADM 10.94.
3. Compute (Table 18, V100-days): LDM-4 271 vs ADM 916 (ADM-G 962 incl. classifier); LDM-8 79. Throughput (samples/sec, A100): LDM-4-G 0.4 vs ADM-G 250 steps 0.07.
4. Text-to-image, MS-COCO 256 (Table 2): LDM-KL-8 FID 23.31, IS 20.03; with CFG s=1.5 (LDM-KL-8-G) FID 12.63, IS 30.29, at 1.45B params vs GLIDE 12.24 (6B) and Make-A-Scene 11.84 (4B).
5. Unconditional (Table 1): CelebA-HQ 256 FID 5.11 (new SOTA at the time, LDM-4, 500 DDIM steps); LSUN-Bedrooms 2.95 (vs ADM 1.90, with about half the parameters and 4x less training compute per Sec 4.2).
6. Inpainting (Sec 4.5): at least 2.7x speed-up between pixel- and latent-based diffusion while improving FID by at least 1.6x.

## Limitations and later fixes

- Stated (Sec 5): sequential sampling still slower than GANs; the autoencoder's reconstruction limits tasks needing pixel-exact accuracy (they suspect their super-resolution models are somewhat limited by this).
- Stated societal: deepfakes, training-data leakage, bias.
- Observed in-source: VQ-reg latents sometimes give better samples despite worse reconstructions (Sec 4); KL latents need variance rescaling or convolutional sampling at higher resolution suffers (App. D.1).
- Later fixes (external, not verified here): Stable Diffusion (f=8 KL VAE, 64x64x4 latent at 512x512, CLIP text encoder, LAION-5B); SDXL (improved autoencoder, larger UNet); DiT (transformer backbone on latents); distillation and consistency models for fewer steps.

## Surprising or worth flagging

- The v2 headline numbers depend on classifier-free guidance (the CFG paper), s = 1.5. The paper does not define s relative to CFG's w.
- App. C (Image Guiding Mechanisms, Eq. 16) writes the classifier-guidance update with a plus sign: `\hat{\epsilon} \leftarrow \model(z_t, t) + \sqrt{1-\alpha_t^2}\; \nabla_{z_t} \log p_{\Phi}(y\vert z_t)`, whereas ADM's Eq. 14 has a minus sign. Likely a sign typo in LDM; ADM's form is the correct one for eps-prediction.
- Table 3 labels ADM and ADM-G as "250 DDIM steps"; ADM's own paper uses 250 ancestral (non-DDIM) steps for those numbers.
- The cross-attention W dimension typo (above) and the "88x8x32" z-shape typo.
