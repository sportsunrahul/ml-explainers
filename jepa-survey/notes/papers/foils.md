# Foils for ch-ssl (PDF only: abstract + method + one key equation each)

Sources: `sources/<name>/paper.pdf` (text via pdftotext) and `abs.html` (dates). "Published" = v1 date on
abs.html; the version read is the PDF in `sources/` (stated per paper). Equation numbers are as printed in
the PDF; equations are transcribed from the PDF (no LaTeX source), so TeX is ours. Venues are given only
when the PDF or abs.html states them.

## MAE (He et al.) -- Masked Autoencoders Are Scalable Vision Learners
- arXiv 2111.06377v3 (read; v3 dated 2021-12-19). Published v1: 2021-11-11. Authors: Kaiming He ...
  Ross Girshick (6). abs.html: "Tech report". Venue not stated in the PDF.
- Idea: mask random patches of an image and reconstruct the missing pixels with an asymmetric
  encoder-decoder (abstract).
- Encoder sees only visible patches (Sec 3 "MAE encoder", p.3): "Our encoder is a ViT but applied only on
  visible, unmasked patches ... our encoder only operates on a small subset (e.g., 25%) of the full set.
  Masked patches are removed; no mask tokens are used." Mask tokens are added only at the decoder input.
- Mask ratio: "masking a high proportion of the input image, e.g., 75%" (abstract); Fig 5 (p.4): 75% is
  good for both linear probing and fine-tuning; Table 1 caption (p.5): default "the masking ratio is 75%".
  (Fig 2 visualisations use 80%: "leaving only 39 out of 196 patches".)
- Loss (Sec 3 "Reconstruction target", p.4; no numbered equations in the paper): "Our loss function
  computes the mean squared error (MSE) between the reconstructed and original images in the pixel space.
  We compute the loss only on masked patches, similar to BERT." Normalised variant: "the normalized pixel
  values of each masked patch. Specifically, we compute the mean and standard deviation of all pixels in a
  patch and use them to normalize this patch. Using normalized pixels as the reconstruction target improves
  representation quality in our experiments."
- Careful: the ablation default is UNnormalised pixels (Table 1 caption, p.5: "the reconstruction target is
  unnormalized pixels"); Table 1d shows normalised pixels improve accuracy; the final models use normalised
  pixels and 1600 epochs (Table 3 caption, p.6: "Here our MAE reconstructs normalized pixels and is
  pre-trained for 1600 epochs").
- Headline: ViT-H "87.8%" top-1 ImageNet-1K with only IN-1K data (abstract); decoder default depth 8,
  width 512 (Table 1 caption).

## BYOL (Grill et al.) -- Bootstrap Your Own Latent
- arXiv 2006.07733v3 (read; v3 dated 2020-09-10). Published v1: 2020-06-13. Authors: Jean-Bastien Grill
  ... Michal Valko (14). Venue not stated in the PDF.
- Idea: an online network predicts the target network's projection of another view of the same image; the
  target is a slow-moving average of the online network; no negative pairs (abstract).
- Sec 3.1 (p.3): online network = encoder f_theta, projector g_theta, predictor q_theta; target has the same
  architecture with weights xi.
- EMA target update, Eq (1):
  `\xi \leftarrow \tau \xi + (1 - \tau)\theta`  ("given a target decay rate tau in [0,1], after each
  training step").
- Predictor: "We then output a prediction q_theta(z_theta) of z'_xi and l2-normalize both"; "this predictor
  is only applied to the online branch, making the architecture asymmetric". Architecture (Sec 3.3, p.5):
  "The predictor q_theta uses the same architecture as g_theta" = linear 4096 -> BN -> ReLU -> linear 256.
- Loss, Eq (2):
  `\mathcal{L}_{\theta,\xi} \triangleq \| \bar{q}_\theta(z_\theta) - \bar{z}'_\xi \|_2^2 = 2 - 2 \cdot \frac{\langle q_\theta(z_\theta), z'_\xi \rangle}{\|q_\theta(z_\theta)\|_2 \cdot \|z'_\xi\|_2}`
  symmetrised by swapping views; minimised "with respect to theta only, but not xi" (stop-gradient, Fig 2).
- Dynamics, Eq (3): `\theta \leftarrow \mathrm{optimizer}(\theta, \nabla_\theta \mathcal{L}^{\mathrm{BYOL}}_{\theta,\xi}, \eta)`, then Eq (1).
- tau schedule (Sec 3.3, p.5): tau_base = 0.996 increased to 1 with
  `\tau \triangleq 1 - (1 - \tau_{\mathrm{base}}) \cdot (\cos(\pi k / K) + 1)/2`.
- Headline: 74.3% top-1 ImageNet linear eval with ResNet-50; 79.6% with a larger ResNet (abstract).
  Table 5a (p.8): tau = 0 destabilises; tau = 1 (never updated target) is stable but stops improving.

## SimSiam (Chen & He) -- Exploring Simple Siamese Representation Learning
- arXiv 2011.10566v1 (read; only version). Published v1: 2020-11-20. Authors: Xinlei Chen, Kaiming He (2).
  abs.html: "Technical report, 10 pages". Venue not stated.
- Idea: a Siamese network with a predictor on one side and stop-gradient on the other learns without
  negative pairs, large batches or a momentum encoder; "a stop-gradient operation plays an essential role in
  preventing collapsing" (abstract).
- Sec 3 (p.2-3): shared encoder f (backbone + projection MLP); prediction MLP h; p_1 = h(f(x_1)),
  z_2 = f(x_2). Negative cosine similarity, Eq (1):
  `\mathcal{D}(p_1, z_2) = - \frac{p_1}{\|p_1\|_2} \cdot \frac{z_2}{\|z_2\|_2}`
- Symmetrised loss, Eq (2): `\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, z_2) + \frac{1}{2}\mathcal{D}(p_2, z_1)`.
- Stop-gradient, Eq (3): `\mathcal{D}(p_1, \mathrm{stopgrad}(z_2))` ("z_2 is treated as a constant in
  this term"); full form, Eq (4):
  `\mathcal{L} = \frac{1}{2}\mathcal{D}(p_1, \mathrm{stopgrad}(z_2)) + \frac{1}{2}\mathcal{D}(p_2, \mathrm{stopgrad}(z_1))`.
  "Here the encoder on x_2 receives no gradient from z_2 in the first term, but it receives gradients from
  p_2 in the second term (and vice versa for x_1)."
- Evidence (Fig 2 and its table, p.3): without stop-gradient the loss hits the minimum -1 immediately and
  output std goes to 0 (collapse); ImageNet linear eval 67.7 (+-0.1, 5 trials) with stop-grad vs 0.1
  without. Removing the predictor h also collapses (Table 1a, Sec 4.2, p.5).
- Hypothesis (Sec 5, p.6): SimSiam is an EM-like alternation over two sets of variables (Eq (5)).

## VICReg (Bardes, Ponce, LeCun) -- Variance-Invariance-Covariance Regularization
- arXiv 2105.04906v3 (read; v3 dated 2022-01-28). Published v1: 2021-05-11. Authors: Adrien Bardes ...
  Yann LeCun (3). Venue: "Published as a conference paper at ICLR 2022" (PDF header).
- Idea: avoid collapse with two explicit regularisers on each branch's embedding, a variance term and a
  covariance term, plus an invariance term between branches; no weight sharing, BN, stop-gradient, memory
  bank or negatives required (abstract).
- Sec 4.1 (p.4-5), batch Z = [z_1..z_n] of d-dim embeddings from the expander h_phi; z^j = dimension j
  across the batch.
- Variance term, Eq (1): `v(Z) = \frac{1}{d} \sum_{j=1}^{d} \max(0, \gamma - S(z^j, \epsilon))`,
  Eq (2): `S(x, \epsilon) = \sqrt{\mathrm{Var}(x) + \epsilon}`; gamma = 1, epsilon = 0.0001 (Sec 4.2).
  "Using the standard deviation and not directly the variance is crucial" (gradient vanishes otherwise).
- Covariance, Eq (3): `C(Z) = \frac{1}{n-1} \sum_{i=1}^{n} (z_i - \bar{z})(z_i - \bar{z})^T`;
  Eq (4): `c(Z) = \frac{1}{d} \sum_{i \neq j} [C(Z)]^2_{i,j}` (sum of squared off-diagonals).
- Invariance, Eq (5): `s(Z, Z') = \frac{1}{n} \sum_i \|z_i - z'_i\|_2^2` ("without any normalization").
- Total, Eq (6): `\ell(Z, Z') = \lambda s(Z, Z') + \mu [v(Z) + v(Z')] + \nu [c(Z) + c(Z')]`;
  Eq (7) sums over the dataset. Coefficients (Sec 4.2, p.5): lambda = mu = 25, nu = 1. Expander: 3 layers
  of size 8192 on a ResNet-50 (2048-d representation); expander discarded after pretraining.

## DINO (Caron et al.) -- Emerging Properties in Self-Supervised Vision Transformers
- arXiv 2104.14294v2 (read; v2 dated 2021-05-24). Published v1: 2021-04-29. Authors: Mathilde Caron ...
  Armand Joulin (7). Venue not stated in the PDF.
- Idea: self-distillation with no labels; student matches the momentum teacher's softmax output across
  crops; collapse avoided by "only a centering and sharpening of the momentum teacher outputs" (Sec 3.1,
  p.4).
- Loss, Eq (2): `\min_{\theta_s} H(P_t(x), P_s(x))`, H(a, b) = -a log b; multi-crop form Eq (3) sums over
  the 2 global views through the teacher and all views x' != x through the student. Teacher = EMA of the
  student (momentum 0.996 -> 1, cosine).
- Centering + sharpening, in one line (Sec 3.1, p.4): centering adds a bias c to the teacher output,
  `g_t(x) \leftarrow g_t(x) + c`, with c an EMA of the batch mean, Eq (4):
  `c \leftarrow m c + (1 - m) \frac{1}{B} \sum_{i=1}^{B} g_{\theta_t}(x_i)`;
  sharpening = "using a low value for the temperature tau_t in the teacher softmax normalization".
  Centering "prevents one dimension to dominate but encourages collapse to the uniform distribution, while
  the sharpening has the opposite effect"; both together suffice (Sec 3.1; Sec 5.3, p.9).
- Temperatures (Sec 3.2, p.5): tau_s = 0.1; tau_t linearly warmed up from 0.04 to 0.07 over 30 epochs.
  Appendix: tau_t above 0.06 collapses.
- Headline: 78.3% top-1 ImageNet k-NN with a small ViT; 80.1% linear with ViT-B (abstract).

## data2vec (Baevski et al.) -- A General Framework for Self-supervised Learning in Speech, Vision and Language
- arXiv 2202.03555v3 (read; v3 dated 2022-10-25). Published v1: 2022-02-07. Authors: Alexei Baevski ...
  Michael Auli (6). Venue not stated in the PDF.
- Idea: student sees a masked view and regresses contextualised latent targets produced by an EMA teacher
  on the unmasked input; same objective across speech, vision, text (abstract). "Instead of predicting
  modality-specific targets such as words, visual tokens or units of human speech which are local in
  nature, data2vec predicts contextualized latent representations that contain information from the entire
  input."
- Teacher (Sec 3.3, p.3): `\Delta \leftarrow \tau \Delta + (1 - \tau)\theta` (unnumbered); tau linearly
  increased from tau_0 to tau_e over tau_n updates.
- Targets (Sec 3.3 "Targets", p.3): "constructed based on the output of the top K blocks of the teacher
  network for time-steps which are masked in student-mode"; per-block normalisation then averaging,
  `y_t = \frac{1}{K} \sum_{l=L-K+1}^{L} \hat{a}^l_t` (unnumbered) for a network with L blocks. Footnote 5:
  "We generally use the output of the FFN prior to the last residual connection in each block as target."
  Normalisation: instance norm for speech, parameter-less layer norm for NLP and vision; "Normalizing the
  targets helps prevent the model from collapsing into a constant representation".
- K values: vision ViT-B/ViT-L "beta = 2, K = 6 and tau = 0.9998" (Sec 4.1, p.4); speech "K = 8" (Sec 4.2,
  p.4); NLP "K = 10 and set beta = 4" (Sec 4.3, p.5). Fig 2 (p.7): averaging K > 1 layers beats K = 1.
- Loss (Sec 3.4, p.3, unnumbered): Smooth L1,
  `L(y_t, f_t(x)) = \frac{1}{2}(y_t - f_t(x))^2 / \beta` if `|y_t - f_t(x)| \le \beta`, else
  `(|y_t - f_t(x)| - \frac{1}{2}\beta)`. (Speech uses plain L2, Sec 4.2.)
- Vision masking (Sec 4.1): BEiT-style blockwise masking of 60% of the 196 patches (224x224, 16x16 patches);
  "the same modified image both in teacher mode and student mode" is stated for augmentation (teacher sees
  the unmasked version per Fig 1).

## MSN (Assran et al.) -- Masked Siamese Networks for Label-Efficient Learning
- arXiv 2204.07141v1 (read; only version). Published v1: 2022-04-14. Authors: Mahmoud Assran ...
  Nicolas Ballas (9). Venue not stated in the PDF.
- Idea: match the representation of a randomly masked anchor view to that of the unmasked target view via
  soft assignments over learnable prototypes; only unmasked patches are processed (abstract). "MSN
  architectures do not directly predict pixel values (or tokens) for the masked patches. Instead, the loss is
  applied directly to the output corresponding to the [CLS] token" (Sec 3, p.5).
- Mask (Sec 3, p.5): random masking drops "potentially non-contiguous patches across the sequence"; focal
  masking keeps one local block and drops everything around it (Fig 4). Target view is never masked. Target
  encoder = EMA of the anchor encoder (momentum 0.996 -> 1.0).
- Prototype matching (Sec 3, p.5), q in R^{K x d} with K > 1 learnable prototypes; for L2-normalised z and q:
  `p_{i,m} := \mathrm{softmax}\left( \frac{z_{i,m} \cdot q}{\tau} \right)` (unnumbered), target prediction
  p^+_i uses the same q with tau^+ < tau ("sharper target predictions").
- Loss, Eq (1): `\frac{1}{MB} \sum_{i=1}^{B} \sum_{m=1}^{M} H(p^+_i, p_{i,m}) - \lambda H(\bar{p})`,
  with `\bar{p} := \frac{1}{MB} \sum_i \sum_m p_{i,m}` (mean-entropy maximisation, ME-MAX); gradients only
  through the anchor predictions.
- Defaults (Appendix A, p.14-15): 1 random-masked 224x224 anchor view + 10 focal 96x96 views; random
  masking ratio 0.15 (ViT-S/16), 0.3 (ViT-B/16, ViT-B/8), 0.7 (ViT-L/7, ViT-B/4); tau = 0.1, tau^+ = 0.025;
  lambda = 1.0; 1024 prototypes of dimension 256; Sinkhorn on targets by default.
- Headline: ImageNet-1K 72.4% top-1 with 5,000 labels; 75.7% with 1% of labels (abstract).

## Cross-foil notes for the outline
- Dates for the timeline (v1): BYOL 2020-06-13, SimSiam 2020-11-20, DINO 2021-04-29, VICReg 2021-05-11,
  MAE 2021-11-11, data2vec 2022-02-07, MSN 2022-04-14.
- SimCLR is named in ch-ssl but there is no `sources/simclr/`; keep it as a name only or add a source.
- "data2vec ... the closest predecessor": true for per-token latent targets from an EMA teacher (K = 6
  layers averaged, not one layer). MSN is equally close on the other axis: a masked anchor view in a
  joint-embedding network with an EMA target, same first author as I-JEPA. Mention both.
- MAE's loss location: "pixels, normalised" is right for the final models, but the Table 1 ablation default
  is unnormalised; say "MSE on pixels, masked patches only, per-patch normalised in the final models".
