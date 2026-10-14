# ebm-tutorial: Dawid & LeCun, Introduction to Latent Variable Energy-Based Models

Lens: theory. Source of truth: `sources/ebm-tutorial/src/*.tex` (lecture1/2/3.tex, appendixA.tex,
SciPostPhysLectNotes_Template.tex) and `paper.pdf` for page, figure and equation numbers.

## 1. Identity
- Title: Introduction to Latent Variable Energy-Based Models: A Path Towards Autonomous Machine Intelligence
- arXiv: 2306.02572v1 (only version). Published (abs.html, v1): 2023-06-05.
- Authors: Anna Dawid (first), Yann LeCun (last); 2 authors.
- Venue: SciPost Phys. Lect. Notes template ("Les Houches Summer School Lecture Notes 2022"); abs.html
  journal ref: J. Stat. Mech. (2024) 104011. abs.html comments: "23 pages + 1-page appendix, 11 figures".
- Content follows three lectures by LeCun, July 2022, Les Houches Summer School on Statistical Physics and
  Machine Learning (Sec 1, p.2; Acknowledgements).
- Repo: none.
- Custom macros used in quoted TeX (`src/our_commands.tex`): `\params` = `\bm{w}`, `\param` = `w`,
  `\diff` = `\mathop{}\!\mathrm{d}`, `\enc` = `\operatorname{Enc}`, `\stress{}` = italics,
  `\highlight{}` = blue box, `\acp{}`/`\ac{}` = acronym package.

## 2. Core problem
Probabilistic models need a normalised distribution over outputs, which is intractable for high-dimensional
continuous data such as video (Sec 2.4, p.7; Sec 3, p.8). SSL on images trained to make a single prediction
"makes them predict the average of all possible predictions. As a result, SSL produces blurry predictions"
(Sec 2.4, p.7). The notes propose EBMs plus latent variables, assembled into JEPA / H-JEPA, as the
building block of LeCun's autonomous-AI architecture (Ref. [6]).

## 3. What prior methods did and why that hurt (per the paper)
- Supervised and RL need too many labels or trials (Sec 2.2).
- Masked / denoising AEs work for text but "are less successful with images ... The reason ... is probably
  the multimodality of predictions. There are many ways of filling in the missing parts of the image or
  video, but the model is allowed to output only one" (Sec 5.3, p.20; cites He et al. MAE as a promising
  combination with transformers).
- Contrastive training "scale[s] badly with data dimension" (Sec 4.1 highlight, p.14); the need to generate
  contrastive samples gives "bad (even exponential!) scaling" (p.15).
- Maximum likelihood is itself a contrastive method and wants an "infinitely deep and infinitely narrow
  canyon" on the data manifold, so it needs regularisation (Sec 4.1.2, p.15).

## 4. The one key idea
Replace "predict y from x" with a scalar compatibility energy F(x, y); inference is argmin over y; a latent z
absorbs what y contains that x does not; training sculpts the landscape so data is low and everything else
is high, either by pushing up chosen points (contrastive) or by limiting the volume of the low-energy region
(regularised). A JEPA applies this at the level of learned representations s_x, s_y rather than in data space.

## 5. Mechanism, step by step
1. Energy: F(x, y) scalar; low for compatible pairs; inference y_check = argmin_y F(x, y) (Sec 3, p.9).
   The energy is used for inference only; training uses a separate loss (highlight, p.9).
2. Latent variable: E(x, y, z); F(x, y) obtained by minimising or marginalising over z (Sec 3.2, p.11;
   Fig 4(a), p.12). "The information capacity of the latent variable must be minimized. Otherwise, the
   training may put all the information needed for the prediction into them" (end of Sec 3.2, p.12).
3. Training principle: "sculpt the energy landscape such that data points that are probable or compatible
   have low energy, while unlikely data points have high energy. There are two main approaches:
   contrastive methods and regularized and architectural methods" (Sec 4 highlight, p.12).
4. Collapse: "given an x the energy landscape could become 'flat', giving essentially the same energy to all
   values of y" (Sec 4, p.13). Fig 5(a) deterministic regression architecture, "no collapse"; Fig 5(b)
   joint-embedding architecture, "can collapse!": two encoders minimising D(s_x, s_y) can "ignore inputs
   entirely and just produce identical constant outputs" (p.13). "Every model that can be multimodal ...
   is susceptible to collapse" (p.13).
5. Contrastive methods: Eq (3) pairwise hinge loss; group losses like InfoNCE (Sec 4.1, p.14). Fig 6(b).
6. Regularised and architectural methods: bound the low-energy volume by construction (PCA, k-means, GMM,
   flows) or add a term that measures that volume; for latent-variable EBMs, limit the information capacity
   of z (discrete, sparse, stochastic, low-dimensional) (Sec 4.2, p.16). Fig 6(c).
7. JEPA: Enc_x and Enc_y (may differ, no shared parameters, different modalities ok); predictor
   Pred(s_x, z); energy = distance between predicted and actual s_y (Sec 6.1, p.20; Fig 9). Trained with a
   regularised loss (Fig 10, p.21): prediction error + R(z) + terms that maximise the information content
   of s_x about x and s_y about y; VICReg given as the example of the information term (p.21).
8. H-JEPA: stacked JEPAs; JEPA-1 short-term predictions from low-level representations; JEPA-2 longer-term
   predictions from higher-level representations; CNN/pooling layers between levels coarse-grain (Sec 6.2,
   p.22; Fig 11). Motivation: "which information is important may be scale-dependent" (p.22).

## 6. Equations
Numbered equations in the PDF: (1)-(12) in the body, (A1)-(A5) in Appendix A. The JEPA energy and the
latent-variable energy are NOT numbered equations; locations given exactly below.

E1. EBM definition. Sec 3, p.9, highlight box (unnumbered). TeX (lecture1.tex):
```
\Acfp{EBM} capture dependencies between variables $x$ and $y$ by associating a scalar energy $F(x,y)$ to each configuration of the variables \cite{LeCun2006EBMtutorial, Huembeli2022EBMs}. Low energies should be assigned to compatible pairs of $x$ and $y$ (e.g., slow driving when snow is present) and high energies to incompatible pairs. The inference consists then in finding values of $y$ that minimize $F_{\params}(x,y)$: $\check{y} = \mathrm{argmin}_y F_{\params}(x,y)$.
```
Gloss: x = condition / context; y = candidate answer; F = scalar score, low = compatible; w (\params) =
trainable weights; y_check = the inferred answer.
Companion statement (p.9, highlight): "the energy function is not the objective function to minimize within
the learning! The energy is used only for inference."

E2. Gibbs-Boltzmann link. Eq (2), Sec 3.1, p.10, label `eq:Gibbs`:
```
P(y \mid x) = \frac{\exp{\left[-\beta F(x,y)\right]}}{\int \diff y' \exp{\left[- \beta F(x,y')\right]}}\,,
```
Paper's wording: denominator is the partition function; beta is the inverse temperature; computing the
partition function "is very challenging, so probabilistic models in continuous domains in high dimensions
are intractable".

E3. Latent-variable energy. Sec 3.2, p.11, inline (unnumbered). TeX verbatim:
```
$F_{\params} (x,y) = \mathrm{argmin}_{z} E_{\params} (x,y,z)$ or $F_{\params} (x,y) = - \frac{1}{\beta} \log \left[ \int \diff z' \exp{\left[-\beta E_{\params} (x,y,z')\right]} \right]$
```
Paper's wording: inference "involves additionally minimizing or marginalizing with respect to these unseen
variables z"; "The inference then is simply $\check{y} = \mathrm{argmin}_{y} F_{\params}(x,y)$."
Our reading: the text writes `argmin_z`, which returns a z, not an energy; the intended form is `min_z`
(Fig 4(a) label reads "minimization or marginalization with respect to z"; Sec 4 opening, p.12, writes
`$\check{y} = \mathrm{argmin}_{y,z} E(x,y,z)$`). The marginalised form is restated in Appendix A, p.25,
just before Eq (A5):
```
F_{\params}(x, y)=-\frac{1}{\beta} \log \int_{z \in \mathcal{Z}} \exp{\left[-\beta E_{\params}(x,y,z)\right]}
```
called "the free energy of the ensemble". Footnote (Sec 3.2): marginalisation "becomes expensive in high
dimensions and is usually replaced with a cheaper minimization instead".
Gloss: z = latent, "capturing the information in y that is not readily available in x" (p.11); E = energy
with z exposed; F = energy after z is resolved.
Worked example (p.11): ellipse, `$E_{r_1, r_2} (y, z) = (y_1 - r_1 \sin{z})^2 + (y_2 - r_2 \cos{z})^2$`,
z = angle; F = squared distance to the ellipse (Fig 4(b)).

E4. Two architectures that can / cannot collapse. Fig 5, p.13 (labels transcribed from
`src/images/EBM-collapse_v2.pdf`; ASCII TeX ours):
- (a) "deterministic architecture (no collapse)": `F_w(x,y) = \| y - \bar{y} \|_{L1}`, with
  `\bar{y}` = neural-network prediction from x. Caption: "Standard deterministic architecture for prediction
  or regression ... is immune to collapse."
- (b) "joint embedding architecture (can collapse!)": `F_{w_1 w_2}(x,y) = D(s_x, s_y)`, `s_x`, `s_y` from
  two networks with parameters w_1, w_2.

E5. Contrastive hinge loss. Eq (3), Sec 4.1, p.14, label `eq:contrastive-hinge-loss`:
```
L_{\rm hinge} (x, y, \hat{y}, \params) = [F_{\params}(x,y) - F_{\params}(x, \hat y) + m(y, \hat y)]^+\,.
```
Gloss: (x, y) training pair (blue dots, Fig 6(b)); y_hat contrastive point (green dots) whose energy must
rise; m = margin depending on the distance between y and y_hat; [.]^+ = hinge. "Suitable contrastive loss
functions need to ensure a non-zero margin to avoid energy collapse."

E6. NLL as a contrastive loss. Eq (4), Sec 4.1.2, p.14, label `eq:loglikelihood_EBM`:
```
L_{\rm NLL} (x, y, \params)=-\frac{1}{\beta} \log P_{\params}(y \mid x)=F_{\params}(x, y)+\frac{1}{\beta} \log \left[\int \diff y^{\prime} \exp{\left[ -\beta F_{\params}\left(x, y^{\prime}\right)\right]}\right]\,,
```
Paper's wording: first term lowers the energy of training points; second term "increases the energy of all
the points in the available space". "Negative log-likelihood is then a contrastive method!" (p.15).

E7. Regularised autoencoder objective. Sec 4.2.1, p.16, `align*` (unnumbered):
```
L_{\rm reg}(y, \params) = D(y, \text{Dec}(\text{Enc}(y))) + R(\text{Enc}(y))\,,
```
D = distance between input and output; R = regulariser on the code. Instances listed: PCA, bottleneck AE,
k-means, Gaussian mixtures, sparse coding (p.16).

E8. JEPA energy. NOT a numbered equation. Appears only as labels inside Fig 9, p.20
(`src/images/JEPA_v3.pdf`); transcription (ASCII TeX ours):
```
F_{w_1 w_2 w_3}(x, y, z) = D(\tilde{s}_x, s_y),\quad \tilde{s}_x = \mathrm{Pred}_{w_3}(s_x, z),\quad s_x = \mathrm{Enc}_{w_1}(x),\quad s_y = \mathrm{Enc}_{w_2}(y)
```
Figure text also reads "comparison of representations", "based on a latent variable z", "e.g., NNs",
"e.g., x - video, y - audio". Body text (lecture3.tex, Sec 6.1): "compares them at the level of learned
internal representations, $s_x$ and $s_y$, where $s_i = \mathrm{Enc}(i)$"; caption: "The aim of JEPA is
to predict $s_y$ out of $s_x$ with help of a latent variable $z$ ... The energy is the distance between
those representations."
Gloss: Enc_{w1}, Enc_{w2} = two encoders (may differ, no shared weights); Pred_{w3} = predictor; z = latent
carrying "relevant information about y that may be missing from x"; D = distance; tilde s_x = predicted
embedding of y.

E9. JEPA regularised training loss. Fig 10, p.21 (labels from `src/images/JEPA-regularization_v3.pdf`):
`D(\tilde{s}_x, s_y)` "minimize the error"; `R(z)` "minimize the information content of a latent variable
z"; `-I(s_x)` "maximize the information content of s_x about x"; `-I(s_y)` "maximize the information
content of s_y about y". Caption: "Adapted from Ref. [6]." Body (p.21): "to prevent the informational
energy collapse, we need to make sure that $s_x$ and $s_y$ carry as much information as possible about $x$
and $y$. Otherwise, the training could lead encoders to be, e.g., constant."
VICReg as the worked example (p.21): minimise `$\mathrm{Cov}(v_i, v_j)\mid_{i \neq j}$` over a batch; keep
variance up via hinge `$[1-\sqrt{\mathrm{Var}(v_i)}]^+$`; expander to kill non-linear dependencies.
Also named as related regularisation routes: BYOL, SimSiam, Barlow Twins (p.21).

E10. H-JEPA. Fig 11, p.22 (`src/images/H-JEPA_v3.pdf`): JEPA-1 encodes x_0, x_1, x_2 and scores
`D(\tilde{s}_{x_0}, s_{x_1})` with `\tilde{s}_{x_0} = \mathrm{Pred}(s_{x_0}, z)`; JEPA-2 sits on top of
JEPA-1's representations and scores `D(\tilde{s}'_{x_0}, s'_{x_2})` with its own Pred and z', i.e. a
longer horizon (x_0 -> x_2) at a coarser level. Caption: "lower-level JEPA-1 is used for short-time
predictions based on many details, and a higher-level JEPA-2 makes predictions for the longer term based on
fewer details. Adapted from Ref. [6]."

## 7. Setup (where things are in the PDF)
| Item | Location |
|---|---|
| Fig 1 modular AI; Fig 2 SSL cake | p.6, p.7 |
| Fig 3 towards EBMs (replace divergence C(y, y_bar) with constraints F(x,y)) | p.9 |
| Table 1 examples of latent variables | p.11 |
| Fig 4 latent-variable EBMs; ellipse example | p.12 |
| Fig 5 EBMs can collapse (deterministic vs joint embedding) | p.13 |
| Fig 6 training: (a) proper, (b) contrastive, (c) regularised; "Adapted from Ref. [6]" | p.14 |
| Sec 4.1 contrastive highlight; Eq (3) | p.14 |
| Sec 4.2 "Architectural and regularized methods" highlight | p.16 |
| Fig 7 Hopfield/Boltzmann; Fig 8 denoising and masked AE | p.18, p.19 |
| Sec 5.3 masked AEs less successful for images (multimodality) | p.20 |
| Fig 9 JEPA; Sec 6.1 | p.20 |
| Fig 10 JEPA regularised loss | p.21 |
| Fig 11 H-JEPA; Sec 6.2 | p.22 |
| Appendix A, Eqs (A1)-(A5) | p.25 |
| Ref. [6] position paper | p.26 |

## 8. Results worth remembering (statements, with location)
- Collapse-avoidance families, in their words:
  - Contrastive (Sec 4.1 highlight, p.14): "Contrastive methods aim to push down the energy of training
    samples and to pull up the energy of suitably-generated contrastive samples that by definition are out
    of the training set. They work for any architecture but scale badly with data dimension."
  - Regularised (Sec 4.2 highlight, p.16): "Another option to prevent collapse is to use regularized
    methods. These methods aim to minimize the volume of space that the low-energy points can take. When
    the energy of input data is lowered, it is required to increase the energy of some parts of the space.
    Because regularized methods are less likely than contrastive methods to fall victim to the curse of
    dimensionality, they seem more promising to train EBM of the future."
  - Fig 6 caption (p.14): "(b) Contrastive methods minimize the energy of training samples while increasing
    the energy of samples outside the training set. (c) Regularized methods limit the volume of space that
    the low-energy region can take."
  - Sec 4 highlight (p.12): "To prevent the energy collapse, i.e., flat energy landscape, we need to
    constrain our energy surface so that data points have low energies but also that points outside the
    regions of high data density have higher energies."
- How the position paper is cited: Ref. [6] (p.26), bib key `LeCunnPathTowardsAI`:
  `Y. LeCun, A path towards autonomous machine intelligence, Version 0.9.2, 2022-06-27 (2022),
  https://openreview.net/forum?id=BZ5a1r-kVsf`. The .bib has the title in Title Case
  ("A Path Towards Autonomous Machine Intelligence"); the rendered .bbl lowercases it. Sec 1 (p.2) names
  it as "his paper 'A Path Towards Autonomous Machine Intelligence' in 2022". OpenReview id: BZ5a1r-kVsf.
  Figs 6, 10, 11 are "Adapted from Ref. [6]"; Sec 4.1.1 points to its "appendix 8.3.3" for contrastive
  losses.
- Why JEPA can discard detail (Sec 6.1, p.20-21): encoders "can learn to filter irrelevant information
  about x and y" (Fig 9 caption); encoders "can have invariance properties and, e.g., map various y's onto
  the same s_y"; z infers "the information necessary to predict s_y that is not present in s_x".
  Conclusion (p.23): JEPAs "make predictions in the representation space, allowing encoders to remove
  irrelevant data features for the task at hand".
- Sec 2.4 (p.7-8): SSL "predict hidden parts of the input (in orange) from visible parts of the input (in
  blue)" (Fig 2 caption); SSL = cake, SL = icing (10-10,000 bits/sample), RL = cherry (a few bits).
- Sec 3.1 (p.10-11) costs of giving up probabilities: "it is challenging to understand uncertainty in an
  energy-based setting"; "energies are uncalibrated".

## 9. Limitations the paper states
- No experiments; a framework summary. Regularised methods' "main challenge is choosing how to limit the
  volume of the low-energy space" (Sec 4.2, p.16). H-JEPA "may be a starting point" (Conclusion, p.23).
- Not addressed here, and left to the later papers in this survey: a concrete I-JEPA instantiation
  (sources/ijepa, which drops the explicit z) and a principled single regulariser (sources/lejepa). Not
  verified in this note; see their own notes.

## 10. Outline feedback (ch-ebm)
1. "F(x,y) = min_z E(x,y,z)": the tutorial's text literally writes `argmin_z` (Sec 3.2, p.11), a typo.
   Write `min_z`, cite "Sec 3.2 and Fig 4(a)", and do not claim it is the tutorial's exact wording. The
   marginalised (free-energy) form is on p.11 and p.25 (before Eq (A5)).
2. "the JEPA energy D(s_y, Pred(s_x, z))": there is no numbered JEPA equation. The energy exists only as
   Fig 9 labels, written `D(\tilde{s}_x, s_y)` with `\tilde{s}_x = Pred_{w3}(s_x, z)`. Argument order is
   reversed relative to the outline. Cite "Fig. 9" and use the figure's form if the page says "as written".
3. "Three architectures: generative, joint-embedding, JEPA": the tutorial never defines a "generative
   architecture". Fig 5 contrasts "(a) deterministic architecture (no collapse)" with "(b) joint embedding
   architecture (can collapse!)". The word "generative" appears only in "latent variable generative EBM
   architecture" (Sec 4, p.13, pointing at Fig 4(b)) and for GANs. The three-way taxonomy belongs to the
   position paper; attribute it there and cite the tutorial for Figs 4, 5, 9 only.
4. Family names: use "contrastive methods" and "regularized and architectural methods" (Sec 4 highlight;
   Sec 4.2 title "Architectural and regularized methods").
5. Visual buttons map cleanly: "collapsed" = Sec 4 flat-landscape sentence (p.13); "contrastive" = Fig 6(b);
   "regularised" = Fig 6(c). Consider a fourth state from Fig 10 for JEPA specifically: error + R(z) +
   information terms, with VICReg named as the example.
6. Dates: tutorial v1 2023-06-05; position paper "Version 0.9.2, 2022-06-27". The outline's "LeCun 2022"
   chapter title is consistent.
