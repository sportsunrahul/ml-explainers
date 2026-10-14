# nagel2021 -- A White Paper on Neural Network Quantization (SPINE)

## 1. Identity
- Title: "A White Paper on Neural Network Quantization".
- arXiv 2106.08295v1 (only version). Submitted 2021-06-15 (abs.html).
- Authors: Markus Nagel (first, equal contribution with Marios Fournarakis) ... Tijmen Blankevoort (last). Qualcomm AI Research.
- Venue: none stated (white paper; NeurIPS 2020 style file with [final] option, no venue claim). Repo: none given.
- Source read: src/main.tex + sections/*.tex + the tables that are actually `\input` (see note in 8). PDF used for page,
  equation, table and figure numbers (27 pages).
- Macro glossary (src/includes_quant.tex): `\mat{X}`, `\vec{x}` = bold; `\mati{W}{1}` = W^{(1)}; `\veci{b}{1}` = b^{(1)};
  `\matq{W}` = hat W; `\matsq{W}` = tilde W; `\ct{x}` = x (identity); `\round*{x}` = round-to-nearest bracket;
  `\floor*{x}`; `\dw` = Delta w; `\dW` = Delta W; `\bngamma, \bnbeta, \bnmu, \bnsigma` = bold upright BN params;
  `\eop{x}` = E[x]; `\tloss{x}` = L(x); `\func{f}{x}` = f(x); `\l`, `\r` = \left, \right; `\clamp` = operator "clamp".

## 2. Core problem
Quantized inference saves energy and latency, but the rounding and clipping noise costs accuracy. The paper gives
practitioners two tested recipes: a push-button post-training pipeline (PTQ, little or no data) and a
quantization-aware training pipeline (QAT) for lower bit-widths, plus a debugging flow when PTQ falls short.

## 3. What prior practice did and why it hurt (per the paper)
- Per-tensor min-max ranges: no clipping but outliers inflate rounding error (Sec. 3.1).
- Per-tensor quantization of depthwise-separable nets after BN folding: channel ranges differ wildly, so INT8 PTQ of
  MobileNetV2 is near random (0.12% top-1, Table 3) (Sec. 3.2).
- Rounding-to-nearest is MSE-optimal per weight but not task-loss-optimal (Sec. 3.4).
- Some QAT work ignores BN folding; that is fine per-channel but wrong for per-tensor deployment (Sec. 4.2).
- Double-forward BN folding in QAT (Jacob 2018) is costly and no better than static folding (Table 7).

## 4. The one key idea
Treat quantization as a hardware-aware noise source and remove its worst parts in a fixed order before any training:
equalize channel ranges (CLE), pick ranges by MSE, choose weight rounding by layer-output reconstruction (AdaRound)
or correct the mean shift (bias correction); only if that is not enough, train through the quantizer with STE.

## 5. Mechanism, step by step
Hardware view (Sec. 2.1, Fig. 1-2): an accelerator is a grid of processing elements C_{n,m} = W_{n,m} x_m feeding
accumulators A_n preloaded with the bias. Quantized: INT8 weights and INT8 inputs, INT32 accumulators, then a
requantization block writes INT8 back to memory (Fig. 2 labels: weights INT8 with scale s_w, inputs INT8 with s_x,
accumulators INT32, "Requantization" -> INT8 -> Memory). Toy array: 16 processing elements, 4 accumulators.

Standard PTQ pipeline (Sec. 3.5, Fig. 8, p16). Blue = required step, turquoise = recommended choice:
1. Pre-trained FP model -> CLE (important for depthwise-separable layers and per-tensor quantization).
2. Add quantizers: symmetric weights, asymmetric activations; per-channel weights if the HW/SW stack supports it.
3. Weight range setting: layer-wise MSE (min-max can be better for per-channel).
4. Decision "Use Data": yes -> AdaRound (calibration set, usually 500-1000 unlabeled images; crucial for 4-bit weights);
   no -> analytic bias correction (needs BN).
5. Activation range setting: MSE on a small calibration set, or BN-based range setting for a fully data-free pipeline.

PTQ debugging flow (Sec. 3.7, Fig. 9, p18). "Error" = FP accuracy minus quantized accuracy, epsilon = acceptable error:
1. Quantized model -> set b_W = b_A = 32 -> error ~ 0? If no -> "Fix Model" (simulation bug) and repeat.
2. Loop X in {Activations, Weights}: quantize all tensors in X -> error < eps? yes -> next X.
3. If no -> "Fix problem": weights side = AdaRound, per-channel, CLE, bias correction; activations side = range setting,
   adjust CLE (vanilla CLE can make activation ranges uneven) -> error < eps? yes -> next X.
4. If still no -> per-layer loop l = 0..L: set b_W = b_A = 32, quantize only tensor Z^(l) -> error < eps? yes -> next layer;
   no -> visualize tensor Z^(l) (per-channel as in Fig. 5, or per-token/per-embedding for BERT) -> fix problem in Z^(l)
   (custom range setting or higher bit-width, e.g. BERT kept some activations at 16 bits) -> error < eps? yes -> next
   layer; no -> QAT.
5. After the loops END -> quantize the whole model -> error < eps? yes -> final quantized model; no -> QAT
   (or higher bit-widths / finer granularity, Sec. 3.7 last paragraph).

QAT (Sec. 4): forward pass identical to simulated quantization (Fig. 4b); backward pass skips the quantizer via STE
(Fig. 10: red backward arrows bypass both the weight quantizer and the activation quantizer). Standard QAT pipeline
(Sec. 4.4, Fig. 12, p24): Pre-trained FP model -> CLE (needed for MobileNet-like nets; optional for others or
per-channel) -> Add quantizers (symmetric weights, asymmetric activations, per-channel weights if supported; model BN
correctly) -> Range estimation (MSE; min-max can be better per-channel) -> Learnable quantization params (lower LR for
them with SGD; or use Adam/RMSProp) -> Train.

BN folding options in QAT (Sec. 4.2, Table 7): (a) static folding into W and b (Eq. 11-12) -- recommended, on par or
better; (b) double forward pass with running-stat correction (Jacob 2018) -- costlier, not better; (c) per-channel only:
keep BN unfolded during training and absorb gamma_k/sqrt(sigma_k^2+eps) into the per-channel scale afterwards (Eq. 41).

QAT initialization (Sec. 4.3): MSE init gives much higher starting accuracy than min-max, but the gap closes after
20 epochs (Table 8, Fig. 11). CLE is required to train MobileNetV2 W4A8 per-tensor at all (Table 9: 0.10 without).

## 6. Equations (TeX verbatim from src; PDF number in brackets)
- [1] `\ct{A}_n =\vec{b}_n + \sum_{m}{\ct{C}_{n,m}}` -- accumulator = bias + sum of PE products (one MAC).
- [2] `\widehat{\vec{x}}  = \ct{s}_{\vec{x}} \cdot \vec{x}_{\text{int}} \approx  \vec{x}` -- a float vector approx. a
  scalar times an integer vector.
- [3] `\hat{\ct{A}}_n = \widehat{\vec{b}}_n +   \ct{s}_{\vec{w}} \ct{s}_{\vec{x}} \sum_{m}{\mat{W}^{\text{int}}_{n,m} \,\vec{x}^{\text{int}}_m}`
  (last line of the align) -- per-tensor scales factor out of the sum, so MACs run in fixed point.
- [4] `\vec{x}_{\text{int}} =\clamp{\left(\round*{\frac{\vec{x}}{\ct{s}}} + \ct{z} ;0, 2^\ct{b}-1 \right)},`
  -- quantize: divide by scale s, round, shift by integer zero-point z, clamp to the unsigned b-bit grid.
- [5] clamp(x; a, c) = a if x < a, x if a <= x <= c, c if x > c.
- [6] `\vec{x}\approx \widehat{\vec{x}}  = \ct{s}\l(\vec{x}_{\text{int}}-\ct{z}\r)` -- de-quantize.
- [7] `\widehat{\vec{x}}= q(\vec{x};\ct{s},\ct{z}, \ct{b}) = \ct{s}\left[\clamp{\left(\round*{\frac{\vec{x}}{\ct{s}}}+\ct{z};0 ,2^\ct{b}-1  \right)}-\ct{z}\right],`
  -- the full quantization function. Grid limits q_min = -s z, q_max = s(2^b - 1 - z); rounding error lies in
  [-s/2, s/2]; values outside [q_min, q_max] incur clipping error. Larger s = less clipping, more rounding error.
- [8a-c] symmetric (z = 0): `\widehat{\vec{x}} &= \ct{s} \, \vec{x}_{\text{int}}`; unsigned
  `\clamp\l(\round*{\frac{\vec{x}}{\ct{s}}};0, 2^\ct{b}-1 \r)`; signed
  `\clamp\l(\round*{\frac{\vec{x}}{\ct{s}}};-2^{\ct{b}-1} , 2^{\ct{b}-1}-1\r)`. Unsigned suits ReLU outputs, signed suits
  roughly symmetric distributions (Fig. 3).
- Power-of-two (Sec. 2.2.2, unnumbered): `\ct{s}=2^{-\ct{k}}` -- scaling becomes a bit shift; less freedom to trade
  rounding vs clipping.
- [11] `\widetilde{\mat{W}}_{k,:} & =  \frac{\bngamma_k \mat{W}_{k,:}}{\sqrt{\bnsigma_k^2 + \epsilon}}`;
  [12] `\widetilde{\vec{b}}_k & = \bnbeta_k - \frac{\bngamma_k \bnmu_k}{\sqrt{\bnsigma_k^2 + \epsilon}}` -- BN folding.
  The per-output-channel factor gamma_k/sqrt(sigma_k^2+eps) is what spreads channel ranges apart.
- [13] asymmetric x asymmetric product: `\ct{s}_{\vec{w}}  \ct{s}_{\vec{x}} \mat{W}_{\text{int}} \mat{x}_{\text{int}}  - \textcolor{red}{\ct{s}_{\vec{w}} \ct{z}_{\vec{w}}  \ct{s}_{\vec{x}} \mat{x}_{\text{int}}} - \textcolor{blue}{\ct{s}_{\vec{w}}  \ct{s}_{\vec{x}} \ct{z}_{\vec{x}} \mat{W}_{\text{int}} + \ct{s}_{\vec{w}} \ct{z}_{\vec{w}} \ct{s}_{\vec{x}} \ct{z}_{\vec{x}}}`
  (line breaks inside `\ct{z}` subscripts removed). Paper: term 1 = symmetric case; terms 3-4 (blue) are known in advance
  and fold into the bias; term 2 (red) depends on input data, "equivalent to adding an extra channel". Hence the common
  choice: asymmetric activations, symmetric weights.
- Per-channel (Sec. 2.4.2, no equation): per-channel weight scales need no accumulator rescaling; per-channel
  activation scales cannot be factored out of the sum. Not all commercial hardware supports per-channel weights.
- [14-15] min-max: `q_{\text{min}} &= \min \mat{V},` `q_{\text{max}} &= \max \mat{V},` -- no clipping; outlier-sensitive.
- [16] MSE: `\argmin_{q_{\text{min}}, q_{\text{max}}} \quad \norm*{\mat{V} - \hat{\mat{V}}(q_{\text{min}}, q_{\text{max}})}^2_F,`
  -- solved by grid search, golden section, or closed-form approximations (Banner).
- [17] cross-entropy (last-layer logits): `\argmin_{q_{\text{min}}, q_{\text{max}}} \quad H\left(\psi\left(\vec{v}\right), \psi\left(\hat{\vec{v}}(q_{\text{min}}, q_{\text{max}})\right)\right)`
  -- H = cross-entropy, psi = softmax, v = logits; keeps the order of the few large logits.
- [18-19] BN-based: `q_{\text{min}} &= \min{(\bnbeta - \alpha \bngamma)}`, `q_{\text{max}} &= \max{( \bnbeta + \alpha \bngamma )}`
  -- per-channel BN shift/scale as mean/std; DFQ uses alpha = 6.
- [20] `f(sx) = sf(x)` -- positive scaling equivariance (ReLU, PReLU; piecewise-linear via reparameterized f-hat).
- [21] `\vec{y} = f(\mati{W}{2} f(\mati{W}{1}\vec{x}+\veci{b}{1}) + \veci{b}{2}) = f(\mati{W}{2} \mat{S} \hat{f}(\mat{S}^{-1}\mati{W}{1} \vec{x} + \mat{S}^{-1} \veci{b}{1}) + \veci{b}{2}) = f(\mati{\widetilde{W}}{2} \hat{f}(\mati{\widetilde{W}}{1} \vec{x} + \veci{\widetilde{b}}{1}) + \veci{b}{2})`
  (three align lines joined) with S = diag(s), W~(2) = W(2) S, W~(1) = S^-1 W(1), b~(1) = S^-1 b(1). Fig. 6: scaling
  channel i of layer 1 by 1/s_i is undone by s_i on input channel i of layer 2.
- [22] CLE: `\vec{s}_i = \frac{1}{\vec{r}_i^{(2)}}\sqrt{\vec{r}_i^{(1)} \vec{r}_i^{(2)}}` -- r_i^(j) = dynamic range of
  channel i in weight tensor j. "They prove that an optimal weight equalization is achieved" by this choice; it makes
  both ranges equal to sqrt(r1 r2). Applied only within residual blocks (skip connections break equivariance).
- [23] bias absorption: b~(2) = W(2) c + b(2), h~ = h - c, b~(1) = b(1) - c.
- [24] `\vec{c}_i = \max\l(0, \min_{\vec{x}} \l(\mat{W}_i^{(1)}\vec{x}+\vec{b}_i^{(1)}\r)\r).` -- data-free estimate
  c_i = max(0, beta_i - 3 gamma_i), which holds for about 99.865% of normal inputs.
- [25-26] bias correction: `\eop{\vec{y}_{\textup{corr}}} &= \eop{\matq{W}\vec{x}} - \dW\eop{\vec{x}} = \eop{\vec{y}}.`
  -- the biased error is Delta W E[x]; subtract it via the bias, no inference overhead. Worst in depthwise layers
  (about 9 weights per output channel for 3x3); main cause is often clipping error.
- [27] empirical: `\dW\eop{\vec{x}} = \eop{\matq{W}\vec{x}} - \eop{\mat{W}\vec{x}}.` (calibration data, layer by layer).
- [28] analytic: `\eop{\vec{x}} = \eop{\text{ReLU}\left(\vec{x}^{\textup{pre}}\right)} = \bngamma \mathcal{N}\left( \frac{-\bnbeta}{\bngamma} \right) + \bnbeta \left[\vec{1}-\Phi\left( \frac{-\bnbeta}{\bngamma} \right)\right]`
  -- clipped-normal mean from the preceding BN; N = std normal PDF, Phi = CDF; element-wise per channel.
- [29] task-loss rounding: `\argmin_{\dw}& \quad \eop{\tloss{\vec{x},\vec{y},\vec{w} + \dw} - \tloss{\vec{x},\vec{y},\vec{w}}}`
  -- each weight may round up or down (binary problem).
- [30] second-order Taylor, converged model (gradient term dropped), layer-block-diagonal Hessian -> QUBO:
  `\argmin_{\dw^{(\ell)}}& \quad \eop{{\dw^{(\ell)}}^T \mati{H}{\veci{w}{\ell}} \dw^{(\ell)}}`. Fig. 7: this cost correlates
  with accuracy over 100 stochastic roundings. Problems: Hessian too costly; QUBO is NP-hard.
- [31] local proxy: `\argmin_{\Delta \mati{W}{\ell}_{k,:}} \quad \eop{\left(\Delta \mati{W}{\ell}_{k,:} \veci{x}{\ell-1}\right)^2}`
  -- MSE of the layer's output, per output row k; no Hessian, no later layers.
- [32] relaxation: `\argmin_{\mat{V}} \quad \norm*{\mat{W} \vec{x} - \matsq{W} \vec{x}}^2_F + \lambda \func{f_{\textup{reg}}}{\mat{V}},`
- [33] soft quantization: `\matsq{W} &= \ct{s} \cdot \clamp{ \left(\floor*{\frac{\mat{W}}{\ct{s}}} + \func{h}{\mat{V}}; \ct{n}, \ct{p} \right)}`
  -- floor plus a learned h(V) in [0,1] (rectified sigmoid); n = q_min/s, p = q_max/s.
- [34] `\func{f_{\textup{reg}}}{\mat{V}} = \sum\limits_{i,j} 1 - |2 \func{h}{\mat{V}_{i,j}} - 1|^\beta,` -- beta annealed
  so h(V) ends at 0 or 1.
- [35] final AdaRound: `\argmin_{\mat{V}} \norm*{ \func{f_{a}}{\mat{W} \vec{x}} - \func{f_{a}}{\matsq{W} \vec{\hat{x}}}}^2_F + \lambda \func{f_{\textup{reg}}}{\mat{V}},`
  -- x-hat = layer input with all preceding layers quantized; f_a = activation function; optimized by SGD.
- [36] STE: `\frac{\partial \round*{\ct{y}}}{\partial \ct{y}} = 1`
- [37] gradient of q w.r.t. input: 1 if q_min <= x_i <= q_max, 0 otherwise (last case block of the align).
- [38] gradient w.r.t. scale: `-\vec{x}_i/\ct{s} + \round*{\vec{x}_i/\ct{s}}` inside the range; n below; p above (LSQ/TQT style).
- [39-40] learnable zero-point: `\widehat{\vec{x}} = q(\vec{x}; \ct{s},\ct{z}) = \ct{s}\cdot \l[ \clamp{\l(\round*{\frac{\vec{x}}{s}} + \round*{z}; n, p \r)} -  \round*{z} \r]`;
  d x-hat / dz = 0 inside the range, -s otherwise.
- [41] per-channel BN absorption: `\widehat{\vec{y}}_k = \widetilde{\ct{s}}_{\vec{w},k} \l(\mat{W}^{\text{int}}_{k,:} \, \vec{x} \r) + \widetilde{\vec{b}}_k`
  with s~_{w,k} = gamma_k s_{w,k} / sqrt(sigma_k^2 + eps) (last line of Eq. 41).

## 7. Measurement setup (analysis lens)
- PTQ results: average of 5 runs; QAT results: average of 3 runs. Validation sets throughout.
- Models/data: ResNet18/50, MobileNetV2, InceptionV3, EfficientNet lite (ImageNet top-1); DeeplabV3 with MobileNetV2
  backbone (Pascal VOC mIoU); EfficientDet-D1 (COCO 2017 mAP); BERT-base (GLUE).
- Notation: W8A8 = 8-bit weights and 8-bit activations. "Per-tensor" / "per-channel" refers to weight quantizer
  granularity; activations are always per-tensor. PTQ pipeline = Fig. 8; QAT pipeline = Fig. 12.
- QAT schedule: DeepLabV3 80 epochs Pascal VOC; EfficientDet 20 epochs COCO 2017; other vision models 20 epochs ImageNet;
  BERT 3-12 epochs per GLUE task; Adam for all; best LR per configuration, no other tuning (Sec. 4.5).
- Calibration size: "Usually, between 500 and 1000 unlabeled images are sufficient" (Sec. 3.5 footnote). Per-experiment
  calibration set sizes are not stated.
- Ablations (Tables 1-5, 7-9) use ResNet18 (FP32 69.68) and MobileNetV2 (FP32 71.72) on ImageNet.
- What would falsify the pipeline claim: a common vision model where W8A8 PTQ with this pipeline loses well over 1 point.
  EfficientDet-D1 per-tensor (-1.79 mAP points) already exceeds the prose "within 0.7%" (see 8).

## 8. Results worth remembering (all values absolute, higher is better)
Only these table files are `\input` into the PDF: 03_range_setting_weights (Table 1), 03_range_setting_activations_asym
(Table 2), 03_CLE (Table 3), 03_bias_correction (Table 4), 03_adaround_ablation (Table 5), 03_standard_ptq (Table 6),
04_bn_ablation (Table 7), 04_quant_init_ablation (Table 8), 04_cle_ablation (Table 9), 04_standard_qat (Table 10).
Not in the PDF: 03_range_setting_activations.tex (symmetric variant, same label as Table 2), 03_CLE_ShuffleNetV2,
03_AdaRound_extensions (contains a stray template table), 03_adaround_intro. Do not cite those.

Table 6 (p17), standard PTQ pipeline, avg 5 runs:
| Model | FP32 | per-tensor W8A8 | per-tensor W4A8 | per-channel W8A8 | per-channel W4A8 |
|---|---|---|---|---|---|
| ResNet18 | 69.68 | 69.60 | 68.62 | 69.56 | 68.91 |
| ResNet50 | 76.07 | 75.87 | 75.15 | 75.88 | 75.43 |
| MobileNetV2 | 71.72 | 70.99 | 69.21 | 71.16 | 69.79 |
| InceptionV3 | 77.40 | 77.68 | 76.48 | 77.71 | 76.82 |
| EfficientNet lite | 75.42 | 75.25 | 71.24 | 75.39 | 74.01 |
| DeeplabV3 (MNv2, VOC mIoU) | 72.94 | 72.44 | 70.80 | 72.27 | 71.67 |
| EfficientDet-D1 (COCO mAP) | 40.08 | 38.29 | 0.31 | 38.67 | 35.08 |
| BERT-base (GLUE, dagger) | 83.06 | 82.43 | 81.76 | 82.77 | 82.02 |
Dagger: a few activations kept at 16 bits. Prose (Sec. 3.6): W8A8 "within 0.7%" for all models -- contradicted by the
EfficientDet-D1 per-tensor row (1.79 points) and MobileNetV2 per-tensor (0.73). EfficientDet-D1 per-tensor W4A8
collapses to 0.31 mAP; per-channel recovers 35.08.

Table 10 (p24), standard QAT pipeline, avg 3 runs:
| Model | FP32 | PT W8A8 | PT W4A8 | PT W4A4 | PC W8A8 | PC W4A8 | PC W4A4 |
|---|---|---|---|---|---|---|---|
| ResNet18 | 69.68 | 70.38 | 69.76 | 68.32 | 70.43 | 70.01 | 68.83 |
| ResNet50 | 76.07 | 76.21 | 75.89 | 75.10 | 76.58 | 76.52 | 75.53 |
| InceptionV3 | 77.40 | 78.33 | 77.84 | 77.49 | 78.45 | 78.12 | 77.74 |
| MobileNetV2 | 71.72 | 71.76 | 70.17 | 66.43 | 71.82 | 70.48 | 66.89 |
| EfficientNet lite | 75.42 | 75.17 | 71.55 | 70.22 | 74.75 | 73.92 | 71.55 |
| DeeplabV3 (MNv2, VOC) | 72.94 | 73.99 | 70.90 | 66.78 | 72.87 | 73.01 | 68.90 |
| EfficientDet-D1 (COCO) | 40.08 | 38.94 | 35.34 | 24.70 | 38.97 | 36.75 | 28.68 |
| BERT-base (GLUE) | 83.06 | 83.26 | 82.64 | 78.83 | 82.44 | 82.39 | 77.63 |
Note: EfficientDet-D1 QAT W8A8 (38.94 / 38.97) is barely above PTQ W8A8 (38.29 / 38.67); QAT matters at W4.

Table 3 (p12), CLE, MobileNetV2 ImageNet, FP32 / INT8: original 71.72 / 0.12; + CLE 71.70 / 69.91; + absorbing bias
71.57 / 70.92; per-channel quantization 71.72 / 70.65. CLE + bias absorption with per-tensor beats per-channel.
Table 4 (p14), bias correction, MobileNetV2: original 71.72 / 0.12; + bias correction 71.72 / 52.02; CLE + bias
absorption 71.57 / 70.92; + bias correction 71.57 / 71.19 (data-free).
Table 5 (p15), AdaRound ablation, ResNet18 W4, first layer / all layers: nearest 52.29 / 23.99; Hessian task loss
(Eq. 30) 68.62 / N/A; continuous relaxation MSE (Eq. 32) 69.58 / 66.56; AdaRound (Eq. 35) 69.58 / 68.60.
Table 1 (p9), weight range setting (symmetric, activations FP32), ResNet18 W8/W6/W4 | MobileNetV2 W8/W6/W4:
min-max 69.57/63.90/0.12 | 71.16/64.48/0.59; MSE 69.45/64.64/18.82 | 71.15/65.43/13.77; min-max per-channel
69.60/69.08/44.49 | 71.21/68.52/18.40; MSE per-channel 69.66/69.24/54.67 | 71.46/68.89/27.17.
Table 2 (p10), activation range setting (asymmetric, weights FP32), ResNet18 A8/A6/A4 | MobileNetV2 A8/A6/A4:
min-max 69.60/68.19/18.82 | 70.96/64.58/0.53; MSE 69.59/67.84/31.40 | 71.35/67.55/13.57; MSE + Xent
69.60/68.91/59.07 | 71.36/68.85/30.94; BN (alpha=6) 69.54/68.73/23.83 | 71.32/65.20/0.66.
Tables 7-9 (QAT ablations, ResNet18 | MobileNetV2): see claims nagel21-t7/t8/t9. Key rows: static folding W4A8
69.76 | 70.17 vs double forward 69.42 | 66.87; MobileNetV2 W4A8 QAT without CLE 0.10, with CLE 70.13.

Fig. 5 (p11), toy-shaping facts (vector coordinates of src/figures/channel_ranges_mnv2.pdf, our reading, uncertain):
first depthwise-separable layer of MobileNetV2 after BN folding, 32 output channels, box = quartiles, whiskers = min
and max (no fliers drawn). Six channels (5, 15, 16, 28, 29, 30) reach beyond +-40: ch28 about -29 to 95, ch30 about
-57 to 94, ch16 about -60 to 63, ch15 about -46 to 59, ch29 about -45 to 54, ch5 about -25 to 67. The other 26 stay
within about +-16 (widest: ch4 about -6.5 to 16; most within about +-5). Channels 11 and 23 are all zero. A per-tensor
8-bit grid sized for +-95 gives a step of about 0.75, so a +-1 channel gets only a handful of levels (our reading).

## 9. Limitations stated
- Homogeneous bit-width only; mixed precision out of scope (Sec. 2.4).
- Group/vector-wise quantizers not considered because most accelerators lack support (Sec. 2.2.3).
- CLE only within residual blocks; vanilla CLE can unbalance activation ranges (Sec. 3.2, 3.7).
- PTQ insufficient for 4-bit activations; W4A4 QAT on depthwise nets still loses up to 5 points (Sec. 4.5).
- Later fixes from the given sources: BRECQ (li2021brecq) extends AdaRound's layer reconstruction to blocks and reaches
  usable W2A4 / W4A4 PTQ.

## 10. Outline feedback
- Table numbers: the standard QAT table is Table 10, not Table 9 (Table 9 is the CLE/bias-correction init ablation).
  CLE, bias correction and AdaRound are Tables 3, 4, 5; Tables 1-2 are range setting (outline says "Tables 2-5").
- PTQ pipeline = Fig. 8 (p16); debugging flow chart = Fig. 9 (p18), Sec. 3.7 (source file 036_ptq_diagnostics.tex).
- The MSE range-setting equation is Eq. 16 (Sec. 3.1).
- "DeepLabV3" in Nagel is MobileNetV2-backbone on Pascal VOC; Wu's DeepLabV3 is ResNet-101 on COCO. Never put them in one
  column without the backbone and dataset.
- The guide should quote Table 6 rows, not the "within 0.7%" sentence (falsified by EfficientDet-D1 per-tensor).
- ch-qat: Fig. 10 is the STE graph; Table 7 gives the BN folding options; QAT setup facts are in Sec. 4.5.
