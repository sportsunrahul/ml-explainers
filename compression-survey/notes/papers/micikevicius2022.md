# micikevicius2022 -- FP8 Formats for Deep Learning (SPINE)

## 1. Identity
- Title: "FP8 Formats for Deep Learning" (LaTeX title: "FP8 Formats For Deep Learning").
- arXiv 2209.05433, read v2 (LaTeX source `src/main.tex` + `paper.pdf`, both v2).
- Dates (abs.html): v1 2022-09-12, v2 2022-09-29.
- Authors (abs.html order): Paulius Micikevicius ... Hao Wu (15 authors; NVIDIA, Arm, Intel). LaTeX author block groups by company.
- Venue: none stated. Repo: none.

## 2. Core problem
16-bit floats are the training/inference norm; INT8 inference needs PTQ calibration or QAT and sometimes still loses accuracy. The paper proposes one 8-bit floating-point interchange format (two encodings) that can be used for both training and inference without changing hyperparameters.

## 3. What prior methods did and why that hurt (per the paper)
- INT8 inference of FP-trained nets "require post-training quantization (PTQ) calibration and sometimes quantization-aware training (QAT)" and "some int8-quantized models may not completely recover the accuracy" (Sec 4.2).
- Earlier FP8 work: 5-bit-exponent training (Wang 2018, Mellempudi 2019), two FP8 formats with BN-statistic retuning (IBM 2019), a single-special-value variant with programmable bias (Graphcore, Noune 2022). Sec 1.
- Programmable exponent bias = only power-of-2 scale factors; the paper prefers a software per-tensor scale of any real value (Sec 3.2).
- FP16-AMP style "skip the update on overflow" is a bad fit for FP8: overflows are more likely, too many skipped updates (Sec 2).

## 4. The one key idea
Two encodings: E4M3 for weights and activations (precision), E5M2 for gradients (range). E5M2 is IEEE-754-like (FP16 with 2 mantissa bits). E4M3 drops infinities and keeps a single NaN mantissa pattern, which buys one more binade (max 448 instead of 240). A per-tensor scale factor in software places each tensor's range on the format.

## 5. Mechanism, step by step
1. Compute in higher precision; GEMM/dot-product outputs are wider than FP8 (Sec 2, para 2).
2. Before casting to FP8, multiply by a scale factor "such that the maximum magnitude in the tensor becomes close to the maximum representable magnitude in the corresponding format" (Sec 2, para 3).
3. Cast to FP8; values that overflow are saturated to the max representable value (Sec 2, para 3). Non-saturating mode "can be provided" (Sec 2, para 4).
4. Special values: inf and NaN in the wider type both become NaN in E4M3 (Sec 2, para 4).
5. Unscale by multiplying with the inverse scale, either after converting back from FP8 or after the GEMM; for GEMMs this is "once per dot-product" (Sec 2, para 3).
6. Rounding mode (RNE, stochastic, ...) is left to the implementation (Sec 2, para 4).
7. Experiments: simulated FP8 (clip to FP8-representable values, arithmetic in FP16/BF16) on GEMM inputs only: activations, weights, activation gradients; GEMM outputs left in higher precision (Sec 4, Sec 4.1).

## 6. Equations / encodings
No numbered equations. The definitions are Table 1 (`\label{tab:encodings}`), verbatim TeX rows:
```
Exponent bias & 7  & 15 \\
Infinities     & N/A & $S.11111.00_{2}$ \\
NaN     & $S.1111.111_{2}$  & $S.11111.\{01, 10, 11\}_{2}$ \\
Zeros & $S.0000.000_{2}$ & $S.00000.00_{2}$ \\
Max normal & $S.1111.110_{2} = 1.75 * 2^{8} = 448$ & $S.11110.11_{2} = 1.75 * 2^{15} = 57,344$ \\
Min normal & $S.0001.000_{2} = 2^{-6}$ & $S.00001.00_{2} = 2^{-14}$ \\
Max subnorm & $S.0000.111_{2} = 0.875 * 2^{-6}$ & $S.00000.11_{2} = 0.75 * 2^{-14}$ \\
Min subnorm & $S.0000.001_{2} = 2^{-9}$ & $S.00000.01_{2} = 2^{-16}$ \\
```
Paper's wording: "S.E.M notation ... S is the sign bit, E is the exponent field (either 4 or 5 bits containing biased exponent), M is either a 3- or a 2-bit mantissa. Values with a 2 in the subscript are binary" (Sec 3).

Decoding rule (our reading, standard IEEE-style, consistent with every Table 1 row and the Sec 3.1 numbers):
- exponent field e = 0: subnormal, value = (-1)^S * (M / 2^m) * 2^(1 - bias).
- e > 0: normal, value = (-1)^S * (1 + M / 2^m) * 2^(e - bias).
- E4M3 (m=3, bias 7): only S.1111.111 is NaN (2 codes: 0x7F, 0xFF); no infinities; S.1111.000..110 are normal numbers 256..448.
- E5M2 (m=2, bias 15): e = 11111 is special: M=00 -> +-inf (2 codes), M=01/10/11 -> NaN (6 codes).

Enumeration check (our script over all 256 codes, rules above):
| | E4M3 | E5M2 |
|---|---|---|
| NaN codes | 2 | 6 |
| Inf codes | 0 | 2 |
| Zero codes (+0, -0) | 2 | 2 |
| Finite codes | 254 | 248 |
| Distinct finite values (+0 = -0) | 253 | 247 |
| Positive non-zero finite values | 126 | 123 |
| Max | 448 | 57344 |
| Min normal | 2^-6 = 0.015625 | 2^-14 |
| Min subnormal | 2^-9 = 0.001953125 | 2^-16 |
| Binades (min subnormal .. max) | 18 | 32 |
Binade counts match the paper: E4M3 "from 17 to 18 binades"; E5M2 "32 binades, including the subnormal values" (Sec 3.1).

## 7. Measurement setup (format + analysis lens)
- Training: same architecture, init, optimizer hyperparameters as the 16-bit baseline; baselines in FP16 or bfloat16 (the paper does not say which per row). Inputs to all GEMMs (incl. first conv and last FC for image models) clipped to FP8; outputs in higher precision. Simulated FP8, arithmetic in FP16/BF16. Which FP8 encoding per tensor in training is not stated per table; recommended use is E4M3 fwd, E5M2 grads (Sec 3).
- Image classification: ImageNet ILSVRC12, top-1 on val. Translation: WMT 2016 En->De, sacreBLEU on newstest2014. LM: Wikipedia (Transformer-XL), Pile variant + Common Crawl (GPT).
- PTQ (Sec 4.2, Table 5): FP16-trained models quantized to int8 or E4M3; per-channel weight scales, per-tensor activation scales; all matmul inputs incl. attention BMMs; weights max-calibrated; activations: best of max, percentile, MSE.
- Scaling study (Sec 4.3, Fig 2): GPT3 1.3B, bfloat16-trained, cast to E4M3 with various exponent biases, no per-tensor scale, wikitext103 perplexity.

## 8. Results worth remembering
Table 2 (ImageNet top-1, Baseline vs FP8 training), all rows:
| Model | Baseline | FP8 |
|---|---|---|
| VGG-16 | 71.27 | 71.11 |
| VGG-16 BN | 73.95 | 73.69 |
| Inception v3 | 77.23 | 77.06 |
| DenseNet 121 | 75.59 | 75.33 |
| DenseNet 169 | 76.97 | 76.83 |
| Resnet18 | 70.58 | 70.12 |
| Resnet34 | 73.84 | 73.72 |
| Resnet50 v1.5 | 76.71 | 76.76 |
| Resnet101 v1.5 | 77.51 | 77.48 |
| ResNeXt50 | 77.68 | 77.62 |
| Xception | 79.46 | 79.17 |
| MobileNet v2 | 71.65 | 71.04 |
| DeiT small | 80.08 | 80.02 |
"With the exception of MobileNet v2, accuracy achieved by FP8 training is within run-to-run variation" (Sec 4.1).

Table 3 (BLEU): GNMT 24.83/24.65; Transformer Base 26.87/26.83; Transformer Large 28.43/28.35.
Table 4 (perplexity): Transformer-XL Base 22.98/22.99; Transformer-XL Large 17.80/17.75; GPT 126M 19.14/19.24; GPT 1.3B 10.62/10.66; GPT 5B 8.94/8.98; GPT 22B 7.21/7.24; GPT 175B 6.65/6.68 (175B reported at 75% of training, baseline unfinished).

Table 5 (PTQ of 16-bit-trained models; 16-bit FP / int8 / E4M3):
| Model | Metric | 16-bit FP | int8 | E4M3 |
|---|---|---|---|---|
| BERT Base | SQuAD v1.1 F1 | 88.19 | 76.89 | 88.09 |
| BERT Large | SQuAD v1.1 F1 | 90.87 | 89.65 | 90.94 |
| GPT3 126M | wikitext103 ppl | 19.01 | 28.37 | 19.43 |
| GPT3 1.3B | wikitext103 ppl | 10.19 | 12.74 | 10.29 |
| GPT3 6.7B | wikitext103 ppl | 8.51 | 10.29 | 8.41 |
No vision rows in Table 5. Unscaled FP8 cast: "increasing the perplexity to 11.0" (Sec 4.2; the sentence does not name the model).

Sec 4.3 / Fig 2 (GPT3 1.3B, wikitext103): GEMM-input-only cast: several biases in [7, 10] match the bfloat16 baseline. With residual-add inputs also in FP8, "no single exponent bias value leads to sufficient accuracy"; bias 7 gives 12.59 vs 10.19 baseline. Per-tensor calibrated scales: 10.29 (GEMM-only) and 10.44 (GEMM+residuals).

No detection or segmentation model appears anywhere in the paper.

## 9. Limitations the paper states
- MobileNet v2 FP8 training not yet within noise ("We continue work on recovering the remaining accuracy", Sec 4.1).
- Only GEMM inputs in FP8; "Moving more tensors to FP8 is the subject of future study" (Sec 4.1).
- Simulated FP8; arithmetic implementation out of scope (Sec 4).
- Scale-selection heuristics "beyond the scope of this paper" (Sec 2).
- 175B baseline not finished (Sec 4.1).
- van Baalen 2303.17951 (Sec 3.2, Sec 5) criticises exactly this setup: only conv/linear in FP8, other ops and stored activations in FP16, so the speed comparison with all-INT8 is not like for like.

## 10. Outline feedback
- ch-fp8 "E4M3 keeps precision (max 448), E5M2 keeps range (max 57344)": correct. Paper prints "57,344"; pick one string form on the page and keep the claim value in the same form.
- ch-fp8 "A per-tensor scale is still needed": supported by Sec 2 and Sec 4.3 (bias 7 -> 12.59 vs per-tensor 10.44 for GEMM+residual).
- viz-fp8 must enumerate the NVIDIA/Arm/Intel encodings above (2 NaN codes in E4M3, 2 inf + 6 NaN in E5M2). Do not reuse van Baalen's variant (Appendix A: only 0x80 reserved, 255 values) for the code list.
- Rounding for the toy: the paper leaves rounding open; TensorRT's castToFp8 is round-to-nearest-even with clip to [-448, 448] (see tensorrt.md). Cite TensorRT for the rounding rule.
- "Sourced: Micikevicius format table and results": the paper has no detection/segmentation rows; only ImageNet classification for vision (Table 2, training, not PTQ). Say so on the page.
