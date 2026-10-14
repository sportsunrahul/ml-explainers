# Fact check: notes/claims/pruning.json and notes/claims/fp8-trt.json

Checker: independent pass, 2026-10-09/10. Scripts and extracted text live in /tmp/fc-pf8/ (not in sources/).
Backups of the pre-edit claim files: /tmp/fc-pf8/backup/.

## Summary

- Claims: 961 (pruning.json 553, fp8-trt.json 408).
- Mechanical pass: all 961 checked. The `value` string occurs verbatim in the source text (LaTeX, `pdftotext`
  output, abs.html, or TensorRT .txt/.html) for 945. The 16 misses are all explained below; none is wrong.
- arXiv ids, versions and dates: every `arxiv` field matches the latest version on abs.html, and every
  date/venue claim matches the abs.html submission history and Comments line.
- Semantic pass: all 961 claims read against the source paragraph or table cell. 348 table claims were matched
  by script to row and column (124 Mishra 2021 Tables 2-4, 224 van Baalen Tables 1-4, 6, 7); the rest by row and
  column inspection of the LaTeX tables and prose.
- FP8: all 256 codes of E4M3 and E5M2 enumerated independently from Micikevicius Table 1 (script
  /tmp/fc-pf8/scripts/fp8enum.py). Every extracted encoding fact holds (details below).
- Equations: every TeX span in notes/papers/{han2016dc, han2015, frankle2019, frankle2020lmc, li2017filters,
  liu2017slim, liu2019rethink, blalock2020, mishra2021nm, micikevicius2022, vanbaalen2023fp8, chen2018tvm}.md
  matches the LaTeX. All quoted sentences in notes/papers/tensorrt.md section 8 (11.0 removals) match the docs.
- Result: no `value` string was wrong. 30 field edits (claim text or loc) in 28 claims. No new `uncertain`
  marks were needed: the one plot-read value (mishra2021-fig3-speedup-k1280) was already uncertain, and the three
  TensorRT placeholders (trt-int4-woq-per-cc, trt-sparsity-gpu, trt-l4-cc) stay uncertain.
- Files stay valid JSON, ids unique, ASCII only, original formatting kept.

## Mechanical misses and how each was resolved

| ids | why the string is not in the text | resolution |
|---|---|---|
| han2016dc-fig11-{alexnet,vggnet,lenet300,lenet5}-{weight,index} (8) | Fig. 11 pie labels are glyphs in src/figures/ratio.pdf, not text | Rendered the figure: 48.6/51.4/0.1, 51.8/48.1/0.1, 51.0/47.5/1.5, 41.0/53.8/5.2. All match. |
| mic-e4m3-finite-codes, mic-e4m3-distinct-values, mic-e5m2-finite-codes, mic-e5m2-distinct-values | computed counts, labelled "our reading" | Confirmed by enumeration (254, 253, 248, 247). |
| trt-fetched | date of the fetch is not in the docs | File mtimes 2026-10-09 18:14; loc corrected (change 23). |
| trt-t4-precisions | composed list | supportData['11.4.0'].gpuPrecision row 7.5: FP32, FP16, INT8 Supported; TF32, BF16, FP8, FP4 N/A. |
| trt-int4-woq-per-cc, trt-sparsity-gpu | placeholders, already uncertain | Kept uncertain. |

## FP8 enumeration (Micikevicius Table 1 rules)

| fact | E4M3 | E5M2 | claims |
|---|---|---|---|
| bias | 7 | 15 | ok |
| NaN codes | 2 (0x7F, 0xFF) | 6 (0x7D-0x7F, 0xFD-0xFF) | ok |
| inf codes | 0 | 2 (0x7C, 0xFC) | ok |
| finite codes / distinct values | 254 / 253 | 248 / 247 | ok |
| positive non-zero finite | 126 | 123 | ok |
| max | 448 = 1.75 * 2^8 | 57344 = 1.75 * 2^15 | ok |
| min normal | 2^-6 | 2^-14 | ok |
| max subnormal | 0.875 * 2^-6 | 0.75 * 2^-14 | ok |
| min subnormal | 2^-9 | 2^-16 | ok |
| binades incl. subnormals | 18 | 32 | ok |
| E4M3 with IEEE specials | max 240, 17 binades | - | ok |
| E4M3 values at exponent 1111 | 256, 288, 320, 352, 384, 416, 448; 480 if m=111 freed | - | ok |
| E5M2 extra magnitudes with one NaN code | - | 3 | ok |

## Changes (id, field, old -> new, reason, loc)

1. `han2016dc-t7-pruning-size` (claim)
   - old: AlexNet pruning-only storage (MB) incl. index
   - new: AlexNet pruning-only storage (MB), Table 7 row 'Pruning' (Han 2015); the Table 7 caption says this 9x figure does not include index overhead
   - reason: Old text said 'incl. index'; Table 7 caption: 'Network pruning reduced the parameters by 9x, not including index overhead.'
   - loc: Table 7, row Pruning
2. `mishra2021-fig3-speedup-k1280` (loc)
   - old: Fig. 3, first bar (read from plot, approximate)
   - new: Fig. 3, first bar (read from plot, approximate; bar top reads about 1.23 in the rendered src/fig_gemmperf.pdf; no printed number)
   - reason: Added the plot reading; value stays a plot read (already uncertain).
   - loc: Fig. 3, first bar (read from plot, approximate; bar top reads about 1.23 in the rendered src/fig_gemmperf.pdf; no printed number)
3. `vb-t2-resnet18-fp32` (claim)
   - old: ResNet18, QAT fine-tuned accuracy, FP32
   - new: ResNet18, FP32 baseline (Table 2 FP32 column; not a QAT result)
   - reason: The FP32 column of Table 2 is the unquantized baseline, not a QAT fine-tuned result.
   - loc: Table 2 (QAT results; LSQ range learning), p.9, row ResNet18, col FP32
4. `vb-t2-mobilenetv2-fp32` (claim)
   - old: MobileNetV2, QAT fine-tuned accuracy, FP32
   - new: MobileNetV2, FP32 baseline (Table 2 FP32 column; not a QAT result)
   - reason: The FP32 column of Table 2 is the unquantized baseline, not a QAT fine-tuned result.
   - loc: Table 2 (QAT results; LSQ range learning), p.9, row MobileNetV2, col FP32
5. `vb-t2-hrnet-fp32` (claim)
   - old: HRNet, QAT fine-tuned accuracy, FP32
   - new: HRNet, FP32 baseline (Table 2 FP32 column; not a QAT result)
   - reason: The FP32 column of Table 2 is the unquantized baseline, not a QAT fine-tuned result.
   - loc: Table 2 (QAT results; LSQ range learning), p.9, row HRNet, col FP32
6. `vb-t2-deeplabv3-fp32` (claim)
   - old: DeeplabV3 (MobileNetV2 backbone, Pascal VOC), QAT fine-tuned accuracy, FP32
   - new: DeeplabV3 (MobileNetV2 backbone, Pascal VOC), FP32 baseline (Table 2 FP32 column; not a QAT result)
   - reason: The FP32 column of Table 2 is the unquantized baseline, not a QAT fine-tuned result.
   - loc: Table 2 (QAT results; LSQ range learning), p.9, row DeeplabV3 (MobileNetV2 backbone, Pascal VOC), col FP32
7. `vb-t2-salsanext-fp32` (claim)
   - old: SalsaNext (SemanticKITTI), QAT fine-tuned accuracy, FP32
   - new: SalsaNext (SemanticKITTI), FP32 baseline (Table 2 FP32 column; not a QAT result)
   - reason: The FP32 column of Table 2 is the unquantized baseline, not a QAT fine-tuned result.
   - loc: Table 2 (QAT results; LSQ range learning), p.9, row SalsaNext (SemanticKITTI), col FP32
8. `vb-t2-bert-fp32` (claim)
   - old: BERT (GLUE avg), QAT fine-tuned accuracy, FP32
   - new: BERT (GLUE avg), FP32 baseline (Table 2 FP32 column; not a QAT result)
   - reason: The FP32 column of Table 2 is the unquantized baseline, not a QAT fine-tuned result.
   - loc: Table 2 (QAT results; LSQ range learning), p.9, row BERT (GLUE avg), col FP32
9. `vb-t2-hrnet-fp32` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row HRNet, col FP32
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP32
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP32
10. `vb-t2-hrnet-int8` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row HRNet, col INT8
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col INT8
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col INT8
11. `vb-t2-hrnet-e2` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row HRNet, col FP8-E2
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP8-E2
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP8-E2
12. `vb-t2-hrnet-e3` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row HRNet, col FP8-E3
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP8-E3
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP8-E3
13. `vb-t2-hrnet-e4` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row HRNet, col FP8-E4
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP8-E4
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 HRNet: no range learning, per-tensor, (all)minmax ranges, SGD lr 1e-5, 100 epochs, batch 4), p.9, row HRNet, col FP8-E4
14. `vb-t2-salsanext-fp32` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row SalsaNext (SemanticKITTI), col FP32
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP32
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP32
15. `vb-t2-salsanext-int8` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row SalsaNext (SemanticKITTI), col INT8
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col INT8
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col INT8
16. `vb-t2-salsanext-e2` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row SalsaNext (SemanticKITTI), col FP8-E2
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP8-E2
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP8-E2
17. `vb-t2-salsanext-e3` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row SalsaNext (SemanticKITTI), col FP8-E3
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP8-E3
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP8-E3
18. `vb-t2-salsanext-e4` (loc)
   - old: Table 2 (QAT results; LSQ range learning), p.9, row SalsaNext (SemanticKITTI), col FP8-E4
   - new: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP8-E4
   - reason: Sec 4.3 says QAT uses LSQ range learning, but App. C.3 says HRNet and SalsaNext rows 'don't learn ranges to reduce memory requirements'.
   - loc: Table 2 (QAT results; this row did NOT use LSQ range learning: App. C.3 SalsaNext: no range learning, per-channel, MSE ranges, SGD lr 5e-6, 75 epochs, batch 4), p.9, row SalsaNext (SemanticKITTI), col FP8-E4
19. `trt-timing-iterations` (loc)
   - old: performance/optimization, Advancing Performance Optimization Techniques, 'Deterministic Tactic Selection'
   - new: performance/optimization, Advancing Performance Optimization Techniques, 'Increasing Average Timing Iterations': 'By default, TensorRT runs each tactic for at least four iterations and takes the average latency'
   - reason: Sentence sits under 'Increasing Average Timing Iterations', not 'Deterministic Tactic Selection'.
   - loc: performance/optimization, Advancing Performance Optimization Techniques, 'Increasing Average Timing Iterations': 'By default, TensorRT runs each tactic for at least four iterations and takes the average latency'
20. `trt-sparsity-gpu` (claim)
   - old: Structured 2:4 sparsity support and GPUs: not present in any saved TensorRT 11.4.0 page
   - new: Structured 2:4 sparsity support by GPU: no saved TensorRT 11.4.0 page lists it; the only trace is the Table 21 alignment row 'FP16: 8 for dense math, 16 for sparse math' (performance/optimization, Optimizing for Tensor Cores)
   - reason: Old text said 'not present in any saved page'; 'sparse math' does occur once (Table 21).
   - loc: searched all sources/tensorrt/*.txt and *.html for 'sparsity', '2:4', 'sparse'
21. `mic-s43-pertensor-gemm` (claim)
   - old: GPT3 1.3B perplexity with calibrated per-tensor scales, GEMM inputs only
   - new: GPT3 1.3B perplexity with calibrated scaling factors (per-channel for weights, per-tensor for activations, the int8 convention), GEMM inputs only
   - reason: Sec 4.3: 'calibrate the tensors to have their own scaling factors (following the convention of int8 quantization to use per-channel and per-tensor scaling factors for weights and activations, respectively)'; old text said 'per-tensor scales'.
   - loc: Sec 4.3 Per-tensor scaling factors; Figure 2 (GPT3 1.3B, wikitext103, bf16-trained, cast to E4M3, no per-tensor scale)
22. `mic-s43-pertensor-residual` (claim)
   - old: GPT3 1.3B perplexity with calibrated per-tensor scales, GEMM inputs + residuals
   - new: GPT3 1.3B perplexity with calibrated scaling factors (per-channel for weights, per-tensor for activations, the int8 convention), GEMM inputs + residuals
   - reason: Sec 4.3: 'calibrate the tensors to have their own scaling factors (following the convention of int8 quantization to use per-channel and per-tensor scaling factors for weights and activations, respectively)'; old text said 'per-tensor scales'.
   - loc: Sec 4.3 Per-tensor scaling factors; Figure 2 (GPT3 1.3B, wikitext103, bf16-trained, cast to E4M3, no per-tensor scale)
23. `trt-fetched` (loc)
   - old: sources/tensorrt (state.json / task)
   - new: sources/tensorrt file mtimes 2026-10-09 18:14; state.json 'created': '2026-10-09'; task statement
   - reason: Old loc cited state.json for the fetch date; state.json has only 'created'; file mtimes confirm.
   - loc: sources/tensorrt file mtimes 2026-10-09 18:14; state.json 'created': '2026-10-09'; task statement
24. `vb-hw-e4-fp16acc` (claim)
   - old: Extra 2-input gates for FP8-E4 with FP16 accumulator vs INT8 with 27-bit fixed-point accumulator
   - new: Extra 2-input gates for FP8-E4 with FP16 accumulator vs INT8 (INT8 baseline = the 27-bit fixed-point accumulator bar: our reading; the text says only 'comparing INT8 with FP8-E4')
   - reason: Paper does not name the INT8 accumulator for the 53% comparison; label the baseline as our reading.
   - loc: Sec 3.3 para after Fig 3; Figure 3 caption
25. `vb-t6-yolov3-fp32` (claim)
   - old: Yolov3 detection mAP, FP32 (AIMET table)
   - new: Yolov3 detection mAP, FP32 (AIMET table); Table 6 does not state the dataset or the mAP definition (Table 1 YoloV5 is COCO, 56.3, a different setup)
   - reason: Table 6 / App. E.1 give no dataset or mAP variant for the detection rows.
   - loc: Table 6 (AIMET INT8 results; Appendix E.1), row Yolov3, col FP32 KPI
26. `vb-t6-yolov3-int8` (claim)
   - old: Yolov3 detection mAP, INT8 via AIMET, method PTQ,INT8+W8A16
   - new: Yolov3 detection mAP, INT8 via AIMET, method PTQ,INT8+W8A16; Table 6 does not state the dataset or the mAP definition (Table 1 YoloV5 is COCO, 56.3, a different setup)
   - reason: Table 6 / App. E.1 give no dataset or mAP variant for the detection rows.
   - loc: Table 6 (AIMET INT8 results; Appendix E.1), row Yolov3, col INT8 KPI (Method: PTQ,INT8+W8A16)
27. `vb-t6-yolov5-fp32` (claim)
   - old: Yolo-v5 detection mAP, FP32 (AIMET table)
   - new: Yolo-v5 detection mAP, FP32 (AIMET table); Table 6 does not state the dataset or the mAP definition (Table 1 YoloV5 is COCO, 56.3, a different setup)
   - reason: Table 6 / App. E.1 give no dataset or mAP variant for the detection rows.
   - loc: Table 6 (AIMET INT8 results; Appendix E.1), row Yolo-v5, col FP32 KPI
28. `vb-t6-yolov5-int8` (claim)
   - old: Yolo-v5 detection mAP, INT8 via AIMET, method PTQ,INT8+W8A16
   - new: Yolo-v5 detection mAP, INT8 via AIMET, method PTQ,INT8+W8A16; Table 6 does not state the dataset or the mAP definition (Table 1 YoloV5 is COCO, 56.3, a different setup)
   - reason: Table 6 / App. E.1 give no dataset or mAP variant for the detection rows.
   - loc: Table 6 (AIMET INT8 results; Appendix E.1), row Yolo-v5, col INT8 KPI (Method: PTQ,INT8+W8A16)
29. `vb-t6-yolor-fp32` (claim)
   - old: YoloR detection mAP, FP32 (AIMET table)
   - new: YoloR detection mAP, FP32 (AIMET table); Table 6 does not state the dataset or the mAP definition (Table 1 YoloV5 is COCO, 56.3, a different setup)
   - reason: Table 6 / App. E.1 give no dataset or mAP variant for the detection rows.
   - loc: Table 6 (AIMET INT8 results; Appendix E.1), row YoloR, col FP32 KPI
30. `vb-t6-yolor-int8` (claim)
   - old: YoloR detection mAP, INT8 via AIMET, method PTQ
   - new: YoloR detection mAP, INT8 via AIMET, method PTQ; Table 6 does not state the dataset or the mAP definition (Table 1 YoloV5 is COCO, 56.3, a different setup)
   - reason: Table 6 / App. E.1 give no dataset or mAP variant for the detection rows.
   - loc: Table 6 (AIMET INT8 results; Appendix E.1), row YoloR, col INT8 KPI (Method: PTQ)

## Statements in outline.md or notes/understanding.md that the sources contradict

1. outline.md, ch-guide: "Table 6 QAT Yolov3 0.653 -> 0.649, Yolo-v5 0.557 -> 0.559."
   Source (vanbaalen2023fp8 src/sections/09_appendix.tex, Table 6 = tab:int8results): the Method column for both
   rows is "PTQ,INT8+W8A16", not QAT ("& Yolov3 & 0.653 & 0.649 & PTQ,INT8+W8A16", "& Yolo-v5 & 0.557 & 0.559 &
   PTQ,INT8+W8A16"). The caption says INT8+W8A16 means mixed precision: "utilizing mostly INT8 and W8A16 for some
   small parts of the network". Only Mnv2-ssd-lite in that block is QAT. Table 6 also gives no dataset or mAP
   definition for these rows. Fix: "Table 6 (AIMET, PTQ with some W8A16 layers) Yolov3 0.653 -> 0.649, Yolo-v5
   0.557 -> 0.559". The claim files already said PTQ.
2. outline.md, ch-guide: "EfficientNMS plugin removed in 11.0."
   Source (release-notes 11.0.0, Deprecated and Removed Features; migration 10.x-to-11.x, Other Removals): the
   16 removed plugins include "EfficientNMS_ONNX_TRT". No saved page lists "EfficientNMS_TRT" as removed (it does
   not occur anywhere in the saved docs). Fix: "EfficientNMS_ONNX_TRT (one of 16 pre-10 plugins) removed in 11.0;
   the saved docs do not say EfficientNMS_TRT was removed."
3. notes/understanding.md: "Its YOLO rows are the only YOLO numbers in the sources (Qualcomm tooling)."
   and outline.md ch-guide: "YOLO rows exist only in van Baalen 2303.17951".
   Source: sources/frantar2022obc/src/neurips.tex, table tab:unstr ("Unstructured pruning for different FLOP
   reduction targets"): "YOLOv5l -- 66.97" with GMP / L-OBS / AdaPrune / ExactOBS at 2x, 3x, 4x (e.g. ExactOBS
   66.14 / 65.35 / 64.05), plus YOLOv5 mixed quantization + 2:4 plots (fig:yolo-mixed, fig:mixed-comp-yolo).
   outline.md ch-gptq-awq already says "OBC has YOLOv5 rows", so the outline contradicts itself. Fix: "YOLO INT8
   accuracy deltas exist only in van Baalen; OBC has YOLOv5 pruning rows (and quantization + 2:4 only as plots)."
4. outline.md, ch-lottery viz-imp: "iterative magnitude pruning loop stepper (train -> prune 20% -> rewind -> repeat)".
   Source (frankle2019 src/intro.tex): "Reset the remaining parameters to their values in $\theta_0$". LTH resets to
   initialization; "rewinding" to iteration k > 0 is the LMC term (frankle2020lmc, Algorithm 2). Also the rate:
   Fig. 2 gives "fc20%" for LeNet but "conv10% fc20%" for Conv-2/4 and "conv15% fc20%" for Conv-6. Fix: "train ->
   prune 20% -> reset to theta_0 -> repeat (LeNet rate)".
5. notes/understanding.md: "Liu 2017 (BN-gamma slimming), Liu 2019 (pruned architecture from scratch is as good)".
   Source (liu2019rethink src/experiments.tex, Table 6, unstructured pruning, ResNet-50 ImageNet): fine-tuned
   76.06 / 76.09 vs Scratch-B 75.70 / 74.91 at 30% / 60% pruned. Scratch matches for structured pruning (Tables
   1, 4, 12 checked cell by cell; 2, 3, 5 per the paper's framing) but not for unstructured pruning on ImageNet. Qualify: "for structured pruning".
6. outline.md, ch-prune: "Sourced: Han Tables 1-6 (compression per network)". Tables 1-5 are compression
   statistics; Table 6 is AlexNet accuracy versus CONV/FC bit-widths (8/5, 8/4, 4/2). Minor label fix.
7. notes/understanding.md: "LMC 2019 (branch)" versus the claim/notes label "Frankle 2020" (arXiv v1 2019-12-11,
   ICML 2020). Label inconsistency only.

Checked and consistent (no change needed): Deep Compression 35x / 49x and every Table 1-8 cell; Table 8 at batch 64
the sparse layer is slower in 16 of 18 cells (Titan X 6/6, Core i7 6/6, Tegra K1 4/6; the two exceptions are TK1
AlexNet FC7 and VGG16 FC7) and the GPU is a GeForce GTX Titan X, 4096 samples averaged, cudaMalloc/cudaMemcpy not
timed, gemv/csrmv at batch 1 and gemm/csrmm at batch 64; the quantized (shared-weight) model was not timed ("we
didn't benchmark the quantized model"). LTH hypothesis statement, IMP steps, 15 reinits per point, LeNet and
Conv-2/4/6 points, the warmup caveat (VGG-19 k = 10000; ResNet-20 lr 0.03, k = 20000), "Resnet-18" = ResNet-20
(LMC Table 1 caption). LMC Table 2 at k = 0 (ResNet-50 76.1 / 73.7 / 73.1 / 73.4), stable at epoch 5 (5.5%) vs 18
(20%), matching < 0.2%, stable < 2% (Fig. 9 caption). Li 2017 Tables 1 and 3; Liu 2017 Table 1(a); Liu 2019
Tables 1, 4, 6, 12; Blalock counts. Mishra 2:4 definition, 2-bit metadata, 12.5% / 25% overhead, A100 Table 1,
recipe (train dense, prune 2 of 4 by magnitude, retrain with the same optimizer and schedule, keep the mask),
detectors pruned after the second phase and only that phase repeated (Sec. 4.3), Table 4 = 15 rows with Dense
FP16 / Sparse FP16 / Sparse INT8 (no dense INT8 column), INT8 = entropy per-tensor activations and max
per-channel weights, Fig. 3 = A100 INT8 GEMM, M = N = 10240, K 1280..20480, "nearly a 2x speedup"; no end-to-end
speedup is measured. van Baalen conclusions (50% to 180%; 53% / 183% gates; "It's all about the outliers"; 255-value
format; Student-t E4 5 bits), Table 1 YoloV5 COCO 56.3 / -1.8. TVM four operator categories, 1.2x to 2x fusion
speedup, gradient tree boosting default cost model. TensorRT build and runtime phases, fusion catalog entries,
quantization schemes (INT8 [-128, 127], FP8 E4M3 [-448, 448], round to nearest even), 11.0 removal sentences
(quoted exactly in notes/papers/tensorrt.md section 8), precision by compute capability (7.5 FP32/FP16/INT8;
8.0/8.6 add TF32, BF16; 8.9/9.0 add FP8 with FP4 in emulation, footnote f3; 10.0+ native FP4), TensorRT 8 engines
do not run on an 11 runtime, FP8 conv kernels > 32 fall back on SM89/90/120/121.

## Remaining doubts

- mishra2021-fig3-speedup-k1280 = "1.2" is a plot read (bar top is about 1.23). Kept uncertain.
- trt-int4-woq-per-cc, trt-sparsity-gpu, trt-l4-cc: placeholders with no page claim. L4 is not a listed device in
  the 11.4.0 support matrix; its CC 8.9 needs another source. Kept uncertain.
- vb-hw-e4-fp16acc: the paper says "comparing INT8 with FP8-E4, there is a 53% increased cost" without naming the
  INT8 accumulator; the 27-bit fixed-point baseline is our reading (now labelled).
- van Baalen Table 1 RangeNet++: baseline printed as 0.305 while its deltas (-1.67 ...) look like points; Table 6
  gives RangeNet++ 47.2 (QAT). Units look inconsistent in the source. Value quoted as printed.
- van Baalen Table 1 YoloV5 56.3 (COCO, App. C.1) and Table 6 Yolo-v5 0.557 are different setups; do not compare.
- Mishra Table 4 "RPN-RN50-FPN-1x" is listed under "bbox Average Precision"; an RPN is usually scored by recall.
  Quoted as the table labels it.
- Source-internal inconsistencies already carried by claims: Han 2015 Table 5 caption 12x vs Table 1 13x (VGG-16);
  Han 2016 Sec. 2 relative-index bits 8 (conv) / 5 (fc) vs Table 4 index bits 4 (AlexNet); Han 2016 "61 million
  parameters" (Sec. 5.2) vs "60 million weights to quantize" (Sec. 6.1); LTH Fig. 2 274K vs Sec. 4 "271,000"
  (ResNet-20); van Baalen Tables 2, 3, 4 give different FP8-QAT numbers for the same nets (69.35 / 69.58 / 69.79 for
  ResNet18).
- Two "quotes" in notes/papers are paraphrases, not verbatim: han2016dc.md "learn the connectivity" (source:
  "learning the connectivity via normal network training"); micikevicius2022.md "skip the update on overflow"
  (source: "Weight update skipping (and reduction of the scaling factor) on overflows"). Not edited (outside the
  claim files).
- Claim ids han2016dc-fig3-centroid{0..3} number the centroids in reverse of the figure's index (centroid0 = index
  3: 2.00 -> 1.96). The claim text is correct; ids left unchanged so page references do not break.
