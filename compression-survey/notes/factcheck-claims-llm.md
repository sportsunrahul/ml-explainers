# Fact check: gptq-awq.json, outliers.json, distill.json

Checker: independent pass, 2026-10-09/10. Scope: 977 claims (gptq-awq 623, outliers 163, distill 191) from
12 papers. Sources: `sources/<name>/` (abs.html, paper.pdf, src/*.tex; wang2019fgfi README.md and code/).
Scripts and extracted text live in `/tmp/fc_llm/` (outside the project). Nothing from `sources/` was executed.

Result in one line: no claim value is wrong. 19 field edits (loc or claim text) and 1 note fix. The real
errors are in `outline.md` and `notes/understanding.md` (Section 4).

## 1. Method

1. Mechanical pass over all 977 claims.
   - Text per paper: `pdftotext -layout` and raw, plus all `src/**/*.tex` with LaTeX comments stripped.
   - GPTQ ships two documents: `src/gptq.tex` matches the v2 PDF; `src/arxiv.tex` is an older layout. Only
     `gptq.tex` was used. Every GPTQ value occurs in it and in the PDF.
   - Each `value` was searched after normalizing TeX markup, unicode minus, ligatures and spaces.
   - `arxiv` id and version: every claim matches its paper, and every PDF stamp matches abs.html.
   - Dates: all 22 date claims were parsed against the abs.html `[vN]` lines (v1 and version read). All match.
2. Near-loc pass: the value must occur on the PDF page that holds the cited table/figure/section caption (+-1
   page), or on the cited page.
3. Row pass: for 654 table claims, the row label and the value must share one LaTeX or PDF line.
4. Semantic pass: every table claim was compared with its LaTeX table (row, column, model size, bit-width,
   group size, dataset, metric). Every prose claim was compared with its sentence. Figure-only claims were
   checked on rendered pages (LLM.int8() Fig. 1; SmoothQuant Figs. 4, 5, 8, 9, 10; AWQ Figs. 2, 3, 8, 9 text
   layer).
5. Equations: every fenced TeX block in the 12 `notes/papers/*.md` files was compared with the source TeX,
   ignoring whitespace. All blocks match. The only line-level misses were the `(1)`..`(5)` tags that
   `dettmers2022.md` appends; without the tags the lines match. GPTQ Algorithm 1 matches `gptq.tex` and the PDF
   line for line. PDF equation numbers cited in the notes match: GPTQ Eq. 1-5, AWQ Eq. 1-5, LLM.int8() Eq. 1-8,
   SmoothQuant Eq. 1-4, Hinton Eq. 1-5, DeiT Eq. 1-3, Wang Eq. 1-3.

## 2. Mechanical results

| check | checked | first-pass misses | resolution |
|---|---|---|---|
| value present in source | 977 | 12 | 10 are dates (substring search missed them; all verified against abs.html). 2 are figure-only: `xiao-fig4-channel-axis` (already uncertain; ticks 0-5000 confirmed visually) and `xiao-fig5-s` (confirmed on the rendered Fig. 5; loc updated). |
| near cited loc | 946 (dates and abs.html locs excluded) | 41 | All checked by hand. 6 are figure readings (LLM.int8() Fig. 1, SmoothQuant Figs. 4/5; on the cited page, as images). 2 are Hinton `10^-6`/`10^-9` (superscripts in the PDF text). 2 are code-only Wang defaults (found at the cited lines of `code/trainval_net_sup.py.txt`). The other 31 have prose locs ("Abstract", "App. A.2.2", "Sec. 5, Setup", "Sec. 6, GPU scenario") that the page heuristic cannot parse. Each value sits in the cited sentence. |
| row label on value's line | 654 | 129 | All are multirow or transposed layouts, checked by hand against the LaTeX: GPTQ Table 2; AWQ Tables 2 (TeX row labels), 4, 6, 9; SmoothQuant Tables 7, 8; Wang delta rows (Tables 1, 3); DeiT Table 1/5 macro rows; LLM.int8() Table 2; Wanda Table 5. Only Wanda Table 5 had a real loc error (transposed, fixed below). |

## 3. Changes made

All edits keep valid JSON, unique ids and ASCII. `distill.json` is unchanged.

| file | id | field | old -> new | reason | source loc |
|---|---|---|---|---|---|
| gptq-awq | wanda-t5-speedup-q-k-v-o-proj | loc | "Table 5, row Speedup, column q/k/v/o_proj" -> "Table 5, row q/k/v/o_proj, column Speedup" | Table is transposed: rows are layers, columns are Dense / 2:4 / Speedup | Wanda Table 5 (`tab:matmul_speedup`) |
| gptq-awq | wanda-t5-speedup-up-gate-proj | loc | "row Speedup, column up/gate_proj" -> "row up/gate_proj, column Speedup" | same | same |
| gptq-awq | wanda-t5-speedup-down-proj | loc | "row Speedup, column down_proj" -> "row down_proj, column Speedup" | same | same |
| gptq-awq | gptq-a6000-gpus | claim | "GPU count FP16 -> 3-bit on A6000" -> "A6000-48GB GPUs needed for FP16 OPT-175B (the 8 in '8->2'; 3-bit GPTQ needs 2)" | Value 8 is the FP16 count only; old text read as the whole transition | GPTQ Table 6, column GPU reduction |
| gptq-awq | gptq-vs-obq-speedup | claim | "Computational speedup of GPTQ over OBQ" -> "... (paper: 'more than three orders of magnitude')" | Paper states a lower bound | GPTQ Sec. 3, "Optimal Brain Quantization" |
| gptq-awq | awq-t10-* (12 ids: vila-7b/13b x fp16/awq-w4a16 x a100/4090/orin) | claim | "VILA throughput (tokens/s) with TinyChat: ..." -> "VILA throughput with TinyChat (Table 10 header 'Throughput'; unit not printed, tokens/s is our reading): ..." | Table 10 prints no unit | AWQ Table 10 (`tab:vila_latency`) |
| outliers | xiao-speedup-pytorch | claim | "... OPT-30B, seq len 256, batch 4, single A100 80GB" -> "... OPT-30B, seq len 256, single A100 80GB (batch 4 is from the Table 11 caption, whose OPT-30B/256 values 343.0 -> 227.6 ms match Fig. 8; Fig. 8 does not print the batch)" | Batch size is not stated for Fig. 8 | SmoothQuant Fig. 8, Table 11 |
| outliers | xiao-fig5-s | loc | "Fig 5 (p4), dashed box under X-hat" -> "... labelled 's = sqrt(max\|X\|/max\|W\|)'; figure image only (no text occurrence), confirmed on rendered page 4" | Value is only in the figure image; verified, not uncertain | SmoothQuant Fig. 5 |

Note fix (not a claim file): `notes/papers/xiao2023.md`, worked example. Old: "X-hat = [[1, -4, 2, 2], [-2, 2,
-1, -3]] with channel max (1, 4, 1, 3)". New: "... with channel max (2, 4, 2, 3) (the dashed box (1, 4, 1, 3)
under X-hat is s, per its label)". Column maxima of the printed X-hat are 2, 4, 2, 3. The note had called it
"Good seed for the toy's readout check", so the toy would have checked against wrong numbers.

Newly marked uncertain: none. The 10 pre-existing `uncertain` claims stay as they are (Section 5).

## 4. Statements in outline.md / understanding.md that the sources contradict

1. **"YOLO only in van Baalen".**
   - understanding.md line 65: "Its YOLO rows are the only YOLO numbers in the sources (Qualcomm tooling)."
   - outline.md line 268: "YOLO rows exist only in van Baalen 2303.17951 (Qualcomm tooling, not TensorRT)".
   - Source: OBC (2208.11580v2) Table 1 has YOLOv5l COCO mAP@0.5 rows: "YOLOv5l -- 66.97" dense; GMP / L-OBS /
     AdaPrune / ExactOBS at 2x/3x/4x FLOP reduction (e.g. ExactOBS 66.14 / 65.35 / 64.05). OBC Table 7 has
     YOLOv5s ExactOBS runtimes. These are already claims `obc-t1-*` and `obc-runtime-yolov5s-quant`.
   - outline.md line 221 itself says "OBC has YOLOv5 rows", so the outline contradicts itself.
   - What this check proves: "only in van Baalen" is false. Among the 12 papers checked here, OBC is the one
     with YOLO rows, and they are pruning rows. The other 23 sources were not checked for YOLO quantization
     rows, so no "only" statement is safe without that check.
2. **OBC YOLO quantization.** outline.md line 221: "OBC has YOLOv5 rows (pruning, quantization + 2:4)".
   - Source: the YOLO quantization + 2:4 results are curves only: main-text figure `fig:yolo-mixed` ("YOLO
     variants") and appendix figure `fig:mixed-comp-yolo` ("Mixed quantization and 2:4 pruning for various BOP
     reduction targets on YOLO models"). OBC Table 4 (quantization) has ResNet18/50 only.
   - Say "pruning rows (Table 1); quantization + 2:4 as figure curves".
3. **Wang 2019 and FPN.** understanding.md line 17: "Wang 2019 imitates FPN features only near objects
   (full-map imitation hurts, Table 1)".
   - Wang Sec. 3.1: "with feature maps, we always refer to the last features where anchor priors are defined on".
   - FPN multi-level imitation is one experiment (Table 5, "multi-layer imitation"). Table 1 is a ShuffleNet toy
     detector on KITTI with a single feature map.
   - Say "imitates the anchor-level feature map near objects (one level; FPN levels in Table 5)".
4. **"20x+".** outline.md line 184: "a few activation channels that are 20x+ larger than the rest".
   - Dettmers: "magnitudes up to 20x larger" (Sec. 1) and "3-20x larger than the largest magnitude of other
     feature dimensions, which usually have a range of [-3.5, 3.5]" (Table 4 caption).
   - SmoothQuant: "~100x larger than most of the activation values" (Sec. 3) and "greater than 70" (Fig. 4).
   - "20x+" merges the two papers. Lines 194-195 of the outline already keep them apart. Fix the intuition line
     to match.
5. **KL argument order.** outline.md line 67: "loss = (1-a) CE(y, q^S) + a T^2 KL(q^T || q^S)", called "DeiT
   Eq. 2's form" on line 77.
   - DeiT Eq. 2 as printed: "(1-\lambda) L_CE(psi(Z_s), y) + \lambda \tau^2 KL(psi(Z_s/\tau), psi(Z_t/\tau))".
     The student comes first.
   - Hinton uses cross-entropy with soft targets, not KL (Sec. 2).
   - KL(teacher || student) is the usual direction and has the same student gradient as Hinton's
     cross-entropy. But it is not what DeiT prints. Label the direction "our reading" or print DeiT's order.
6. **LLM.int8() slowdown threshold (precision, not a contradiction).** understanding.md line 55 and outline.md
   line 196: "slower than FP16 below 13B-scale layers".
   - App. Table 5 supports this: 6.7B 0.86x, 13B 1.22x.
   - Main text Sec. 3.4 says: "The quantization overhead can slow inference for models with less than 6.7B
     parameters".
   - Cite Table 5 (GPU not stated) for the 13B threshold, not the prose.

Checked and consistent: the GPTQ toy recipe (outline line 223: H = 2XX^T + lambda I, lambda = 1% of mean
diag, upper Cholesky factor of H^-1, err / U_jj); "175B in ~4 GPU hours" (Table 2: 4.2h OPT, 3.8h BLOOM);
3-bit single-A100 with "the model would not fully fit using 4 bits"; AWQ Table 1, Eq. 1-5, grid size 20 over
[0, 1], group 128, Pile calibration; OpenFlamingo Table 6 and VILA Tables 7/10; LLM.int8() alpha = 6.0,
0.1%, 6.7B emergence, the perplexity-not-size argument; SmoothQuant Eq. 2-4, alpha 0.5/0.75, O1-O3, Table 3/4
rows, 1.51x/1.56x; Hinton Eq. 1-4, T^2, MNIST 67/146/74 at T = 20, speech Table 1, 3% data Table 5,
"condiderably", BMW/garbage truck/carrot; Wang psi = 0.5, lambda code-only 0.01, Table 1 -F/-D rows, Table 5
31.2 -> 34.8 (teacher 36.9); DeiT-B 81.8 -> 83.4; FitNets tables and appendix hyper-parameters; PTQ4ViT has no
detection results.

## 5. Remaining doubts

- Pre-existing `uncertain` claims (plot readings; kept): `dettmers-fig1-8bit-175b` 0.31,
  `dettmers-fig1-16bit-175b` 0.76, `dettmers-fig1-8bit-6.7b` 0.45. The rendered Fig. 1 agrees to about +-0.01:
  8-bit 175B about 0.315; 16-bit 175B about 0.765; 8-bit 6.7B about 0.455.
- Also kept uncertain: `xiao-fig4-ridge-count` 3, `xiao-fig4-channel-axis` 5000, `xiao-fig4-token-axis` 300,
  `xiao-fig4-floor` 1, `xiao-fig4-smoothed-ridge` 2, `xiao-fig4-weight-axis` 5 and `xiao-fig10-w8a8-low-alpha`
  0. The rendered figures agree with each reading. They stay plot readings.
- AWQ Fig. 9 caption "up to 3.9x" on 4090 (`awq-f9-4090-max`, `awq-f9-4090-range`) cannot be reproduced from
  the bar labels. The best ratios are 194/52 = 3.73 (Llama-2-7B) and 124/33 = 3.76 (Falcon-7B). The claim
  quotes the caption correctly. If the page shows both the bars and "3.9x", say the bars give about 3.8x.
- AWQ Fig. 3 caption says "Llama-2-7B on NVIDIA RTX 4090". Sec. 4.1 says it profiles "LLaMA-7B". The
  `awq-f3-*` claims follow the caption.
- Paper-internal inconsistencies, already captured as separate claims: AWQ Fig. 2 (43.2 / 13.0) vs Tables
  1/3 (23.54 / 11.39); AWQ RTN OPT-1.3B 119.00 (Table 1) vs 119.47 (Table 3); AWQ 30 vs 33 tok/s; Wang +6.7
  (table) vs 6.6 (prose).
- Inconsistencies with no separate claim:
  - GPTQ OPT-125M FP16 is 27.65 in Table 3 and 27.66 in Table 8.
  - GPTQ Table 6 prints 3.24x; the prose says "about 3.25x".
  - LLM.int8() Sec. 4.2 gives "the maximum 16-bit value 65535". The real FP16 maximum is 65504. No claim uses
    it; do not quote it.
- Hinton Sec. 4 prose calls 58.9% / 10.9% "development set" numbers. Table 1 heads the column "Test Frame
  Accuracy". The claim already notes this.
- AWQ Table 10 throughput: the unit is not printed. tokens/s is our reading (now labelled in the claims).
