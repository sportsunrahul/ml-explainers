# Fork F: ch-fp8, ch-tensorrt, ch-guide, kits 80/81/90, atlas fp8 + guide

All 151 data-claim spans in the three chapters match their claim values. Quotes from the TensorRT docs,
Micikevicius, van Baalen and TVM were found verbatim in sources/ (dash and "us" ASCII-fied). Section,
table and figure numbers checked: Micikevicius Sec. 2, 3.1, 4.3, Tables 1/2/5; van Baalen Sec. 2, 4.1,
Fig. 3, Fig. 5, Tables 1/2/6; Wu Tables 5/7/9; Nagel Tables 6/10; BRECQ Table 5; Mishra Table 4 (15 rows);
TVM Sec. 3. Toy byte counts in 81-fuse.js recomputed: 8.59 -> 7.03 -> 6.25 -> 3.91 MiB; BN fold saves
1.56 MiB, matching the prose. FP8 readouts match notes/sim-fp8.txt.

## Must-fix

1. parts/kit/90-guide.js (KIT.guideSteps, FP8 bullet): "TensorRT falls back to higher precision for FP8
   convolutions with kernels larger than 32." The step shows for every CC >= 8.9, including CC 10.0 (B200).
   Source (release notes 11.0.0, Limitations) limits it to SM89/90/120/121; SM100 is not listed.
   FIXED: now "On SM89, SM90, SM120 and SM121, TensorRT falls back to FP16 or FP32 kernels for FP8
   convolutions with kernel sizes larger than 32 (for example 7x7)."
2. parts/ch-fp8.html: "The answer is E4M3, by a small margin." Our own computation (notes/sim-fp8.txt)
   gives MSE INT8 1.96e-3 vs E4M3 6.56e-4, a 3x gap (4.50 vs 5.29 bits). FIXED: "with about a third of
   INT8's mean squared error, which is less than one bit of accuracy more" (data-toy span).

## Nits

- ch-fp8.html: "Van Baalen et al. state the same ordering for their three test shapes". Their Fig. 5 also
  has E2/E3 (E2 is best on the normal). FIXED wording: "find the same winners among these formats".
- ch-fp8.html, ch-tensorrt.html: the scale is "stored in FP32, FP16 or BF16". The schemes page says "a
  16-bit or 32-bit floating point scalar"; BF16 is not named there. Not edited.
- ch-fp8.html: "E4M3 puts ... eight marks into every doubling". True for normal binades only; the subnormal
  binades have fewer. Not edited.
- 90-guide.js ptqLoss.cnn = 0.2 is a round-up of 0.13 (Wu Table 7 Faster R-CNN); the flowchart prints
  "about 0.2 points ... (our reading)". Acceptable.
- 90-guide.js: 2:4 bullet appears for every CC >= 8.0. The saved support matrix does not list sparsity
  support; the chapter says so. Our reading, not sourced.

## Open doubts

- ch-fp8.html "On a detector": "That is simulated quantization on their own tooling". Van Baalen do not say
  "simulated" for Table 1, nor name the tool (AIMET is named only for Table 6). Likely true; unsourced.
  Same for "Qualcomm tooling" in ch-guide Table E and the guide YOLO step.
- 90-guide.js: "FP8 pays off for outlier-heavy transformer layers". Van Baalen say FP8-E4 is more accurate
  there in PTQ, but recommend W8A16 or QAT as more efficient fixes (Sec. 8). Wording may overstate.
- ch-guide.html badge "Sourced": the figure is our ordered recipe with sourced numbers in it; the caption
  says the order and stops are our reading. "Sourced" is defensible but generous.

Test build: `bash build.sh parts/ORDER build/fc-F` ends PASS.
