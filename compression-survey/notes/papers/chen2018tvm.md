# chen2018tvm -- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (BRANCH)

## 1. Identity
- arXiv 1802.04799, read v3 (LaTeX `src/TVMTechReport.tex` + `paper.pdf`). Dates (abs.html): v1 2018-02-12, v3 2018-10-05.
- Authors (abs.html order): Tianqi Chen ... Arvind Krishnamurthy (12 authors; UW, AWS, SJTU, UC Davis, Cornell). Venue not stated in abs.html or the v3 PDF.

## 2. One key idea (two sentences)
A compiler that applies graph-level rewrites (operator fusion, constant folding, static memory planning, data layout transforms) and then generates each fused operator's code itself, instead of relying on a vendor library that must ship every fused pattern. It searches each operator's schedule space (tiling, unrolling, ordering) with a learned cost model that predicts which configurations run fastest, and measures only the promising ones on real hardware.

## 3. Fusion categories and rules (Sec 3, para "Operator Fusion", verbatim)
"we recognize four categories of graph operators: (1) injective (one-to-one map, e.g., add), (2) reduction (e.g., sum), (3) complex-out-fusable (can fuse element-wise map to output, e.g., conv2d), and (4) opaque (cannot be fused, e.g., sort)."
"Multiple injective operators can be fused into another injective operator. A reduction operator can be fused with input injective operators (e.g., fuse scale and sum). Operators such as conv2d are complex-out-fusable, and we can fuse element-wise operators to its output."
Why it matters: "Operator fusion combines multiple operators into a single kernel without saving the intermediate results in memory." Library-based fusion does not scale: "the required number of fused pattern implementations grows combinatorially with the number of data layouts, data types, and accelerator intrinsics" (Sec 3).

## 4. Learned cost model (Sec 5.2)
- Schedule explorer proposes configurations; an ML model "takes the lowered loop program as input and predicts its running time on a given hardware back-end"; trained on measurements collected during the search, updated periodically; "does not require the user to input detailed hardware information".
- Trained with a rank objective ("we need not predict the absolute execution times directly").
- Default model: gradient tree boosting (XGBoost) on loop features (memory access count, reuse ratio per buffer per loop level, one-hot loop annotations); TreeRNN on the AST had similar quality, but tree boosting "performs prediction twice as fast".
- Search: parallel simulated annealing guided by the model; top candidates measured on devices via RPC (Sec 5.3-5.4).
- Table 1 contrasts blackbox auto-tuning (high data cost), predefined cost model (high model bias, needs hardware info), ML cost model (low/low, learns from history).

## 5. Key numbers
- Fusion: "fused operators generate up to a 1.2x to 2x speedup by reducing memory accesses" (Sec 3; Fig 4, NVIDIA Titan X; workloads conv+bn+relu 128x28x28 1x1x128x256, depthwise-conv+bn+relu 512x14x14 3x3x512, RNN cell and LSTM cell hidden 128). Batch size not stated.
- End-to-end GPU: speedups "ranging from 1.6x to 3.8x" over MXNet v1.1, TensorFlow v1.7 and TensorFlow XLA on NVIDIA Titan X; baselines use cuDNN v7 and cuBLAS v8; DQN gets 3.8x from unconventional 4x4 stride-2 conv2d (Sec 6.1, Fig 14). Batch size not stated.

## 6. Outline feedback
- ch-tensorrt "TVM as the open-source analogue (fusion rules, learned cost model)": correct framing. The TVM rule "complex-out-fusable + element-wise on its output" is the same idea as TensorRT's Conv+ReLU / Conv+generic-activation fusion; TensorRT times tactics on the target GPU, while TVM predicts with a learned model and times only the top candidates. Use the 1.2x to 2x fusion number with "Titan X" and say batch size is not stated.
