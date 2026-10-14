# tensorrt -- NVIDIA TensorRT documentation (non-arXiv primary source)

## 1. Identity
- NVIDIA TensorRT docs, "latest" = 11.4.0, fetched 2026-10-09 (sources/tensorrt/*.txt are text copies; *.html originals).
- Cite as: page path + section heading, e.g. "architecture/how-trt-works, The Build Phase".
- Pages read: architecture_how-trt-works, architecture_architecture-overview, performance_optimization, performance_fusion-catalog, performance_best-practices, inference-library_{capabilities, precision-control, accuracy-considerations, work-with-quantized-types, quantized-types-workflows, quantized-types-schemes, quantized-types-explicit-quantization}, getting-started_{release-notes-11_11.0.0, support-matrix}, api_migration_tensorrt-10x-to-11x.
- ASCII note: quotes below replace curly apostrophes, arrows and en dashes with ASCII; no other change.

## 2. Core problem
Turn a trained network into the fastest correct GPU program for one GPU model and one TensorRT version, at the precision the model specifies.

## 3. What came before and why it changed (per the docs)
TensorRT 10.x "weak typing": builder flags (kFP16, kINT8, ...) let TensorRT pick lower precision per layer, and INT8 scales came from calibrators (implicit quantization). 11.0.0 removed both. Migration guide, "Strongly Typed Networks Are Now Required": "While this made reduced-precision kernels available without per-layer types, it could compromise accuracy and did not provide adequate control to model authors."

## 4. The one key idea
A build phase compiles the network once for a target GPU: fuse layers, then time candidate kernels ("tactics") on that GPU and keep the fastest. Precision is no longer chosen by TensorRT: since 11.0 the model states every tensor type, and quantization is explicit Q/DQ nodes that TensorRT fuses into low-precision kernels.

## 5. Mechanism, step by step (build phase)
1. Two phases. how-trt-works, intro: "A build phase in which a builder compiles your network and selects the fastest available kernel (a low-level GPU function) for each layer on your target GPU. The output is a serialized binary called an engine (also referred to as a plan file)." "A runtime phase in which a runtime loads the engine into your application and executes it on the GPU."
2. Graph rewrites. capabilities, The Build Phase: "The builder eliminates dead computations, folds constants, reorders, and combines operations to run more efficiently on the GPU. It also has multiple implementations for each layer with varying data formats. Then, it computes an optimal schedule to execute the model, minimizing the combined cost of kernel executions and format transforms."
3. Fusion first. fusion-catalog, Layer Fusion: "In the first phase, layers are fused whenever possible. Fusions transform the network into a simpler form but preserve the same overall behavior." Fused layers get names like "ip1 + relu1" or "fusedPointwiseNode(add1, relu1)".
4. Weight folding. how-trt-works, Memory > The Build Phase: "When TensorRT combines weights, such as convolution with batch normalization, additional temporary weight tensors are created."
5. Tactic selection by timing. how-trt-works, Determinism: "The TensorRT builder uses timing to find the fastest kernel to implement a given layer. Timing kernels are subject to noise, such as other work running on the GPU and GPU clock speed fluctuations." optimization, Deterministic Tactic Selection: "TensorRT runs through all the possible tactics in the engine-building phase and selects the fastest ones." Each tactic runs "for at least four iterations" and the latency is averaged.
6. Internal formats are part of tactic selection. capabilities, Tensors and Data Formats: "The builder chooses internal formats as part of tactic selection; applications cannot set them directly."
7. Timing cache. optimization, Reducing Engine Build Time > Timing Cache: "The information it contains is specific to the targeted device, CUDA, TensorRT versions, and BuilderConfig parameters that can change the layer implementation". Reusing one cache across builds makes TensorRT pick identical tactics (Deterministic Tactic Selection > Using Timing Cache).
8. Search effort knob: builder optimization level, default 3; lower = faster build, possibly slower engine.
9. Engine is tied to GPU + version. capabilities, Build Phase note: "By default, TensorRT engines are specific to both the TensorRT version and the GPU on which they were created." how-trt-works, Compatibility: "By default, serialized engines are only guaranteed to work correctly when used with the same OS, CPU architectures, GPU models, and TensorRT versions used to serialize the engines."
10. Cross-major versions. support-matrix, Engine Portability: "TensorRT 10 version-compatible engines can run with a TensorRT 11 runtime, and TensorRT 8 version-compatible engines can run with a TensorRT 10 runtime. No other combination across major versions is supported. For example, a TensorRT 8 engine cannot run with a TensorRT 11 runtime."
11. Do not build on a busy GPU: "Running the builder in parallel with other GPU work can perturb the timings, resulting in poor optimization." (capabilities, Build Phase note).

## 6. Equations (quantized-types-schemes, verbatim TeX from the .txt)
- INT8: `x_{q}=quantize\left(x, s \right)=roundWithTiesToEven(clip(\frac{x}{s}, -128,127))`; `{x}=dequantize\left(x_{q}, s\right)=x_{q}\ast s`. s = scale, FP16 or FP32 scalar. Symmetric: no zero-point.
- FP8: `x_{q}=quantize\left(x, s \right)=castToFp8(clip(\frac{x}{s}, -448,448))`; castToFp8 "rounds to the nearest value representable in FP8E4M3, ties are rounded to an even number".
- INT4: `roundWithTiesToEven(clip(\frac{x}{s}, -8,7))`, per-block scales, block sizes {64, 128}.
- NVFP4: `castToFp4(clip(\frac{x}{s}, -6,6))`, block size 16.
- Dynamic per-block scale: `{scale}=max_{i\in \left\{ 0...blockSize-1 \right\}}\left( \frac{abs\left( x_{i} \right)}{qTypeMax} \right)` (quantized-types-workflows, Dynamic Quantization).
- Commuting rule (explicit-quantization, TensorRT Processing of Q/DQ Networks): `{Op}` commutes with quantization if `{Q}\left(Op\left(x \right) \right)=={Op}\left(Q\left(x \right) \right)`; with dequantization if `{Op}\left(DQ\left(x \right) \right)=={DQ}\left(Op\left(x \right) \right)`.
- Per-channel conv weights (quantized-types-workflows, Quantization Granularities): `output[k,c,r,s] := clamp(round(input[k,c,r,s] / scale[k]))`, K = output-channel axis (required axis for conv).
- Symmetric statement (work-with-quantized-types): "TensorRT uses a symmetric quantization scheme, where both activations and weights are mapped to quantized values centered around zero."

## 7. Facts tables

### 7a. Precision support by compute capability, TensorRT 11.4.0
Source: support-matrix.html inline JS `supportData['11.4.0'].gpuPrecision` (the block the "GPU Architecture and Precision Support for TensorRT 11.4.0" explorer renders via `getVersionData('11.4.0')`). The first gpuPrecision block in the file (offset ~105918) is 11.3.0; do not cite it. The .txt lost this table; the data is in the HTML, no fetch needed.
| CC | Listed device | TF32 | FP32 | FP16 | BF16 | INT8 | FP8 | FP4 |
|---|---|---|---|---|---|---|---|---|
| 7.5 | T4 | N/A | yes | yes | N/A | yes | N/A | N/A |
| 8.0 | A100 | yes | yes | yes | yes | yes | N/A | N/A |
| 8.6 | A10 | yes | yes | yes | yes | yes | N/A | N/A |
| 8.9 | L40S | yes | yes | yes | yes | yes | yes | yes (f3) |
| 9.0 | H100, GH200 480 GB | yes | yes | yes | yes | yes | yes | yes (f3) |
| 10.0 | B200 | yes | yes | yes | yes | yes | yes | yes |
| 10.3 | B300 | yes | yes | yes | yes | yes | yes | yes |
| 10.7 | Vera Rubin | yes | yes | yes | yes | yes | yes | yes |
| 11.0 | DRIVE AGX Thor | yes | yes | yes | yes | yes | yes | yes |
| 12.0 | RTX PRO 6000 Blackwell | yes | yes | yes | yes | yes | yes | yes |
| 12.1 | DGX Spark | yes | yes | yes | yes | yes | yes | yes |
(12.1 also has a "Windows on ARM DLA" row: FP16 + DLA only.) f3 = "Supported in hardware emulation mode (hardware does not accelerate FP4 linear operations)." Feature table: "Native FP4 execution requires NVIDIA Blackwell architecture GPUs (SM 100+)." Floor: "TensorRT 11.4.0 supports NVIDIA hardware with compute capability SM 7.5 or higher."
- INT4 weight-only: no gpuPrecision column. Feature table: introduced 10.0.0 EA, "GEMM weight-only quantization; verify the target GPU in the precision table and quantization guide." Per-CC support: not verifiable here.
- L4 is not a listed device. Only L40S stands for CC 8.9. Any L4 statement needs another source.
- Structured 2:4 sparsity: not on any saved page (searched all .txt and .html). Only trace: Table 21 "FP16: 8 for dense math, 16 for sparse math" alignment. Not verifiable from the saved set.

### 7b. Quantized types (quantized-types-schemes, Table 14; work-with-quantized-types)
| | INT8 | FP8 | MXFP8 | INT4 | NVFP4 |
|---|---|---|---|---|---|
| Representation | 8-bit 2's complement | S1E4M3 | S1E4M3 | 4-bit 2's complement | S1E2M1 |
| Weights | per-tensor / per-axis | per-tensor / per-axis | per-block (32) | per-block ({64, 128}) | per-block (16) |
| Activations | per-tensor | per-tensor | dynamic per-block (32) | No | dynamic per-block |
| Scale dtype | FP32, FP16, BF16 | FP32, FP16, BF16 | E8M0 | FP32, FP16, BF16 | FP32, FP16, BF16 |
- Weight-only: INT4 only ("TensorRT only supports INT4 for weight quantization"; "WoQ is available only for INT4 block quantization with GEMM layers"). FP4 block quantization covers weights and activations.
- FP8 in TensorRT = E4M3 only (ONNX E4M3FN, opset 19). E5M2 is not a TensorRT quantized type.
- Rounding: round-to-nearest-even everywhere ("2.5 rounds to 2 and 3.5 rounds to 4", Quantized Types Rounding Modes).
- UINT8 / asymmetric quantization: GPU path no; DLA only (capabilities, Supported Types).
- Recommended granularity (explicit-quantization, Q/DQ Layer-Placement Recommendations): "Use per-tensor quantization for activations and per-channel quantization for weights. This configuration has been demonstrated empirically to lead to the best quantization accuracy."

### 7c. Fusion catalog (performance/fusion-catalog, Types of Fusions table + Supported Layer Fusions)
ReLU+ReLU; Conv+ReLU (any conv); Conv+GELU (FP16 or INT8 I/O, same in/out precision, Turing+); Conv+Clip; Scale+Activation; Conv+ElementWise (sum/min/max; no broadcast except across batch); Padding+Conv/Deconv (non-negative padding); Shuffle(permute only)+Reduce (keepDimensions); Shuffle+Shuffle; Scale erased if identity; Conv+Scale (kUNIFORM or kCHANNEL; disabled for non-constant power) "Adjusting the convolution weights can fuse"; Conv+generic pointwise activation (after pointwise fusion); Reduce(avg over HW)->Pooling; Conv(+act)+Pooling (same precision); Depthwise separable conv (dw conv+act -> conv+act; INT8 only; CC >= 7.2); Softmax+Log; Softmax+TopK; reduction fusions GELU, L1Norm, Sum of Squares, L2Norm, LogSum, LogSumExp.
Pointwise fusion: adjacent Activation, single-value Constant, ElementWise, Pointwise, kUNIFORM Scale, Unary layers merge into one Pointwise layer (size limited).
Why fusion helps (optimization, Enabling Layer Fusion): removes kernel-launch overhead "(typically 5-15 us per launch on the host)"; "Avoids materializing intermediate tensors in DRAM"; unlocks "fused Conv+ReLU, fused Conv+Bias+Activation, or pointwise super-kernels".
BN folding: no catalog entry named BatchNorm. Evidence: (a) how-trt-works "combines weights, such as convolution with batch normalization"; (b) Conv+Scale (kCHANNEL) fusion by adjusting conv weights; (c) explicit-quantization "Do not simulate batch normalization and ReLU fusions in the training framework because TensorRT optimizations guarantee the preservation of these operations' arithmetic semantics."

### 7d. Q/DQ processing (inference-library/quantized-types-explicit-quantization)
1. "TensorRT expects a Q/DQ layer pair on each input of quantizable layers." Quantizable layers fuse with a DQ input and a Q output and are replaced by "quantized layers that operate on quantized data".
2. Q/DQ propagation: "TensorRT propagates Quantize nodes backward (so quantization happens as early as possible) and Dequantize nodes forward (so dequantization happens as late as possible)." Layers that commute (max pooling) swap places; non-commuting quantizable layers (average pooling) use Q/DQ fusion.
3. Weights: high-precision weights are quantized at build time with the IQuantizeLayer scale; "The quantized (low-precision) weights are stored in the engine plan file."
4. Placement rules: "Quantize all inputs of weighted operations (Convolution, Transposed Convolution, and GEMM)." "By default, do not quantize the outputs of weighted operations." "Quantize the residual input in skip connections. TensorRT can fuse element-wise addition following weighted layers ... The precision of the first input to the element-wise addition layer determines the fusion output's precision." If that input is INT8, "the output of the fused convolution is also INT8, and the trailing Q-layer is fused with the convolution."
5. "Performance can decrease if TensorRT cannot fuse the operations with the surrounding Q/DQ layers, so be conservative when adding Q/DQ nodes".
6. Older GPUs may lack low-precision kernels ("could not find any implementation"); fix by removing those Q/DQ nodes and re-validating accuracy.
7. Not supported: pre-quantized ONNX ops QLinearConv, QLinearMatmul, ConvInteger, MatmulInteger.
8. Calibration happens before ONNX export: "Calibration is performed before ONNX export (for example, during PTQ or QAT with Model Optimizer), and the resulting scales are embedded directly into the Q/DQ nodes." (quantized-types-workflows).

## 8. TensorRT 11.0 removals (exact sentences)
- release-notes 11.0.0, Announcements: "Strongly typed networks are now the default: createNetworkV2() produces a strongly typed network by default in TensorRT 11.0.0. Weak typing is no longer supported."
- release-notes 11.0.0, Deprecated and Removed Features: "Implicit quantization has been removed. The IInt8Calibrator class and all subclasses (IInt8EntropyCalibrator, IInt8EntropyCalibrator2, IInt8MinMaxCalibrator, IInt8LegacyCalibrator), setInt8Calibrator(), setDynamicRange(), resetDynamicRange(), and all related calibration and dynamic-range APIs have been removed. Use explicit quantization with Q/DQ nodes instead. Models can be quantized using the TensorRT Model Optimizer or by manually inserting Q/DQ nodes."
- same section: "All weak-typing APIs have been removed. ITensor::setType, ITensor::setDynamicRange, ILayer::setPrecision, ILayer::setOutputType, INormalizationLayer::setComputePrecision, and the per-precision BuilderFlag values (kFP16, kBF16, kFP8, kINT8, kINT4, kFP4, kOBEY_PRECISION_CONSTRAINTS, kPREFER_PRECISION_CONSTRAINTS) are no longer available."
- release-notes 11.0.0, Breaking ABI Changes: "TensorRT 11.0 removed weak typing APIs along with several other deprecated APIs."
- release-notes 11.0.0, trtexec: "The --stronglyTyped flag has no effect in TensorRT 11.0.0+, since strongly typed networks are now the default."
- migration 10x-to-11x, Strongly Typed Networks Are Now Required: "TensorRT 11.x removes all weak typing APIs." ... "Similarly, implicit quantization is not supported in TensorRT 11.x. To quantize a model, you must add quantize and dequantize layers to the network." ... "For most models, use ModelOpt to apply quantization effectively." FP16 replacement: ModelOpt AutoCast ("python -m modelopt.onnx.autocast --onnx_path model.onnx"); INT8/FP8 PTQ: "python -m modelopt.onnx.quantization --onnx_path model.onnx --calibration_data data.npz --output_path model_quantized.onnx".
- precision-control, Strongly Typed Networks: "Since 11.0, TensorRT supports only Strongly Typed networks, meaning that the model must specify the precision for all tensors explicitly." "TensorRT will still autotune over different data layouts to find an optimal set of kernels for the network."
- explicit-quantization, intro: "Use explicit quantization with Strong Typing. Precision-control build flags are not required and should not be specified."
- capabilities, Strong Typing vs Weak Typing: "Weak typing was removed in 11.0."
- architecture-overview, Complementary Software: Model Optimizer "Replaces the deprecated PyTorch and TensorFlow Quantization Toolkits."
- Also removed in 11.0: 16 pre-10 plugins incl. BatchedNMS_TRT, BatchedNMSDynamic_TRT, EfficientNMS_ONNX_TRT, NMS_TRT, NMSDynamic_TRT (replacements mostly addNMS() etc.); tactic sources kCUBLAS, kCUBLAS_LT, kCUDNN ("TensorRT no longer uses these external libraries for core operations"); static libraries. APIs deprecated in 11.0.0 retained until March 2027.

## 9. Accuracy advice and stated limitations
- accuracy-considerations, Mitigation Strategies: "Perform critical, precision-sensitive calculations in FP32 and use reduced precision for less sensitive operations"; "Linear operations are particularly good candidates for reduced precision". Sensitive examples: "Sigmoid or Softmax amplify small numerical differences due to their exponential component"; IReduceLayer prone to FP16 overflow; FP16 max 65,504.
- accuracy-considerations, Quantized Data Types: FP8E4M3 / FP4E2M1 "have non-uniform distributions with more values concentrated near 0"; clamping vs rounding tradeoff "expressed in the scale selection".
- optimization, Tradeoffs Between Accuracy and Performance: layer constraints "such as LayerNorm output should not be INT8"; use Polygraphy to compare layer outputs; "An experimental debug precision tool can help automatically find layers to run with high precision"; a bad tactic can change accuracy between builds; fix with the Editable Timing Cache.
- precision-control: FP32 accumulation is requested by casting around the GEMM (ICastLayer); "the accumulation type is the input type or a wider type, at the whim of the optimizer."
- release-notes 11.0.0, Limitations: "The FP8 Convolutions on GPUs with SM89/90/120/121 do not support kernel sizes larger than 32 (for example, 7x7 convolutions); FP16 or FP32 fallback kernels will be used with suboptimal performance. Therefore, do not add FP8 Q/DQ ops before Convolutions with large kernel sizes for better performance."
- release-notes 11.0.0, Known Issues (Performance): "A non-zero tilingOptimizationLevel might introduce engine build failures for some networks on L4 GPUs." ONNX Runtime <= 1.24.4 TensorRT EP does not compile against 11.0 (uses removed setInt8Calibrator etc.); ORT 1.27+ adopts 11.x.
- Stale text inside the 11.4.0 docs (flag, do not cite as current): explicit-quantization says "The latest PyTorch version (PyTorch 2.0) does not support FP8 formats"; optimization "Tradeoffs Between Accuracy and Performance" still says "By default, TensorRT chooses to run a layer in a precision that results in optimal performance" and "consider recalibrating", which conflicts with strong typing; how-trt-works Table 10 still lists "calibrator pointer" in IBuilderConfig state.

## 10. Outline feedback
- ch-tensorrt viz-fuse steps are all sourced: BN fold (how-trt-works + Conv+Scale), conv+bias+ReLU (Enabling Layer Fusion; Conv+ReLU), Q/DQ absorbed into an INT8 conv (explicit-quantization, Q/DQ pair on each input), residual add fused when its first input is INT8 (placement rules). Label the residual rule exactly: the first input's precision decides the fused output precision.
- ch-tensorrt "implicit quantization (calibrators) removed in 11.0; explicit Q/DQ via Model Optimizer": correct; quote section 8 above. Also say "weak typing removed" (kFP16/kINT8 builder flags gone): even an FP16 engine now needs typed tensors (AutoCast).
- ch-guide order "FP16 engine -> INT8 PTQ ..." needs a TensorRT 11 footnote: an FP16 engine means converting tensor types (ModelOpt AutoCast or Cast nodes), and INT8 PTQ means ModelOpt calibration that writes Q/DQ into ONNX; TensorRT itself no longer calibrates. Our reading for the deployment context (state.json): an entropy-calibrator trtexec flow on TensorRT 8.5 (23.02 image) has no TensorRT 11 equivalent, and a TensorRT 8 engine cannot run on a TensorRT 11 runtime.
- ch-guide GPU widget: sourced rows are T4 = CC 7.5 (FP16/INT8 only, no BF16, no FP8); A100/A10 = no FP8; L40S/H100 = FP8 yes, FP4 emulated only; Blackwell (10.0+) native FP4. The outline's "FP8 on Ada/Hopper+" matches (L40S 8.9, H100 9.0). L4 is not in the table; "2:4 on Ampere+" is not sourced by these TensorRT pages (use the Mishra 2021 source or mark it).
- ch-guide: add the FP8 conv limitation (kernel > 32, e.g. a 7x7 stem, falls back to FP16/FP32 on SM89/90/120/121).
- ch-cost roofline: no saved TensorRT page gives tensor-core throughput or memory bandwidth numbers, so the FP16:INT8 = 1:2 roof stays a labelled toy.
- Claims trt-int4-woq-per-cc, trt-sparsity-gpu and trt-l4-cc are uncertain markers ("no page claim available"); their value strings are placeholders, not text to print.
- ch-grid note "TensorRT uses symmetric (Z = 0) INT8": correct (work-with-quantized-types). INT8 range in TensorRT is [-128, 127], not [-127, 127].
- ch-fp8 "TensorRT support matrix (which compute capabilities run FP8)": use 7a. TensorRT's FP8 is E4M3 only; E5M2 is not a TensorRT quantized type.
- ch-granularity: TensorRT recommends per-channel weights + per-tensor activations for INT8/FP8; per-block only for INT4/FP4/MXFP8.
