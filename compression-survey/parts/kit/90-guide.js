/* Decision guide: our reading of the sourced evidence, turned into an ordered recipe. Choices: GPU compute capability,
   model family, accuracy budget (absolute points of mAP or mIoU), and whether fine-tuning is possible. Every number in
   the step text is a data-claim span; KIT.guideFlow draws the same steps as a flowchart. */
KIT.GUIDE = {
  gpus: { cc75: 'compute capability 7.5 (T4)', cc80: 'compute capability 8.0 or 8.6 (A100, A10)', cc89: 'compute capability 8.9 (L40S)', cc90: 'compute capability 9.0 (H100)', cc100: 'compute capability 10.0 or newer (B200)' },
  models: { cnn: 'ResNet-style CNN detector or segmenter', dw: 'depthwise-heavy model (EfficientDet, MobileNet)', yolo: 'YOLO-family detector', vit: 'transformer detector or ViT segmenter' },
  // Expected INT8 PTQ loss in absolute points, our reading of the evidence: cnn = worst best-calibrator row of Wu Table 7
  // (Faster R-CNN, 0.13), dw = EfficientDet-D1 per-channel W8A8 (Nagel Table 6); null = no sourced vision rows.
  ptqLoss: { cnn: .2, dw: 1.41, yolo: null, vit: null },
};
KIT.cl = (id, v) => `<span data-claim="${id}">${v}</span>`;
KIT.guideSteps = s => {
  const G = KIT.GUIDE, cc = +s.gpu.slice(2), fp8 = cc >= 89, sparse = cc >= 80, steps = [];
  const exp = G.ptqLoss[s.model], within = exp != null && exp <= s.budget;
  steps.push({ key: 'base', lever: 'measure', title: 'Measure FP32, then build an FP16 engine',
    text: 'Record accuracy on your own validation set and latency at your real batch size and input resolution. An FP16 engine is the usual first step; measure it too, since it is the baseline every later step is compared with.', ev: [] });
  let ptq = 'Quantize weights to INT8 per output channel and activations per tensor, using images that look like deployment data for calibration (Wu et al. used ' + KIT.cl('wu20-t9-faster-r-cnn-calib-samples', '512') + ' for their detectors). Try entropy and the 99.99 to 99.9999 percentiles, and keep the best on your validation set; plain max calibration usually loses more. In TensorRT ' + KIT.cl('trt-removal-version', '11.0.0') + ' and later there are no built-in calibrators: insert quantize and dequantize (Q/DQ) nodes, for example with NVIDIA Model Optimizer.';
  const ev2 = s.model === 'cnn' ? ['ev-wu'] : s.model === 'dw' ? ['ev-nagel'] : s.model === 'yolo' ? ['ev-yolo'] : [];
  if (s.model === 'dw') ptq += ' Depthwise-heavy networks lose more: EfficientDet-D1 drops from ' + KIT.cl('nagel21-t6-efficientdet-d1-fp32', '40.08') + ' to ' + KIT.cl('nagel21-t6-efficientdet-d1-pc-w8a8', '38.67') + ' mAP with per-channel W8A8. Apply cross-layer equalization first (chapter 6).';
  if (s.model === 'yolo') ptq += ' The only YOLO rows in these sources use Qualcomm tooling, not TensorRT: naive INT8 PTQ cost YoloV5 ' + KIT.cl('vb-t1-yolov5-int8', '-1.8') + ' mAP. Measure your own model.';
  if (s.model === 'vit') ptq += ' Transformer blocks have post-softmax, post-GELU and outlier-channel activations that a uniform per-tensor grid fits badly (chapter 10). No source here measures a transformer detector in INT8: measure, and expect to keep some layers in FP16.';
  steps.push({ key: 'ptq', lever: 'quant', title: 'INT8 post-training quantization', text: ptq, ev: ev2,
    stop: within ? `Expected loss is within your ${s.budget}-point budget: stop here if your measurement agrees.` : null });
  steps.push({ key: 'sens', lever: 'quant', title: 'Find the sensitive layers; keep them in FP16',
    text: 'Quantize one layer at a time, rank layers by the accuracy each one costs, and return the worst few to FP16 (Wu et al. call this partial quantization; Nagel et al. give a debugging flow). Depthwise layers are frequent offenders: in EfficientNet b0, ' + KIT.cl('wu20-fig3-depthwise-count', '7') + ' of the ' + KIT.cl('wu20-effb0-skip', '10') + ' most sensitive layers were depthwise.', ev: ['ev-nagel'] });
  if (s.train) steps.push({ key: 'qat', lever: 'quant', title: 'Short quantization-aware fine-tune',
    text: 'Start from the calibrated ranges and fine-tune with fake quantization and the straight-through estimator (chapter 7). For ResNet-style detectors it rarely beats good PTQ: Faster R-CNN scored ' + KIT.cl('wu20-t7-faster-r-cnn-ptq-acc', '36.82') + ' after PTQ and ' + KIT.cl('wu20-t7-faster-r-cnn-qat-acc', '36.76') + ' after QAT. It helps more where PTQ struggles: EfficientDet-D1 W4A8 per-channel goes from ' + KIT.cl('nagel21-t6-efficientdet-d1-pc-w4a8', '35.08') + ' (PTQ) to ' + KIT.cl('nagel21-t10-efficientdet-d1-pc-w4a8', '36.75') + ' (QAT).', ev: ['ev-wu', 'ev-nagel'] });
  else steps.push({ key: 'adv', lever: 'quant', title: 'No training? Use a stronger PTQ method',
    text: 'AdaRound and BRECQ need only calibration images. They matter most below 8 bits: with 4-bit weights, Faster R-CNN R50 keeps ' + KIT.cl('brecq21-t5-faster-rcnn-resnet-50-brecq-w4a8', '38.29') + ' mAP with BRECQ, but only ' + KIT.cl('brecq21-t5-faster-rcnn-resnet-50-bc-w4a8', '0.25') + ' with plain rounding and bias correction.', ev: ['ev-brecq'] });
  const more = [];
  if (sparse && s.train) more.push('2:4 sparsity with retraining (chapter 9): COCO detectors stayed within about a point of dense, for example RetinaNet-RN50 ' + KIT.cl('mishra2021-t4-retinanet-rn50-bbox-dense-fp16', '34.8') + ' dense vs ' + KIT.cl('mishra2021-t4-retinanet-rn50-bbox-sparse-int8', '35.1') + ' sparse INT8; the paper reports up to ' + KIT.cl('mishra2021-math-speedup', '2') + 'x math throughput, not an end-to-end speedup, so measure.');
  if (!sparse) more.push('This GPU is older than the A100 generation that Mishra et al. built 2:4 sparsity for; check its documentation before planning on sparse tensor cores.');
  if (fp8) more.push('FP8 is available from compute capability ' + KIT.cl('trt-fp8-min-cc', '8.9') + '. For CNNs van Baalen et al. find INT8 usually as accurate, and for outlier-heavy layers they recommend 16-bit activations (W8A16) or QAT rather than FP8 alone (chapter 12). On SM89, SM90, SM120 and SM121, TensorRT falls back to FP16 or FP32 kernels for FP8 convolutions with kernel sizes larger than ' + KIT.cl('trt-fp8-conv-kernel-limit', '32') + ' (for example 7x7).');
  if (s.train) more.push('Distill a smaller backbone (chapter 3) when even INT8 is too slow: imitating teacher features near objects lifted a halved Faster R-CNN student from ' + KIT.cl('wang-coco-fpn-r50h-ap', '31.2') + ' to ' + KIT.cl('wang-coco-fpn-r50h-i-ap', '34.8') + ' AP.');
  more.push('Do not expect 4-bit weight-only formats (GPTQ, AWQ) to speed up a convolutional detector: they help layers whose bytes are mostly weights (chapter 1).');
  steps.push({ key: 'more', lever: 'struct', title: 'Need more speed? Change the structure', text: more.map(t => '<span class="g-bullet">' + t + '</span>').join(' '), ev: ['ev-sparse', 'ev-other'] });
  return { steps, exp, within };
};
KIT.guideHTML = s => {
  const r = KIT.guideSteps(s), G = KIT.GUIDE;
  const head = `<p class="g-sum"><b>${G.models[s.model]}</b> on a GPU with ${G.gpus[s.gpu]}, budget ${s.budget} point${s.budget === 1 ? '' : 's'}, ${s.train ? 'fine-tuning possible' : 'no fine-tuning'}. Our reading of the evidence below:</p>`;
  return head + '<ol class="g-steps">' + r.steps.map(st => `<li data-step="${st.key}"><b>${st.title}.</b> ${st.text}` +
    (st.ev.length ? ` <span class="g-ev">Evidence: ${st.ev.map(e => `<a href="#${e}">${{ 'ev-wu': 'table A', 'ev-nagel': 'table B', 'ev-brecq': 'table C', 'ev-sparse': 'table D', 'ev-yolo': 'table E', 'ev-other': 'table E' }[e]}</a>`).join(', ')}.</span>` : '') +
    (st.stop ? `<span class="g-stop">${st.stop}</span>` : '') + '</li>').join('') + '</ol>';
};
KIT.guideFlow = (el, opts = {}) => {
  const S = { gpu: 'cc75', model: 'cnn', budget: 1, train: true, u: 1 };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const r = KIT.guideSteps(S), n = r.steps.length, narrow = w < 560, C = k => PE.css(k);
    const col = { measure: C('--blue'), quant: C('--yellow'), struct: C('--brown') };
    const top = 18, bot = opts.bottomPad || 0, rowH = (h - top - bot - 10) / n, bw = narrow ? w * .62 : Math.min(330, w * .42), x = narrow ? 12 : w * .08;
    r.steps.forEach((st, i) => {
      const y = top + i * rowH, bh = Math.min(46, rowH * .66), a = PE.clamp(S.u * n - i, 0, 1);
      if (i) PE.arrow(ctx, x + bw / 2, y - rowH + bh + 2, x + bw / 2, y - 2, C('--muted'), 1.5, 7, a);
      PE.box(ctx, x, y, bw, bh, col[st.lever], '', a);
      ctx.globalAlpha = a; ctx.font = `600 ${narrow ? 11 : 12}px system-ui, sans-serif`;
      const words = st.title.split(' '), lines = []; words.forEach(wd => { const cur = lines[lines.length - 1]; if (cur && ctx.measureText(cur + ' ' + wd).width < bw - 18) lines[lines.length - 1] = cur + ' ' + wd; else lines.push(wd); });
      lines.slice(0, 2).forEach((ln, k) => PE.label(ctx, ln, x + bw / 2, y + bh / 2 + 4 + (k - (Math.min(2, lines.length) - 1) / 2) * 14, C('--fg'), narrow ? 11 : 12, 'center'));
      ctx.globalAlpha = 1;
      if (st.key === 'ptq' && !narrow) PE.label(ctx, r.exp == null ? '' : `expected INT8 loss about ${r.exp} points vs budget ${S.budget} (our reading)`, x + bw + 16, y + bh / 2 + (st.stop ? 22 : 4), C('--muted'), 11, 'left', 400);
      if (st.stop) { const sx = x + bw + 14, sy = y + bh / 2; PE.arrow(ctx, x + bw + 2, sy, sx + (narrow ? 26 : 46), sy, C('--green'), 2 + 2 * glow('stop'), 8, a);
        PE.label(ctx, narrow ? 'stop?' : 'within budget: stop', sx + (narrow ? 30 : 52), sy + 4, C('--green'), 12, 'left'); }
    });
    if (r.exp == null) PE.label(ctx, 'no sourced INT8 rows for this family: measure', narrow ? 12 : x + bw + 16, narrow ? h - bot - 4 : top + rowH * 1 + 20, C('--red'), 11, 'left', 400);
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.05 : .5), draw);
  let stop = () => {};
  function set(next) { Object.assign(S, next); S.u = 0; stop(); stop = KIT.tween(S, { u: 1 }, 700, cv.redraw); }
  return { set, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
