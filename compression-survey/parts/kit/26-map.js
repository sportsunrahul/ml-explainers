/* Map chart: small multiples of KIT.SCAN, all titles as neutral bars, the chosen title subset overlaid. */
KIT.mapchart = (el, opts = {}) => {
  const S = { split: 'llm', u: 1 }, topics = ['distillation', 'quantization', 'pruning'], Y = KIT.SCAN.years;
  const prev = { split: 'llm' };
  function draw(ctx, w, h) {
    ctx.clearRect(0, 0, w, h);
    const narrow = w < 560, C = n => PE.css(n), fg = C('--fg'), mut = C('--muted'), p2 = C('--panel2'), rule = C('--rule');
    const n = topics.length, gap = narrow ? 34 : 26;
    const pw = narrow ? w - 52 : (w - 52 - gap * (n - 1)) / n, ph = narrow ? (h - 30 - gap * n) / n : h - 70;
    topics.forEach((t, k) => {
      const x0 = 44 + (narrow ? 0 : k * (pw + gap)), y0 = narrow ? 22 + k * (ph + gap) : 26;
      const all = KIT.SCAN.data[t].all, top = Math.max(...all) * 1.08, bw = pw / Y.length;
      const sub = KIT.SCAN.data[t][S.split], old = KIT.SCAN.data[t][prev.split];
      ctx.strokeStyle = rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, y0 + ph); ctx.lineTo(x0 + pw, y0 + ph); ctx.stroke();
      [0, .5, 1].forEach(f => { const v = Math.round(top * f / 100) * 100, y = y0 + ph - ph * v / top;
        if (f) { ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + pw, y); ctx.stroke(); ctx.globalAlpha = 1; }
        PE.label(ctx, String(v), x0 - 5, y + 4, mut, 10, 'right', 400); });
      Y.forEach((yr, i) => {
        const x = x0 + i * bw + bw * .14, bwi = bw * .72, ha = ph * all[i] / top;
        ctx.fillStyle = p2; ctx.fillRect(x, y0 + ph - ha, bwi, ha);
        if (yr === 2026) { ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = mut; ctx.strokeRect(x + .5, y0 + ph - ha + .5, bwi - 1, ha - 1); ctx.restore(); }
        const v = PE.lerp(old[i], sub[i], S.u), hs = ph * v / top;
        ctx.fillStyle = fg; ctx.fillRect(x, y0 + ph - hs, bwi, hs);
        if (i % (narrow ? 3 : 4) === 0 || i === Y.length - 1) PE.label(ctx, String(yr), x + bwi / 2, y0 + ph + 13, mut, 10, 'center', 400);
      });
      PE.label(ctx, t, x0, y0 - 8, fg, 12, 'left', 600);
    });
  }
  const cv = PE.canvas(el, opts.aspect || (w => w < 560 ? 1.35 : .42), draw);
  let stop = () => {};
  function set(next) {
    if (next.split && next.split !== S.split) { prev.split = S.split; S.split = next.split; S.u = 0; stop(); stop = KIT.tween(S, { u: 1 }, 650, cv.redraw); }
    cv.redraw();
  }
  return { set, state: S, redraw: cv.redraw, resize: cv.resize, destroy: () => { stop(); cv.destroy(); } };
};
KIT.mapText = split => {
  const i = KIT.SCAN.years.indexOf(2025), d = KIT.SCAN.data, name = { all: 'all titles', llm: 'titles that name an LLM or language model', dense_vision: 'titles that name detection or segmentation' }[split];
  return `In 2025, ${name}: ` + ['distillation', 'quantization', 'pruning'].map(t => `${t} ${d[t][split][i]} of ${d[t].all[i]}`).join(', ') + '. The dashed 2026 bars stop at 2026-10-09.';
};

/* Family tree: lanes by lever, time runs down (desktop) or one dated column (phones). Pills are buttons. */
KIT.TREE = {
  lanes: ['Distillation', 'Quantization, vision', 'Pruning', 'LLM era', 'Formats, compilers'],
  nodes: [
    ['bengio', 1, '1308.3432', 'STE', 'Bengio et al.: estimating gradients through hard thresholds; the identity straight-through estimator.'],
    ['fitnets', 0, '1412.6550', 'FitNets', 'Romero et al.: distill intermediate features ("hints") through a small regressor.'],
    ['hinton', 0, '1503.02531', 'Distillation', 'Hinton, Vinyals, Dean: train a student on the teacher\'s softmax at a raised temperature.'],
    ['han15', 2, '1506.02626', 'Prune + retrain', 'Han et al.: remove small weights, retrain the rest.'],
    ['dc', 2, '1510.00149', 'Deep Compression', 'Han, Mao, Dally: pruning, k-means weight sharing and Huffman coding in one pipeline.'],
    ['lifilt', 2, '1608.08710', 'Filter pruning', 'Li et al.: remove whole convolution filters by L1 norm, so the dense layer shrinks.'],
    ['slim', 2, '1708.06519', 'Network slimming', 'Liu et al.: an L1 penalty on BatchNorm scales picks channels to remove.'],
    ['jacob', 1, '1712.05877', 'Integer-only', 'Jacob et al.: scale and zero-point, integer-only matrix multiply, quantization-aware training.'],
    ['tvm', 4, '1802.04799', 'TVM', 'Chen et al.: an open deep-learning compiler with operator fusion and learned tuning.'],
    ['lth', 2, '1803.03635', 'Lottery ticket', 'Frankle, Carbin: sparse subnetworks that train well from their original initialization.'],
    ['pact', 1, '1805.06085', 'PACT', 'Choi et al.: learn the activation clipping value during training.'],
    ['krish', 1, '1806.08342', 'Whitepaper 2018', 'Krishnamoorthi: per-channel weights, the clipped STE, practical recommendations.'],
    ['rethink', 2, '1810.05270', 'Rethinking pruning', 'Liu et al.: a structurally pruned network trained from scratch does as well as fine-tuning.'],
    ['lsq', 1, '1902.08153', 'LSQ', 'Esser et al.: learn the quantizer step size with a scaled gradient.'],
    ['dfq', 1, '1906.04721', 'Data-free (CLE)', 'Nagel et al.: cross-layer equalization and bias correction without data.'],
    ['wang', 0, '1906.03609', 'Detector imitation', 'Wang et al.: distill a detector by imitating features near objects.'],
    ['lmc', 2, '1912.05671', 'Rewinding', 'Frankle et al.: at scale, reset to an early iteration instead of initialization.'],
    ['blalock', 2, '2003.03033', 'State of pruning', 'Blalock et al.: pruning results are hard to compare; no standard benchmarks.'],
    ['wu', 1, '2004.09602', 'NVIDIA INT8 study', 'Wu et al.: calibration methods, per-channel weights and QAT measured on detectors and segmenters.'],
    ['adaround', 1, '2004.10568', 'AdaRound', 'Nagel et al.: learn whether each weight rounds up or down.'],
    ['deit', 0, '2012.12877', 'DeiT', 'Touvron et al.: a distillation token for vision transformers with a convnet teacher.'],
    ['brecq', 1, '2102.05426', 'BRECQ', 'Li et al.: reconstruct block outputs to push post-training quantization to 4 bits and below.'],
    ['nm', 2, '2104.08378', '2:4 sparsity', 'Mishra et al.: two of every four weights zero, run on sparse tensor cores.'],
    ['nagel', 1, '2106.08295', 'White paper 2021', 'Nagel et al.: the standard PTQ and QAT pipelines and a debugging flow.'],
    ['ptq4vit', 1, '2111.12293', 'PTQ4ViT', 'Yuan et al.: grids for the post-softmax and post-GELU shapes in vision transformers.'],
    ['int8', 3, '2208.07339', 'LLM.int8()', 'Dettmers et al.: keep outlier feature dimensions in FP16, the rest in INT8.'],
    ['obc', 3, '2208.11580', 'OBC', 'Frantar et al.: exact second-order (OBS) compression, one weight at a time.'],
    ['fp8', 4, '2209.05433', 'FP8 formats', 'Micikevicius et al.: the E4M3 and E5M2 eight-bit floating-point formats.'],
    ['gptq', 3, '2210.17323', 'GPTQ', 'Frantar et al.: column-by-column rounding with Hessian error feedback, 3-4 bit weights.'],
    ['smooth', 3, '2211.10438', 'SmoothQuant', 'Xiao et al.: move activation outliers into the weights with a per-channel scale.'],
    ['sparsegpt', 3, '2301.00774', 'SparseGPT', 'Frantar, Alistarh: one-shot pruning with the same reconstruction machinery.'],
    ['fp8int8', 4, '2303.17951', 'FP8 vs INT8', 'van Baalen et al.: for inference, INT8 is often as accurate and cheaper in silicon.'],
    ['awq', 3, '2306.00978', 'AWQ', 'Lin et al.: protect the weight channels that see large activations by scaling them.'],
    ['wanda', 3, '2306.11695', 'Wanda', 'Sun et al.: prune by weight magnitude times input activation norm.'],
    ['trt', 4, null, 'TensorRT 11', 'NVIDIA TensorRT 11.4 docs: fusion, tactic selection, explicit Q/DQ quantization.'],
  ],
  edges: [['fitnets', 'wang'], ['hinton', 'wang'], ['hinton', 'deit'], ['bengio', 'jacob'], ['jacob', 'krish'], ['krish', 'wu'], ['pact', 'lsq'],
    ['dfq', 'nagel'], ['adaround', 'brecq'], ['adaround', 'nagel'], ['wu', 'nagel'], ['han15', 'dc'], ['dc', 'lth'], ['lth', 'lmc'], ['lifilt', 'rethink'],
    ['slim', 'rethink'], ['obc', 'gptq'], ['obc', 'sparsegpt'], ['gptq', 'sparsegpt'], ['int8', 'smooth'], ['dfq', 'smooth'], ['smooth', 'awq'],
    ['gptq', 'awq'], ['sparsegpt', 'wanda'], ['fp8', 'fp8int8'], ['nm', 'trt'], ['fp8', 'trt'], ['tvm', 'trt']],
};
KIT.treeDate = id => id ? '20' + id.slice(0, 2) + '-' + id.slice(2, 4) : '2026';
KIT.tree = (host, opts = {}) => {
  const T = KIT.TREE, byKey = Object.fromEntries(T.nodes.map(n => [n[0], n]));
  host.style.position = 'relative';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('aria-hidden', 'true');
  svg.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;overflow:visible';
  host.appendChild(svg);
  const pills = T.nodes.map(n => { const b = document.createElement('button'); b.type = 'button'; b.id = 'btn-tree-' + n[0]; b.className = 'tree-pill';
    b.textContent = n[3]; b.setAttribute('aria-pressed', 'false'); b.title = n[3] + ', ' + KIT.treeDate(n[2]); host.appendChild(b); return b; });
  const S = { sel: null };
  const t2y = id => { const d = KIT.treeDate(id); return +d.slice(0, 4) + (+(d.slice(5, 7) || 6) - 1) / 12; };
  let pos = {};
  function layout() {
    const w = host.clientWidth, narrow = w < 640;
    pos = {};
    if (narrow) {
      const order = T.nodes.slice().sort((a, b) => t2y(a[2]) - t2y(b[2]));
      order.forEach((n, i) => { pos[n[0]] = { x: 64, y: 8 + i * 34 }; });
      host.style.height = (order.length * 34 + 16) + 'px';
    } else {
      // one row per paper in date order: the vertical order is exact, the spacing is not to scale
      const lw = (w - 60) / T.lanes.length, y0 = 40;
      T.nodes.slice().sort((a, b) => t2y(a[2]) - t2y(b[2])).forEach((n, i) => { pos[n[0]] = { x: 60 + lw * n[1] + 6, y: y0 + i * 27, lw }; });
      host.style.height = (y0 + T.nodes.length * 27 + 12) + 'px';
    }
    pills.forEach((b, i) => { const p = pos[T.nodes[i][0]]; b.style.left = p.x + 'px'; b.style.top = p.y + 'px';
      b.style.maxWidth = narrow ? (w - 76) + 'px' : (p.lw - 14) + 'px'; });
    draw(narrow, w);
  }
  function draw(narrow, w) {
    const mut = PE.css('--muted'), fg = PE.css('--fg'), rule = PE.css('--rule');
    let s = '';
    if (narrow) {
      T.nodes.slice().sort((a, b) => t2y(a[2]) - t2y(b[2])).forEach(n => { const p = pos[n[0]];
        s += `<text x="0" y="${p.y + 17}" font-size="11" fill="${mut}" font-family="system-ui">${KIT.treeDate(n[2])}</text>`; });
    } else {
      const lw = (w - 60) / T.lanes.length;
      T.lanes.forEach((l, i) => { s += `<text x="${60 + lw * i + 6}" y="22" font-size="12" font-weight="600" fill="${fg}" font-family="system-ui">${l}</text>`;
        if (i) s += `<line x1="${60 + lw * i - 4}" y1="30" x2="${60 + lw * i - 4}" y2="100%" stroke="${rule}"/>`; });
      let lastYear = '';
      T.nodes.slice().sort((a, b) => t2y(a[2]) - t2y(b[2])).forEach(n => { const yr = KIT.treeDate(n[2]).slice(0, 4), p = pos[n[0]];
        if (yr !== lastYear) { lastYear = yr; s += `<text x="0" y="${p.y + 16}" font-size="11" fill="${mut}" font-family="system-ui">${yr}</text>`;
          s += `<line x1="34" y1="${p.y - 2}" x2="100%" y2="${p.y - 2}" stroke="${rule}" opacity=".6"/>`; } });
      T.edges.forEach(([a, b]) => { const A = pills[T.nodes.indexOf(byKey[a])], B = pills[T.nodes.indexOf(byKey[b])];
        const on = S.sel && (S.sel === a || S.sel === b);
        const x1 = A.offsetLeft + A.offsetWidth / 2, y1 = A.offsetTop + A.offsetHeight, x2 = B.offsetLeft + B.offsetWidth / 2, y2 = B.offsetTop;
        s += `<path d="M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}" fill="none" stroke="${on ? fg : mut}" stroke-width="${on ? 1.8 : 1}" opacity="${on ? 1 : .35}"/>`; });
    }
    svg.innerHTML = s;
  }
  function set(next) { if ('sel' in next) S.sel = next.sel; pills.forEach((b, i) => b.setAttribute('aria-pressed', String(T.nodes[i][0] === S.sel))); draw(host.clientWidth < 640, host.clientWidth); }
  const ro = new ResizeObserver(layout); ro.observe(host);
  const off = PE.onTheme(() => draw(host.clientWidth < 640, host.clientWidth));
  layout();
  return { set, state: S, pills, destroy: () => { ro.disconnect(); off(); } };
};
KIT.treeText = key => {
  const T = KIT.TREE, n = T.nodes.find(x => x[0] === key); if (!n) return 'Select a paper to see what it did and what it builds on.';
  const from = T.edges.filter(e => e[1] === key).map(e => T.nodes.find(x => x[0] === e[0])[3]);
  const to = T.edges.filter(e => e[0] === key).map(e => T.nodes.find(x => x[0] === e[1])[3]);
  return `<b>${n[3]}</b> (${n[2] ? 'arXiv ' + n[2] + ', ' : ''}${KIT.treeDate(n[2])}). ${n[4]}` +
    (from.length ? ` Builds on: ${from.join(', ')}.` : '') + (to.length ? ` Leads to: ${to.join(', ')}.` : '');
};
