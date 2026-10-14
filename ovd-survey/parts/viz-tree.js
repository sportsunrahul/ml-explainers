<style>
.tree-tip { position:absolute; z-index:3; max-width:17rem; pointer-events:none; font:.82rem/1.45 system-ui, sans-serif; color:var(--fg);
  background:var(--panel2); border:1px solid var(--rule); border-radius:9px; padding:.45rem .6rem; }
.tree-tip b { font-weight:600; }
.tree-tip .id { color:var(--muted); }
details.tree-table { margin:1rem 0 1.4rem; max-width:var(--stage, 60rem); }
details.tree-table summary { font:.92rem system-ui, sans-serif; color:var(--muted); cursor:pointer; }
details.tree-table summary:focus-visible { outline:2px solid var(--yellow); outline-offset:2px; }
#tree-table td, #tree-table th { text-align:left; vertical-align:top; }
#tree-table td:nth-child(2) { white-space:nowrap; }
</style>
<script>
(() => {
  // Lanes: where text enters (the tap strip's nodes: 2 encoder or neck, 3 queries, 4 decoder / region features, 5 score).
  // ramp = step on the yellow ordinal ramp (more of the network reads the text); vis = visual prompts (blue); ev = grey.
  const LANES = [
    { id: 'cls', name: 'classifier only', taps: [5], ramp: 0, enters: 'at the classifier only' },
    { id: 'vis', name: 'visual prompts', taps: [3, 5], enters: 'box or text prompts at query selection and the classifier (late fusion)' },
    { id: 'neck', name: 'neck fusion, folded', taps: [2, 5], ramp: 1, enters: 'in the neck, then folded into weights for deployment' },
    { id: 'deep', name: 'deep fusion', taps: [2, 5], ramp: 2, enters: 'cross-attention in the top encoder layers (deep fusion)' },
    { id: 'full', name: 'fusion + queries + decoder', taps: [2, 3, 4, 5], ramp: 3, enters: 'in the encoder, at query selection and in the decoder' },
    { id: 'ev', name: 'evidence, benchmarks', taps: [], enters: 'a test of detectors, not a detector' },
  ];
  // [key, name, arXiv id, v1 date (abs.html; claim id), lane, spine?, one-line idea, override for "where text enters"]
  const P = [
    ['ovrcnn', 'OVR-CNN', '2011.10678', '2020-11-20', 'ovrcnn-published', 'cls', 0, 'Pretrain a vision-to-language projection on image captions, then detect with fixed word embeddings as the classifier; it named the task.'],
    ['mdetr', 'MDETR', '2104.12763', '2021-04-26', 'mdetr-v1-date', 'deep', 0, 'DETR whose encoder reads image and text tokens together; each box predicts which words it refers to.', 'one joint encoder over image and text tokens'],
    ['vild', 'ViLD', '2104.13921', '2021-04-28', 'vild-published', 'cls', 1, 'Teach the region head to see like CLIP: score regions with CLIP text vectors and pull them toward CLIP\'s embedding of each crop.'],
    ['glip', 'GLIP', '2112.03857', '2021-12-07', 'glip-v1-date', 'deep', 1, 'Detection as phrase grounding: score every region against every prompt token, with image-text cross-attention in the top layers.'],
    ['regionclip', 'RegionCLIP', '2112.09106', '2021-12-16', 'regionclip-published', 'cls', 0, 'CLIP labels proposals with concepts parsed from captions; a region-level CLIP encoder is pretrained on these pairs.'],
    ['detic', 'Detic', '2201.02605', '2022-01-07', 'detic-published', 'cls', 0, 'Image-level labels train the classifier on the largest proposal; the class rows are CLIP text embeddings.'],
    ['owlvit', 'OWL-ViT', '2205.06230', '2022-05-12', 'owlvit-v1-date', 'cls', 1, 'Drop the pooling of a CLIP ViT: every output token predicts one box and one embedding, scored against text or image queries.'],
    ['glipv2', 'GLIPv2', '2206.05836', '2022-06-12', 'glipv2-v1-date', 'deep', 0, 'GLIP plus region-word negatives from the whole batch, computed before fusion, and a masked-language loss.'],
    ['gdino', 'Grounding DINO', '2303.05499', '2023-03-09', 'gdino-v1-date', 'full', 1, 'DINO with text at three places: a feature enhancer, language-guided query selection and text cross-attention in the decoder.'],
    ['owlv2', 'OWLv2', '2306.09683', '2023-06-16', 'owlv2-v1-date', 'cls', 1, 'Self-train on web images that OWL-ViT labelled with word N-grams from their alt text, at billions of examples.'],
    ['d3', 'D3', '2307.12813', '2023-07-24', 'd3-v1-date', 'ev', 0, 'Described object detection: every description is checked on every image, including "without" descriptions; models ignore negation.'],
    ['ovdeval', 'OVDEval', '2308.13177', '2023-08-25', 'ovdeval-v1-date', 'ev', 0, 'Probes attributes, positions, relations and negation with hard negatives, scored so that one label per object counts.'],
    ['fsod', 'FSOD-VLM', '2312.14494', '2023-12-22', 'fsod-vlm-v1-date', 'ev', 0, 'Few-shot detection with VLMs: revisits the task as aligning a pretrained detector to a target\'s own label policy from a few examples (nuImages).'],
    ['mmg', 'MM-Grounding-DINO', '2401.02361', '2024-01-04', 'mmgdino-v1-date', 'full', 0, 'An open rebuild of Grounding DINO training, with more open data and random negative prompts.'],
    ['yolow', 'YOLO-World', '2401.17270', '2024-01-30', 'yoloworld-v1-date', 'neck', 1, 'A YOLO detector with text gating its neck; the user\'s vocabulary is encoded once and folded into conv weights.'],
    ['trex2', 'T-Rex2', '2403.14610', '2024-03-21', 'trex2-v1-date', 'vis', 0, 'Text prompts and example boxes trained into one embedding space; either can name the class.'],
    ['gd15', 'Grounding DINO 1.5', '2405.10300', '2024-05-16', 'gdino15-v1-date', 'full', 0, 'Grounding DINO scaled up in backbone and data (Pro), and slimmed for edge devices (Edge).', 'as Grounding DINO; the Edge model fuses text into the coarsest feature level only'],
    ['ovdino', 'OV-DINO', '2407.07844', '2024-07-10', 'ovdino-v1-date', 'full', 0, 'Keeps the encoder image-only: text picks the queries and enters the decoder through gated cross-attention.', 'at query selection and in the decoder; no fusion in the encoder'],
    ['dinox', 'DINO-X', '2411.14347', '2024-11-21', 'dinox-v1-date', 'full', 0, 'Grounding DINO 1.5\'s design with a CLIP text encoder, box and point prompts, and a prompt-free mode.'],
    ['llmdet', 'LLMDet', '2501.18954', '2025-01-31', 'llmdet-v1-date', 'full', 0, 'Co-trains MM-Grounding-DINO with a small language model that writes captions from its features; the language model is dropped at test time.', 'as Grounding DINO, plus a caption-writing language model during training only'],
    ['yoloe', 'YOLOE', '2503.07465', '2025-03-10', 'yoloe-v1-date', 'cls', 0, 'A plain YOLO that scores anchors against prompt embeddings from text, an example box, or no prompt at all.', 'at the classifier only, through a small network folded into the last conv'],
    ['rf100', 'Roboflow100-VL', '2505.20612', '2025-05-27', 'rf100vl-v1-date', 'ev', 0, 'Datasets of concepts that are rare on the web, measured from zero-shot to full labels on one test split.'],
  ].map(([key, name, id, date, claim, lane, spine, idea, enters]) => ({ key, name, id, date, claim, lane, spine: !!spine, idea, enters,
    ev: lane === 'ev', hollow: key === 'ovdino', t: Date.parse(date) }));
  const byKey = Object.fromEntries(P.map(p => [p.key, p]));
  // "Builds on", each stated in the later paper (notes/papers/*.md).
  const E = [
    ['mdetr', 'glip', 'GLIP says its grounding formulation is inspired by MDETR'],
    ['vild', 'detic', 'Detic uses CLIP text embeddings as the classifier, following ViLD'],
    ['glip', 'glipv2', 'GLIPv2 keeps GLIP\'s detection-as-grounding loss'],
    ['owlvit', 'owlv2', 'an OWL-ViT model labels OWLv2\'s training images'],
    ['glip', 'gdino', 'Grounding DINO trains on detection and grounding data the way GLIP does'],
    ['gdino', 'mmg', 'MM-Grounding-DINO rebuilds its training in the open'],
    ['glip', 'yolow', 'a GLIP model proposes YOLO-World\'s pseudo-boxes'],
    ['gdino', 'trex2', 'T-Rex2 uses its language-guided query selection'],
    ['gdino', 'gd15', 'Grounding DINO 1.5 keeps its design and scales it'],
    ['gdino', 'ovdino', 'OV-DINO keeps its query selection, without the encoder fusion'],
    ['detic', 'ovdino', 'OV-DINO turns captions into image-sized boxes, after Detic'],
    ['gd15', 'dinox', 'DINO-X starts from its early-fusion encoder-decoder'],
    ['trex2', 'dinox', 'DINO-X takes its box and point prompts from T-Rex2'],
    ['mmg', 'llmdet', 'LLMDet co-trains MM-Grounding-DINO'],
    ['yolow', 'yoloe', 'YOLOE starts from YOLO-World and removes its neck fusion'],
  ];
  const laneOf = id => LANES.findIndex(l => l.id === id);
  const enters = p => p.enters || LANES[laneOf(p.lane)].enters;
  const parents = p => E.filter(e => e[1] === p.key);
  const MODES = { spine: p => p.spine, branches: p => !p.ev, all: () => true };

  const fig = document.getElementById('viz-tree'), svg = document.getElementById('tree-svg'), out = document.getElementById('tree-readout');
  const tip = document.getElementById('tree-tip'), NS = 'http://www.w3.org/2000/svg', MC = document.createElement('canvas').getContext('2d');
  const tw = (s, fs, wt) => { MC.font = `${wt} ${fs}px system-ui, sans-serif`; return MC.measureText(s).width; };
  const S = { mode: 'all', alpha: P.map(() => 1), sel: -1, hover: -1 };
  let stop = () => {}, lastW = 0, pos = [];

  function el(tag, a, txt, parent = svg) {
    const e = document.createElementNS(NS, tag);
    for (const k in a) e.setAttribute(k, a[k]);
    if (txt != null) e.textContent = txt;
    parent.appendChild(e); return e;
  }
  const T = (x, y, s, o = {}) => el('text', { x, y, 'font-family': 'system-ui, sans-serif', 'font-size': o.fs || 11, fill: o.fill,
    'font-weight': o.wt || 400, 'text-anchor': o.anchor || 'start', opacity: o.op == null ? 1 : o.op }, s);
  function laneColors(k) {
    const dark = parseInt(k.bg.slice(1, 3), 16) < 128, us = dark ? [.45, .63, .82, 1] : [.55, .7, .85, 1];
    return LANES.map(l => l.ramp != null ? DK.mixh(k.bg, k.Y, us[l.ramp]) : l.id === 'vis' ? k.B : DK.mixh(k.bg, k.fg, .45));
  }

  // Desktop: x = date, y = lane. Labels take one of four slots around the node, first free slot wins.
  function layoutWide(W) {
    const LW = 176, px0 = LW + 22, px1 = W - 30, t0 = Date.parse('2020-09-01'), t1 = Date.parse('2025-08-01');
    const X = t => px0 + (t - t0) / (t1 - t0) * (px1 - px0), SL = [-12, 21, -26, 35];
    const lab = P.map(() => null), laneH = LANES.map(() => 58);
    LANES.forEach((l, li) => {
      const ends = [-1e9, -1e9, -1e9, -1e9]; let prevX = -1e9, nud = 0;
      P.map((p, i) => [p, i]).filter(([p]) => p.lane === l.id).sort((a, b) => a[0].t - b[0].t).forEach(([p, i]) => {
        const fs = p.spine ? 12 : 11.5, wt = p.spine ? 700 : 500, w = tw(p.name, fs, wt), x = X(p.t);
        const lx = Math.min(Math.max(x - w / 2, px0 - 16), W - 6 - w);
        let s = [0, 1, 2, 3].find(j => ends[j] < lx - 6); if (s == null) s = ends.indexOf(Math.min(...ends));
        ends[s] = lx + w; if (s > 1) laneH[li] = 86;
        nud = x - prevX < 13 ? (nud ? 0 : 7) : 0; prevX = x;
        lab[i] = { lx, s, fs, wt, w, x, nud };
      });
    });
    let y = 34; const cy = LANES.map((l, li) => { const c = y + laneH[li] / 2; y += laneH[li]; return c; });
    pos = P.map((p, i) => { const L = lab[i], c = cy[laneOf(p.lane)]; return { x: L.x, y: c + L.nud, ly: c + SL[L.s], lx: L.lx, s: L.s, fs: L.fs, wt: L.wt }; });
    return { H: y + 30, X, cy, laneH, px0, px1, LW, top: 34, bottom: y };
  }
  // Narrow: y = date (pushed apart to a minimum gap), x = lane column; names to the right of the columns.
  function layoutNarrow(W) {
    const L0 = 32, colX = i => L0 + 10 + i * 19, nameX = colX(LANES.length - 1) + 16, top = 40 + LANES.length * 15 + 8;
    const order = P.map((p, i) => i).sort((a, b) => P[a].t - P[b].t), t0 = P[order[0]].t, t1 = P[order[order.length - 1]].t;
    const Y = {}; let last = -1e9;
    order.forEach(i => { let y = top + 12 + (P[i].t - t0) / (t1 - t0) * 440; y = Math.max(y, last + 21); Y[i] = y; last = y; });
    pos = P.map((p, i) => ({ x: colX(laneOf(p.lane)), y: Y[i], lx: nameX, ly: Y[i] + 4, fs: p.spine ? 12 : 11.5, wt: p.spine ? 700 : 500 }));
    return { H: last + 22, colX, nameX, top, order };
  }

  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const k = DK.C(), W = Math.max(280, svg.clientWidth), narrow = W < 620, LC = laneColors(k);
    const g = narrow ? layoutNarrow(W) : layoutWide(W), faint = DK.mixh(k.bg, k.fg, .14), edge = DK.mixh(k.bg, k.fg, .38);
    const focus = S.hover >= 0 ? S.hover : S.sel, fk = focus >= 0 ? P[focus].key : null;
    if (narrow) {
      T(8, 16, 'Lanes: where text enters', { fill: k.mu, fs: 12 });
      LANES.forEach((l, i) => { const x = g.colX(i), y = 40 + i * 15;
        el('line', { x1: x, x2: x, y1: y + 5, y2: g.H - 8, stroke: faint });
        el('circle', { cx: x, cy: y - 4, r: 4, fill: LC[i] });
        T(x + 8, y, l.name, { fill: k.fg, fs: 10.5 }); });
      // year separators between the last paper of one year and the first of the next
      for (let j = 1; j < g.order.length; j++) {
        const a = P[g.order[j - 1]], b = P[g.order[j]], ya = a.date.slice(0, 4), yb = b.date.slice(0, 4);
        if (ya !== yb) { const y = (pos[g.order[j - 1]].y + pos[g.order[j]].y) / 2; el('line', { x1: 0, x2: W, y1: y, y2: y, stroke: k.rule }); T(2, y + 13, yb, { fill: k.mu, fs: 10.5 }); }
      }
      T(2, pos[g.order[0]].y + 4, P[g.order[0]].date.slice(0, 4), { fill: k.mu, fs: 10.5 });
    } else {
      T(8, 16, 'First arXiv version, left to right; one lane for each place text enters', { fill: k.mu, fs: 12 });
      for (let yr = 2021; yr <= 2025; yr++) { const x = g.X(Date.parse(`${yr}-01-01`));
        el('line', { x1: x, x2: x, y1: g.top, y2: g.bottom, stroke: k.rule }); T(x, g.bottom + 16, String(yr), { fill: k.mu, fs: 11, anchor: 'middle' }); }
      let y = g.top;
      LANES.forEach((l, i) => {
        if (i) el('line', { x1: 8, x2: W - 8, y1: y + .5, y2: y + .5, stroke: k.rule });
        const c = g.cy[i];
        el('circle', { cx: 14, cy: c - 4 - (l.taps.length ? 6 : 0), r: 4.5, fill: LC[i] });
        T(24, c - (l.taps.length ? 6 : 0), l.name, { fill: k.fg, fs: 11.5, wt: 600 });
        // mini tap strip: six pipeline boxes, the ones that read text filled in the lane color
        l.taps.length && [0, 1, 2, 3, 4, 5].forEach(j => { const on = l.taps.includes(j);
          el('rect', { x: 24 + j * 12, y: c + 4, width: 9, height: 7, rx: 1.5, fill: on ? LC[i] : 'none', stroke: on ? LC[i] : DK.mixh(k.bg, k.B, .7), 'stroke-width': 1 }); });
        y += g.laneH[i];
      });
    }
    // edges under the nodes
    E.forEach(([a, b]) => {
      const ia = P.indexOf(byKey[a]), ib = P.indexOf(byKey[b]), al = Math.min(S.alpha[ia], S.alpha[ib]); if (al < .02) return;
      const p = pos[ia], q = pos[ib], on = fk && (fk === a || fk === b);
      // A same-lane edge that skips other papers bows out of the lane, so it does not read as a chain through them.
      const A = P[ia], B = P[ib], skip = A.lane === B.lane ? P.filter((r, j) => r.lane === A.lane && r.t > A.t && r.t < B.t && S.alpha[j] > .02).length : 0;
      const h = skip ? (8 + 5 * skip) * 1.33 : 0;
      const d = narrow ? (skip ? `M${p.x},${p.y} C${p.x - h},${p.y} ${q.x - h},${q.y} ${q.x},${q.y}` : `M${p.x},${p.y} C${p.x},${(p.y + q.y) / 2} ${q.x},${(p.y + q.y) / 2} ${q.x},${q.y}`)
        : (skip ? `M${p.x},${p.y} C${p.x},${p.y + h} ${q.x},${q.y + h} ${q.x},${q.y}` : `M${p.x},${p.y} C${(p.x + q.x) / 2},${p.y} ${(p.x + q.x) / 2},${q.y} ${q.x},${q.y}`);
      el('path', { d, fill: 'none', stroke: on ? k.fg : edge, 'stroke-width': on ? 2 : 1.2, opacity: al * (fk && !on ? .45 : 1) });
    });
    P.forEach((p, i) => {
      const a = S.alpha[i], q = pos[i], col = LC[laneOf(p.lane)], on = i === focus; if (a < .02) return;
      const grp = el('g', { opacity: a });
      if (narrow) el('line', { x1: q.x + 7, x2: q.lx - 4, y1: q.y, y2: q.y, stroke: faint }, null, grp);
      else if (q.s > 1) el('line', { x1: q.x, x2: q.x, y1: q.y + (q.s === 2 ? -6 : 6), y2: q.ly + (q.s === 2 ? 3 : -12), stroke: faint }, null, grp);
      if (on) el('circle', { cx: q.x, cy: q.y, r: 12, fill: 'none', stroke: k.fg, 'stroke-width': 1.5 }, null, grp);
      if (p.ev) el('rect', { x: q.x - 5, y: q.y - 5, width: 10, height: 10, rx: 1.5, fill: col }, null, grp);
      else if (p.hollow) el('circle', { cx: q.x, cy: q.y, r: 5, fill: k.bg, stroke: col, 'stroke-width': 2.5 }, null, grp);
      else el('circle', { cx: q.x, cy: q.y, r: p.spine ? 7 : 5.5, fill: col, stroke: p.spine ? k.fg : k.bg, 'stroke-width': p.spine ? 2 : 1.5 }, null, grp);
      const nt = T(q.lx, q.ly, p.name, { fill: p.spine || on ? k.fg : DK.mixh(k.bg, k.fg, .8), fs: q.fs, wt: on ? 700 : q.wt });
      nt.setAttribute('opacity', a); nt.setAttribute('stroke', k.bg); nt.setAttribute('stroke-width', 3); nt.setAttribute('paint-order', 'stroke');
    });
    svg.setAttribute('viewBox', `0 0 ${W} ${g.H}`); svg.style.height = g.H + 'px';
    placeTip();
  }

  function placeTip() {
    const i = S.hover >= 0 ? S.hover : S.sel;
    if (i < 0 || S.alpha[i] < .5) { tip.hidden = true; return; }
    const p = P[i], q = pos[i], r = svg.getBoundingClientRect(), f = fig.getBoundingClientRect(), sc = r.width / (svg.viewBox.baseVal.width || r.width);
    tip.innerHTML = `<b>${p.name}</b> <span class="id">arXiv ${p.id}</span><br>${p.idea}`;
    tip.hidden = false;
    const tw0 = tip.offsetWidth, th = tip.offsetHeight, x = r.left - f.left + q.x * sc, y = r.top - f.top + q.y * sc;
    tip.style.left = PE.clamp(x + 14 + tw0 > f.width - 8 ? x - 14 - tw0 : x + 14, 8, Math.max(8, f.width - tw0 - 8)) + 'px';
    tip.style.top = (y + 14 + th > r.bottom - f.top ? y - 14 - th : y + 14) + 'px';
  }
  const visible = () => P.map((p, i) => i).filter(i => MODES[S.mode](P[i])).sort((a, b) => P[a].t - P[b].t);
  const claim = p => `<span data-claim="${p.claim}">${p.date}</span>`;
  function sayDefault() {
    out.textContent = { spine: 'The spine: the six papers with their own chapter on this page.',
      branches: 'The spine plus twelve branches: the papers that built on it or beside it.',
      all: 'All 22 papers, with the four evidence papers in grey.' }[S.mode] + ' Point at a paper or tap it, or focus the chart and use the arrow keys.';
  }
  function saySel() {
    const p = P[S.sel], ps = parents(p);
    out.innerHTML = `<b>${p.name}</b>, first version ${claim(p)}, arXiv ${p.id}. Text enters: ${enters(p)}. ${p.idea}` +
      (ps.length ? ` Builds on ${ps.map(e => `${byKey[e[0]].name} (${e[2]})`).join('; ')}.` : '');
  }
  // Buttons SET what is shown; pressing one twice equals pressing it once.
  function set(mode) {
    S.mode = mode; S.hover = -1; if (S.sel >= 0 && !MODES[mode](P[S.sel])) S.sel = -1;
    ['spine', 'branches', 'all'].forEach(m => document.getElementById('btn-tree-' + m).setAttribute('aria-pressed', String(m === mode)));
    const from = S.alpha.slice(), to = P.map(p => (MODES[mode](p) ? 1 : 0));
    stop(); stop = PE.animate(650, u => { S.alpha = from.map((a, i) => PE.lerp(a, to[i], u)); draw(); });
    S.sel >= 0 ? saySel() : sayDefault();
  }
  const hit = e => { const r = svg.getBoundingClientRect(), sc = (svg.viewBox.baseVal.width || r.width) / r.width;
    const x = (e.clientX - r.left) * sc, y = (e.clientY - r.top) * sc; let best = -1, bd = 16;
    P.forEach((p, i) => { if (S.alpha[i] < .5) return; const d = Math.hypot(pos[i].x - x, pos[i].y - y); if (d < bd) { bd = d; best = i; } });
    return best; };
  ['spine', 'branches', 'all'].forEach(m => document.getElementById('btn-tree-' + m).addEventListener('click', () => set(m)));
  document.getElementById('btn-tree-reset').addEventListener('click', () => { S.sel = -1; set('all'); });
  svg.addEventListener('pointermove', e => { const i = hit(e); if (i !== S.hover) { S.hover = i; draw(); } });
  svg.addEventListener('pointerleave', () => { if (S.hover >= 0) { S.hover = -1; draw(); } });
  svg.addEventListener('pointerdown', e => { const i = hit(e); if (i < 0) return; S.sel = i; draw(); saySel(); });
  svg.addEventListener('keydown', e => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (e.key === 'Escape') { S.sel = -1; draw(); sayDefault(); return; }
    if (d == null) return; e.preventDefault();
    const v = visible(), j = v.indexOf(S.sel); S.hover = -1;
    S.sel = v[j < 0 ? (d > 0 ? 0 : v.length - 1) : PE.clamp(j + d, 0, v.length - 1)]; draw(); saySel();
  });
  svg.addEventListener('blur', () => { if (S.hover < 0) tip.hidden = true; });
  // Table view: the same data without hover.
  document.querySelector('#tree-table tbody').innerHTML = P.slice().sort((a, b) => a.t - b.t).map(p => `<tr><td><a href="https://arxiv.org/abs/${p.id}" title="arXiv ${p.id}">${p.name}</a></td><td>${claim(p)}</td><td>${enters(p)}</td><td>${parents(p).map(e => byKey[e[0]].name).join(', ') || '-'}</td><td>${p.idea}</td></tr>`).join('');
  new ResizeObserver(() => { const w = svg.clientWidth; if (w !== lastW) { lastW = w; draw(); } }).observe(svg);
  PE.onTheme(draw);
  draw(); sayDefault();
})();
</script>
