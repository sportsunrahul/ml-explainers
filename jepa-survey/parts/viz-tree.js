<script>
(() => {
  const fig = document.getElementById('viz-tree'), svg = document.getElementById('tree-svg'), out = document.getElementById('tree-readout');
  const L = { found: 0, image: 1, video: 2, theory: 3, other: 4 }, LN = ['foundations', 'images', 'video and world models', 'theory and recipes', 'other modalities'];
  const N = [
    ['byol', 'BYOL', '2020-06-13', 'found', null, 'EMA target plus a predictor; no negatives.'],
    ['vicreg', 'VICReg', '2021-05-11', 'found', null, 'Explicit variance and covariance terms.'],
    ['mae', 'MAE', '2021-11-11', 'found', null, 'Masked pixel reconstruction: the foil.'],
    ['data2vec', 'data2vec', '2022-02-07', 'found', 'byol', 'Regress a teacher\'s hidden features at masked positions.'],
    ['lecun', 'JEPA (position paper)', '2022-06-27', 'found', 'vicreg', 'The diagram: encode both, predict in embedding space, regularise.'],
    ['ijepa', 'I-JEPA', '2023-01-19', 'image', 'lecun', 'Multi-block masking, EMA target, narrow predictor; no augmentations.'],
    ['mcjepa', 'MC-JEPA', '2023-07-24', 'other', 'vicreg', 'VICReg multi-tasked with optical flow.'],
    ['graph', 'Graph-JEPA', '2023-09-27', 'other', 'ijepa', 'Targets on a hyperbola.'],
    ['ajepa', 'A-JEPA', '2023-11-27', 'other', 'ijepa', 'Spectrogram patches with a curriculum mask.'],
    ['vjepa', 'V-JEPA', '2024-02-15', 'video', 'ijepa', 'Tube masks through time; L1; attentive probe.'],
    ['recon', 'Learning by reconstruction', '2024-02-17', 'theory', 'mae', 'Pixel loss fits the least useful directions first.'],
    ['iwm', 'IWM', '2024-03-01', 'image', 'ijepa', 'Condition the predictor on the transformation: a world model.'],
    ['pointjepa', 'Point-JEPA', '2024-04-25', 'other', 'ijepa', 'A sequencer for point clouds.'],
    ['noisy', 'How JEPA avoids noisy features', '2024-07-03', 'theory', 'ijepa', 'Deep linear theory: weight = rho^L.'],
    ['djepa', 'D-JEPA', '2024-10-02', 'other', 'ijepa', 'Adds a diffusion loss; generates.'],
    ['dinowm', 'DINO-WM', '2024-11-07', 'video', 'iwm', 'Frozen DINOv2 patches plus a trained predictor; CEM planning.'],
    ['pldm', 'PLDM', '2025-02-20', 'video', 'vicreg', 'End-to-end latent dynamics from offline data; a case for planning.'],
    ['intphys', 'Intuitive physics', '2025-02-17', 'video', 'vjepa', 'Prediction error as surprise.'],
    ['vjepa2', 'V-JEPA 2', '2025-06-11', 'video', 'vjepa', 'Scale, then an action-conditioned predictor and zero-shot planning.'],
    ['llmjepa', 'LLM-JEPA', '2025-09-11', 'other', 'ijepa', 'A JEPA term between two views of text.'],
    ['gauss', 'Gaussian embeddings', '2025-10-07', 'theory', 'lecun', 'The anti-collapse term estimates density.'],
    ['lejepa', 'LeJEPA', '2025-11-11', 'theory', 'gauss', 'Isotropic Gaussian is optimal; SIGReg enforces it.'],
    ['vljepa', 'VL-JEPA', '2025-12-11', 'other', 'vjepa', 'Predict text embeddings from video.'],
    ['causal', 'Causal-JEPA', '2026-02-11', 'video', 'ijepa', 'Mask objects, not patches.'],
    ['lewm', 'LeWorldModel', '2026-03-13', 'video', 'lejepa', 'Two losses, end to end from pixels.'],
    ['vjepa21', 'V-JEPA 2.1', '2026-03-15', 'video', 'vjepa2', 'Dense loss on every token; deep supervision.'],
    ['varjepa', 'Var-JEPA', '2026-03-20', 'theory', 'ijepa', 'A variational JEPA.'],
    ['lejepawm', 'When does LeJEPA learn a world model?', '2026-05-25', 'theory', 'lejepa', 'Linear identifiability under Gaussian latents.'],
    ['mjepa', 'MJEPA', '2026-06-23', 'other', 'vjepa', 'One encoder for audio and video.'],
    ['hjepa', 'H-JEPA', '2026-10-05', 'video', 'lewm', 'A hierarchy of LeWorldModels; top-down planning.'],
  ].map(([id, name, date, lane, parent, note]) => ({ id, name, date, lane, parent, note, t: Date.parse(date) }));
  const SPINE = new Set(['lecun', 'ijepa', 'vjepa', 'vjepa2', 'vjepa21', 'lejepa', 'lewm', 'hjepa', 'iwm', 'dinowm', 'pldm']);
  const S = { show: 'spine', sel: null };
  const NS = 'http://www.w3.org/2000/svg';
  function draw() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const narrow = svg.clientWidth < 520, W = narrow ? 360 : 760, H = narrow ? 420 : 330; svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const t0 = Date.parse('2020-04-01'), t1 = Date.parse('2026-12-31'), X = t => 70 + (t - t0) / (t1 - t0) * (W - 90), Y = lane => 36 + L[lane] * ((H - 60) / 4);
    const el = (tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.appendChild(e); return e; };
    LN.forEach((n, i) => { const y = 36 + i * ((H - 60) / 4); el('line', { x1: 70, x2: W - 10, y1: y, y2: y, stroke: 'var(--rule)' }); el('text', { x: 4, y: y + 4, fill: 'var(--muted)', 'font-size': narrow ? 8 : 10, 'font-family': 'system-ui' }, narrow ? n.split(' ')[0] : n); });
    for (let yr = 2021; yr <= 2026; yr++) { const x = X(Date.parse(`${yr}-01-01`)); el('line', { x1: x, x2: x, y1: 20, y2: H - 18, stroke: 'var(--rule)', 'stroke-dasharray': '2 4' }); el('text', { x, y: H - 4, 'text-anchor': 'middle', fill: 'var(--muted)', 'font-size': 10, 'font-family': 'system-ui' }, yr); }
    const vis = N.filter(n => S.show === 'all' || SPINE.has(n.id) || n.lane === 'found');
    const pos = Object.fromEntries(N.map((n, i) => [n.id, [X(n.t), Y(n.lane) + ((i % 3) - 1) * (narrow ? 9 : 12)]]));
    vis.forEach(n => { if (!n.parent || !vis.find(v => v.id === n.parent)) return; const [x1, y1] = pos[n.parent], [x2, y2] = pos[n.id];
      el('path', { d: `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`, fill: 'none', stroke: S.sel && (S.sel === n.id || S.sel === n.parent) ? 'var(--yellow)' : 'var(--green)', 'stroke-width': S.sel && (S.sel === n.id || S.sel === n.parent) ? 2 : 1, opacity: .7 }); });
    vis.forEach(n => { const [x, y] = pos[n.id], on = S.sel === n.id;
      const c = el('circle', { cx: x, cy: y, r: on ? 7 : SPINE.has(n.id) ? 5.5 : 4, fill: SPINE.has(n.id) ? 'var(--blue)' : 'var(--brown)', stroke: on ? 'var(--yellow)' : 'none', 'stroke-width': 2, tabindex: 0, role: 'button', 'aria-label': n.name, style: 'cursor:pointer' });
      c.addEventListener('click', () => select(n.id)); c.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(n.id); } });
      if (SPINE.has(n.id) || on || !narrow) el('text', { x: x + 7, y: y - 6, fill: on ? 'var(--fg)' : 'var(--muted)', 'font-size': narrow ? 8 : 9.5, 'font-family': 'system-ui' }, n.name.length > 18 && !on ? n.name.slice(0, 16) + '...' : n.name); });
  }
  function select(id) { S.sel = id; const n = N.find(x => x.id === id); draw(); const p = n.parent && N.find(x => x.id === n.parent);
    out.textContent = `${n.name} (${n.date}): ${n.note}${p ? ` Builds on ${p.name}.` : ''}`; }
  fig.querySelectorAll('[data-show]').forEach(b => b.addEventListener('click', () => { S.show = b.dataset.show; fig.querySelectorAll('[data-show]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); draw(); }));
  document.getElementById('btn-tree-reset').addEventListener('click', () => { S.show = 'spine'; S.sel = null; fig.querySelectorAll('[data-show]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.show === 'spine'))); draw(); out.textContent = 'Select a paper to read its delta card.'; });
  new ResizeObserver(draw).observe(svg); PE.onTheme(draw); draw(); out.textContent = 'Select a paper to read its delta card.';
})();
</script>
