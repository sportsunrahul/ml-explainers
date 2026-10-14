<script>
(() => {
  // Interlude: one generated token through Mamba 2.8B (recurrent mode) and Llama 3 8B (KV cache). Sizes from CoreBill.
  const fig = document.getElementById('viz-interlude'), el = document.getElementById('interlude-canvas'), out = document.getElementById('interlude-readout');
  const texts = [...fig.querySelectorAll('.interlude-text > div')], btn = k => document.getElementById('btn-interlude-' + k);
  const STAGES = ['token', 'proj', 'memory', 'out', 'layers', 'next'], LS = { '1k': 1024, '8k': 8192, '128k': 131072, '1m': 1048576 };
  const M = CoreBill.MAMBA, Q = CoreBill.LLAMA, by = CoreBill.BYTES, fb = CoreBill.fmtBytes, fn = n => n.toLocaleString('en-US');
  const inner = M.expand * M.dModel, ssmL = inner * M.dState, convL = inner * (M.dConv - 1), stateL = (ssmL + convL) * by;
  const kvL = 2 * Q.kvHeads * Q.headDim, mW = 3 * M.expand * M.dModel ** 2, qW = 4096 * (4096 + 2 * kvL / 2 + 4096) + 3 * 4096 * 14336;
  const S0 = { stage: 0, L: 8192, u: 1 }, S = { ...S0 };
  let cancel = () => {};
  function rr(c, x, y, w, h, r, fill, stroke, lw = 1.4, a = 1) { c.save(); c.globalAlpha = a; c.beginPath(); c.roundRect(x, y, w, h, r); if (fill) { c.fillStyle = fill; c.fill(); } if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw; c.stroke(); } c.restore(); }
  function draw(c, W, H) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), R: PE.css('--red'), Br: PE.css('--brown') };
    c.clearRect(0, 0, W, H);
    const narrow = W < 520, k = S.stage, u = S.u, colW = W / 2, fs = narrow ? 9 : 11;
    const rows = [{ key: 1, y: 34, t: 'token' }, { key: 1, y: 82, t: 'embedding' }, { key: 2, y: 146, t: 'projections' }, { key: 3, y: 196, t: null }, { key: 4, y: 246, t: null }, { key: 6, y: 330, t: 'head, sample' }, { key: 6, y: 378, t: 'next token' }];
    const bh = narrow ? 26 : 30;
    [0, 1].forEach(m => {
      const x0 = m * colW + (narrow ? 6 : 16), bw = narrow ? colW * .5 : Math.min(colW * .48, 190), cx = x0 + bw / 2;
      PE.label(c, m ? 'Llama 3 8B' : 'Mamba 2.8B', x0, 18, m ? C.Br : C.Y, narrow ? 11 : 13);
      // Layer frame.
      const lit5 = k === 5;
      rr(c, x0 - 4, 134, bw + 8, 152, 8, null, lit5 ? C.fg : C.rule, lit5 ? 2.2 : 1.2);
      PE.label(c, m ? `x ${Q.layers} layers` : `x ${M.layers} layers`, x0 + bw + (narrow ? -2 : 8), 292, lit5 ? C.fg : C.mu, fs, narrow ? 'right' : 'left');
      rows.forEach((r, i) => {
        const t = r.t || (i === 3 ? (m ? 'attention' : 'conv, Δ B C, state step') : (m ? 'out proj + MLP' : 'gate, out proj'));
        const on = k === r.key || (k === 5 && i >= 2 && i <= 4), col = i === 0 ? C.B : i === 6 ? C.G : i === 3 ? (m ? C.Br : C.Y) : C.fg;
        rr(c, x0, r.y, bw, bh, 6, PE.mix(C.bg, col.length === 7 ? col : '#888888', on ? .16 + .1 * u : .05), col, on ? 2.2 : 1.1, on ? 1 : .55);
        PE.label(c, narrow && t.length > 14 ? t.replace('conv, Δ B C, state step', 'state step').replace('out proj + MLP', 'out + MLP').replace('gate, out proj', 'gate, out').replace('head, sample', 'head') : t, cx, r.y + bh / 2 + 4, on ? C.fg : C.mu, fs, 'center');
        if (i > 0) PE.arrow(c, cx, rows[i - 1].y + bh + 2, cx, r.y - 2, C.mu, 1.2, 6, .5);
      });
      // Moving dot along the path to the current stage.
      if (k >= 1 && u < 1) { const tgt = [0, 1, 2, 3, 4, 4, 6][k], y = PE.lerp(rows[Math.max(0, tgt - 1)].y, rows[tgt].y, u); PE.dot(c, cx, y + bh / 2, 4, C.B); }
      // Memory beside the memory box.
      const mx = x0 + bw + (narrow ? 6 : 18), my = 186, room = colW - (mx - m * colW) - (narrow ? 4 : 14), memOn = k === 3 || k === 5 || k === 6;
      if (!m) {
        const n = 4, s = narrow ? 8 : 11;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) rr(c, mx + j * (s + 2), my + i * (s + 2), s, s, 2, PE.mix(C.bg, C.Y, .25 + .5 * (((i * 5 + j * 3 + (k === 6 ? 2 : 0)) % 4) / 4)), C.Y, 1, memOn ? 1 : .6);
        PE.label(c, narrow ? fb(stateL) : `state: ${fb(stateL)} per layer`, mx, my + n * (s + 2) + 12, C.Y, fs);
        PE.label(c, 'any length', mx, my + n * (s + 2) + 25, C.mu, fs);
        if (k === 3) { PE.arrow(c, mx - 2, my + 10, x0 + bw + 2, 207, C.Y, 1.6, 6, u); PE.arrow(c, x0 + bw + 2, 214, mx - 2, my + 30, C.Y, 1.6, 6, u); }
      } else {
        const f = PE.clamp(Math.log2(S.L / 512) / Math.log2(2048), .05, 1), w = Math.max(10, room * f), h = narrow ? 44 : 52;
        rr(c, mx, my, w, h, 3, PE.mix(C.bg, C.Br, .35), C.Br, 1.2, memOn ? 1 : .6);
        for (let x = mx + 5; x < mx + w; x += 5) PE.polyline(c, [[x, my], [x, my + h]], C.Br, .6, null, .4);
        if (k === 6) rr(c, mx + w - 4, my, 4, h, 1, C.R, null);
        PE.label(c, narrow ? fb(kvL * by * S.L) : `KV cache: ${fb(kvL * by * S.L)} per layer`, mx, my + h + 12, C.Br, fs);
        PE.label(c, `at ${CoreBill.fmtTok(S.L)} tokens`, mx, my + h + 25, C.mu, fs);
        if (k === 3) { PE.arrow(c, mx - 2, my + h / 2, x0 + bw + 2, 210, C.Br, 1.5 + 4 * f, 7, u); PE.arrow(c, x0 + bw + 2, 202, mx + 6, my + 8, C.R, 1.3, 6, u); }
      }
    });
  }
  const cv = PE.canvas(el, w => 412 / w, draw);
  function readout() {
    const L = S.L, kvLayer = kvL * by * L, Lt = fn(L);
    return [
      `Context ${Lt} tokens. Per layer, Mamba holds ${fb(stateL)} of state; Llama holds ${fb(kvLayer)} of cache. Pick a stage.`,
      `Token in: one row of the embedding table, a vector of ${fn(M.dModel)} numbers (Mamba) or 4,096 (Llama).`,
      `Projections: Mamba ${fn(M.dModel)} -> 2 x ${fn(inner)}; Llama 4,096 -> query 4,096, key ${fn(kvL / 2)}, value ${fn(kvL / 2)}.`,
      `Memory step, per layer at ${Lt} tokens: Mamba reads and overwrites ${fn(ssmL)} state numbers and ${fn(convL)} convolution inputs (${fb(stateL)}). Llama writes ${fn(kvL)} numbers (${fb(kvL * by)}) and reads ${fn(L)} x ${fn(kvL)} = ${CoreBill.fmtNum(L * kvL)} (${fb(kvLayer)}).`,
      `Out: Mamba gates and projects 5,120 -> ${fn(M.dModel)}; Llama projects to 4,096, then its MLP goes 4,096 -> 14,336 -> 4,096.`,
      `All layers, per token at ${Lt} tokens: Mamba state ${fb(stateL * M.layers)} read and rewritten, plus about ${fb(mW * M.layers * by)} of projection weights. Llama cache ${fb(kvLayer * Q.layers)} read, plus about ${fb(qW * Q.layers * by)} of layer weights.`,
      `Next token: Mamba's memory stays at ${fb(CoreBill.ssmBytes)}; Llama's cache grows by ${fb(CoreBill.kvPerToken)} to ${fb(CoreBill.kvBytes(L + 1))}.`,
    ][S.stage];
  }
  const press = () => { STAGES.forEach((s, i) => btn(s).setAttribute('aria-pressed', String(S.stage === i + 1))); Object.entries(LS).forEach(([k, v]) => btn(k).setAttribute('aria-pressed', String(S.L === v))); texts.forEach(d => { d.hidden = +d.dataset.stage !== S.stage; }); };
  function go(stage) { cancel(); S.stage = PE.clamp(stage, 0, 6); S.u = 0; press(); out.textContent = readout(); cancel = PE.animate(700, u => { S.u = u; cv.redraw(); }); }
  STAGES.forEach((s, i) => btn(s).addEventListener('click', () => go(i + 1)));
  Object.entries(LS).forEach(([k, v]) => btn(k).addEventListener('click', () => { S.L = v; press(); out.textContent = readout(); cancel(); S.u = 0; cancel = PE.animate(600, u => { S.u = u; cv.redraw(); }); }));
  btn('fwd').addEventListener('click', () => go(S.stage + 1));
  btn('prev').addEventListener('click', () => go(S.stage - 1));
  btn('reset').addEventListener('click', () => { cancel(); Object.assign(S, S0); press(); out.textContent = readout(); cv.redraw(); });
  press(); out.textContent = readout();
})();
</script>
