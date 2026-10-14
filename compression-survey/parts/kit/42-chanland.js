/* viz-chanland: a toy depthwise 3x3 layer after BN folding, 32 output channels x 9 weights, shaped after the
   spread in Nagel 2021 Fig. 5 as read off the figure (six wide channels, two all-zero channels, the rest small).
   Both grids are asymmetric 8-bit min-max grids (256 codes) with 0.0 exact: one grid for the whole tensor, or one
   per output channel. "Levels inside a channel's range" counts the codes between the codes of its min and max. */
KIT.CHAN_WIDE = { 4: [-25, 67], 14: [-46, 59], 15: [-60, 63], 27: [-29, 95], 28: [-45, 54], 29: [-57, 94], 3: [-6.5, 16] };
KIT.CHAN_DEAD = [10, 22];
KIT.chanWeights = (() => {
  let cache = null;
  return () => {
    if (cache) return cache;
    const r = PE.rng(5), W = [];
    for (let k = 0; k < 32; k++) {
      let lo, hi;
      if (KIT.CHAN_WIDE[k]) [lo, hi] = KIT.CHAN_WIDE[k];
      else if (KIT.CHAN_DEAD.includes(k)) { W.push(new Array(9).fill(0)); continue; }
      else { const a = .6 + 4.2 * Math.pow(r(), 1.6); lo = -a * (.4 + .8 * r()); hi = a * (.5 + .7 * r()); }
      const w = []; const imin = Math.floor(r() * 9); let imax = Math.floor(r() * 9); if (imax === imin) imax = (imin + 4) % 9;
      for (let t = 0; t < 9; t++) w.push(t === imin ? lo : t === imax ? hi : lo + (hi - lo) * (.15 + .7 * (r() + r() + r()) / 3));
      W.push(w);
    }
    cache = W; return W;
  };
})();
KIT.chanGrid = (lo, hi) => { lo = Math.min(lo, 0); hi = Math.max(hi, 0); if (hi - lo < 1e-9) return { S: 0, Z: 0 }; const S = (hi - lo) / 255; return { S, Z: Math.min(255, Math.max(0, Math.round(-lo / S))) }; };
KIT.chanCode = (G, v) => G.S === 0 ? G.Z : Math.min(255, Math.max(0, Math.round(v / G.S) + G.Z));
KIT.chanStats = mode => {
  const W = KIT.chanWeights(), all = W.flat();
  const T = KIT.chanGrid(Math.min(...all), Math.max(...all));
  let se = 0; const ch = [];
  W.forEach(w => {
    const G = mode === 'tensor' ? T : KIT.chanGrid(Math.min(...w), Math.max(...w));
    const q = w.map(v => KIT.chanCode(G, v)), wq = q.map(c => G.S * (c - G.Z));
    let e = 0, p = 0; w.forEach((v, t) => { e += (v - wq[t]) ** 2; p += v * v; }); se += e;
    const levels = q.length ? Math.max(...q) - Math.min(...q) + 1 : 1;
    ch.push({ G, wq, levels, rel: p > 0 ? Math.sqrt(e / p) : 0, lo: Math.min(...w), hi: Math.max(...w), dead: p === 0 });
  });
  const live = ch.filter(c => !c.dead), sortL = live.map(c => c.levels).sort((a, b) => a - b), sortR = live.map(c => c.rel).sort((a, b) => a - b);
  const med = a => a.length % 2 ? a[(a.length - 1) / 2] : (a[a.length / 2 - 1] + a[a.length / 2]) / 2;
  return { mode, T, ch, mse: se / all.length, medLevels: med(sortL), minLevels: sortL[0], medRel: med(sortR), nSmall: live.filter(c => c.levels <= 16).length };
};
KIT.chanland = (el, opts = {}) => {
  const W = KIT.chanWeights(), stats = { tensor: KIT.chanStats('tensor'), channel: KIT.chanStats('channel') };
  const S = { mode: 'tensor', zoom: 0, m: 0 };     // m: 0 = per-tensor grid shown, 1 = per-channel grid shown
  const XS = .155, X = k => (k - 15.5) * XS, Zt = t => (t - 4) * .05, YT = 1.45 / 95;
  const amax = W.map(w => Math.max(1e-9, ...w.map(Math.abs)));
  const Y = (k, v) => PE.lerp(v * YT, KIT.CHAN_DEAD.includes(k) ? 0 : v / amax[k] * .85, S.zoom);
  const asp = opts.aspect || (w => w < 560 ? .75 : .6);
  // camera distance that fits the 32 channels across the width for the current height / width ratio
  const fitDist = () => { const w = el.clientWidth || 600; return Math.max(4.9, 3.05 * asp(w) / .364); };
  const s3 = PE3.scene(el, { aspect: asp, yaw: -.3, pitch: .58, dist: fitDist(), fov: 40, items: [] });
  function ticks(k, G, alpha, link) {
    const out = [], w = W[k]; if (G.S === 0) return out;
    const q0 = KIT.chanCode(G, Math.min(...w)), q1 = KIT.chanCode(G, Math.max(...w)), n = q1 - q0 + 1, z0 = Zt(-.6), z1 = Zt(8.6), x = X(k);
    if (n > 24) {
      const ya = Y(k, G.S * (q0 - G.Z)), yb = Y(k, G.S * (q1 - G.Z));
      out.push({ type: 'quad', pts: [[x - .05, ya, z0], [x - .05, yb, z0], [x - .05, yb, z1], [x - .05, ya, z1]], fill: '--yellow', stroke: null, alpha: alpha * .16, link });
      return out;
    }
    for (let q = q0; q <= q1; q++) { const y = Y(k, G.S * (q - G.Z)); out.push({ type: 'line', pts: [[x, y, z0], [x, y, z1]], color: '--yellow', width: 1.4, alpha: alpha * .9, link }); }
    return out;
  }
  function build() {
    const it = [], st = stats.tensor, sc = stats.channel, m = S.m;
    it.push({ type: 'line', pts: [[X(-1), 0, Zt(-1)], [X(32), 0, Zt(-1)]], color: '--muted', width: 1, alpha: .6 });
    it.push({ type: 'label', at: [X(4), -1.08, Zt(0)], text: 'output channel 1 to 32', color: '--muted', font: '500 11px system-ui, sans-serif' });
    it.push({ type: 'label', at: [X(32.2), 0, Zt(4)], text: (el.clientWidth || 600) < 560 ? '9 each' : '9 weights each', color: '--muted', font: '500 11px system-ui, sans-serif', align: 'left' });
    const blueP = [], yelP = [];
    for (let k = 0; k < 32; k++) {
      if (m < 1) it.push(...ticks(k, st.T, 1 - m, 'grid'));
      if (m > 0) it.push(...ticks(k, sc.ch[k].G, m, 'grid'));
      W[k].forEach((v, t) => {
        const wq = PE.lerp(st.ch[k].wq[t], sc.ch[k].wq[t], m), yv = Y(k, v), yq = Y(k, wq), x = X(k), z = Zt(t);
        it.push({ type: 'line', pts: [[x, 0, z], [x, yv, z]], color: '--blue', width: .8, alpha: .35, link: 'weights' });
        blueP.push([x, yv, z]); yelP.push([x, yq, z]);
        if (Math.abs(yq - yv) > .012) it.push({ type: 'line', pts: [[x, yv, z], [x, yq, z]], color: '--red', width: 2.2, alpha: .95, link: 'error' });
      });
    }
    it.push({ type: 'points', pts: blueP, color: '--blue', size: 2.6, link: 'weights' });
    it.push({ type: 'points', pts: yelP, color: '--yellow', size: 2.2, link: 'grid' });
    const tall = 27, small = 23;
    it.push({ type: 'label', at: [X(tall), Y(tall, 95) + .16, Zt(4)], text: 'widest channel', color: '--fg', font: '600 12px system-ui, sans-serif' });
    const sm = S.mode === 'tensor' ? stats.tensor.ch[small] : stats.channel.ch[small];
    it.push({ type: 'label', at: [X(small), PE.lerp(-.25, -.95, S.zoom), Zt(4)], text: `a small channel: ${sm.levels} levels`, color: '--yellow', font: '600 12px system-ui, sans-serif' });
    s3.items = it; s3.redraw();
  }
  build();
  let stop = () => {};
  const tgt = { m: 0, zoom: 0 };     // targets survive an interrupted tween
  function set(next) {
    if (next.mode) { S.mode = next.mode; tgt.m = next.mode === 'tensor' ? 0 : 1; }
    if (next.zoom != null) tgt.zoom = next.zoom;
    const from = { m: S.m, zoom: S.zoom }, to = { m: tgt.m, zoom: tgt.zoom };
    stop(); stop = s3.tween(800, u => { S.m = PE.lerp(from.m, to.m, u); S.zoom = PE.lerp(from.zoom, to.zoom, u); build(); });
  }
  // pe3 sizes from width only; a one-frame width nudge makes its ResizeObserver pick up a new stage height.
  const resize = () => { s3.setView({ dist: fitDist() }); el.style.width = 'calc(100% - 1px)'; requestAnimationFrame(() => { el.style.width = ''; }); };
  return { set, info: () => ({ s: stats[S.mode], other: stats[S.mode === 'tensor' ? 'channel' : 'tensor'], zoom: S.zoom }), state: S, scene: s3,
    redraw: () => s3.redraw(), resize, destroy: () => { stop(); s3.destroy(); } };
};
KIT.chanlandText = i => {
  const s = i.s, f = v => (+v.toPrecision(2)).toString();
  return s.mode === 'tensor'
    ? `<b>Per-tensor grid</b>: one scale S = ${s.T.S.toFixed(3)} for all 32 channels. The median channel spans only ${s.medLevels} of the 256 levels, the narrowest ${s.minLevels}; ${s.nSmall} channels get 16 levels or fewer. Total MSE ${f(s.mse)}. A typical channel's weights are off by ${(100 * s.medRel).toFixed(1)}% of their own size.`
    : `<b>Per-channel grid</b>: 32 scales, one per output channel. Every live channel spans all 256 levels (the two all-zero channels need one). Total MSE ${f(s.mse)}. A typical channel's weights are off by ${(100 * s.medRel).toFixed(2)}% of their own size.`;
};
