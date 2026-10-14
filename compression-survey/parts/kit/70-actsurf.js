/* viz-actsurf: a toy linear layer Y = X W shaped after SmoothQuant Fig. 4 (OPT-13B), as read off the plot: three
   outlier channels that persist across every token (two near 78 and 72, one lower at 30), two small spikes, and a floor
   near 1 or less. 24 tokens x 48 input channels; W is 48 input channels x 24 outputs, flat and small (|w| near 0.1, as
   flat as Fig. 4's original weights). The arithmetic is
   exact: per-tensor symmetric INT8 (SmoothQuant Eq. 1), LLM.int8() (vector-wise scales plus FP16 outlier channels,
   Dettmers Eq. 7-8, threshold 6.0), and SmoothQuant smoothing s_j = max|X_j|^a / max|W_j|^(1-a) (Eq. 3-4). */
KIT.ACT = { C: 48, T: 24, CO: 24, ridges: { 6: -78, 20: 30, 41: -72 }, spikes: { 3: 4.5, 33: 3.5 }, thr: 6 };
KIT.actData = (() => {
  let cache = null;
  return () => {
    if (cache) return cache;
    const A = KIT.ACT, r = PE.rng(13), g = PE.gauss(r), X = [], W = [];
    for (let t = 0; t < A.T; t++) {
      const row = [];
      for (let j = 0; j < A.C; j++) {
        let v = g() * .45;
        if (A.ridges[j] != null) v = A.ridges[j] * (1 + .06 * g());
        else if (A.spikes[j] != null && r() < .3) v = A.spikes[j] * (.8 + .4 * r()) * (r() < .5 ? -1 : 1);
        row.push(v);
      }
      X.push(row);
    }
    for (let j = 0; j < A.C; j++) { const k = .95 + .1 * r(), row = []; for (let o = 0; o < A.CO; o++) row.push((r() < .5 ? -1 : 1) * k * (.1 + .005 * g())); W.push(row); }
    const mx = Array.from({ length: A.C }, (_, j) => Math.max(...X.map(rw => Math.abs(rw[j]))));
    const mw = W.map(rw => Math.max(...rw.map(Math.abs)));
    cache = { X, W, mx, mw }; return cache;
  };
})();
KIT.actQ8 = (v, D) => D > 0 ? Math.max(-127, Math.min(127, Math.round(v / D))) * D : 0;
KIT.actScale = alpha => { const { mx, mw } = KIT.actData(); return mx.map((m, j) => Math.pow(m, alpha) / Math.pow(mw[j], 1 - alpha)); };
// mode: 'tensor' (s given, per-tensor INT8 on both) or 'int8' (LLM.int8(): s = 1, outlier channels kept in FP16).
KIT.actStats = (mode, s) => {
  const A = KIT.ACT, { X, W, mx } = KIT.actData(), C = A.C, T = A.T, CO = A.CO;
  if (!s || mode === 'int8') s = mx.map(() => 1);
  const Xh = X.map(r => r.map((v, j) => v / s[j])), Wh = W.map((r, j) => r.map(v => v * s[j]));
  const out = new Set(mode === 'int8' ? mx.map((m, j) => m > A.thr ? j : -1).filter(j => j >= 0) : []);
  let Xq, Wq;
  if (mode === 'int8') {
    Xq = Xh.map(r => { const D = Math.max(...r.filter((v, j) => !out.has(j)).map(Math.abs)) / 127; return r.map((v, j) => out.has(j) ? v : KIT.actQ8(v, D)); });
    const Dw = Array.from({ length: CO }, (_, o) => Math.max(...Wh.filter((r, j) => !out.has(j)).map(r => Math.abs(r[o]))) / 127);
    Wq = Wh.map((r, j) => r.map((v, o) => out.has(j) ? v : KIT.actQ8(v, Dw[o])));
  } else {
    const Dx = Math.max(...Xh.map(r => Math.max(...r.map(Math.abs)))) / 127, Dw = Math.max(...Wh.map(r => Math.max(...r.map(Math.abs)))) / 127;
    Xq = Xh.map(r => r.map(v => KIT.actQ8(v, Dx))); Wq = Wh.map(r => r.map(v => KIT.actQ8(v, Dw)));
  }
  const rel = (a, b) => { let e = 0, p = 0; a.forEach((v, i) => { e += (v - b[i]) ** 2; p += v * v; }); return p > 0 ? Math.sqrt(e / p) : 0; };
  const xErr = Array.from({ length: C }, (_, j) => rel(Xh.map(r => r[j]), Xq.map(r => r[j])));
  const wErr = Wh.map((r, j) => rel(r, Wq[j]));
  const med = a => { const b = a.slice().sort((x, y) => x - y); return (b[(b.length - 1) >> 1] + b[b.length >> 1]) / 2; };
  let e = 0, p = 0;
  for (let t = 0; t < T; t++) for (let o = 0; o < CO; o++) {
    let y = 0, yq = 0; for (let j = 0; j < C; j++) { y += X[t][j] * W[j][o]; yq += Xq[t][j] * Wq[j][o]; }
    e += (yq - y) ** 2; p += y * y;
  }
  const chMax = Array.from({ length: C }, (_, j) => Math.max(...Xh.map(r => Math.abs(r[j]))));
  const xMax = Math.max(...chMax), wMax = Math.max(...Wh.map(r => Math.max(...r.map(Math.abs))));
  // SmoothQuant Sec. 3: a channel with max m_i under a per-tensor max m gets about 2^8 m_i / m levels.
  const levels = 256 * med(chMax) / xMax;
  return { s, Xh, Wh, out, xErr, wErr, xMed: med(xErr), wMed: med(wErr), yErr: Math.sqrt(e / p), xMax, wMax, levels };
};
KIT.actsurf = (el, opts = {}) => {
  const A = KIT.ACT, D = KIT.actData();
  const S = { mode: 'orig', alpha: .5, lg: D.mx.map(() => 0), m8: 0 };   // lg: log s_j shown; m8: 0..1 LLM.int8() blend
  const target = () => S.mode === 'smooth' ? KIT.actScale(S.alpha).map(Math.log) : D.mx.map(() => 0);
  const XS = .074, X = j => (j - (A.C - 1) / 2) * XS, ZX = t => .16 + t * .064, ZW = o => -.16 - o * .064, H = 1.05;
  const narrow = () => (el.clientWidth || 600) < 560;
  const dist = () => opts.dist ? opts.dist(el.clientWidth || 600) : narrow() ? 7.2 : 7.0;
  const s3 = PE3.scene(el, { aspect: opts.aspect || (w => w < 560 ? .9 : .6), yaw: -.62, pitch: .86, dist: dist(), fov: 40, items: [] });
  let pal = {};
  const readPal = () => { pal = { blue: PE.css('--blue'), red: PE.css('--red'), green: PE.css('--green') }; };
  const heat = err => PE3.mix(pal.blue, pal.red, Math.pow(PE.clamp(err / .3, 0, 1), .75));
  let info = null;
  function build() {
    readPal();
    const s = S.lg.map(Math.exp), st = KIT.actStats('tensor', s), s8 = S.m8 > 0 ? KIT.actStats('int8') : null;
    info = S.mode === 'int8' ? s8 : st;
    const it = [];
    // floor outlines of the two patches
    const x0 = X(-.5), x1 = X(A.C - .5);
    it.push({ type: 'line', pts: [[x0, 0, ZX(0)], [x1, 0, ZX(0)], [x1, 0, ZX(A.T - 1)], [x0, 0, ZX(A.T - 1)], [x0, 0, ZX(0)]], color: '--muted', width: 1, alpha: .5 });
    it.push({ type: 'line', pts: [[x0, 0, ZW(0)], [x1, 0, ZW(0)], [x1, 0, ZW(A.CO - 1)], [x0, 0, ZW(A.CO - 1)], [x0, 0, ZW(0)]], color: '--muted', width: 1, alpha: .5 });
    const xh = (t, j) => Math.abs(st.Xh[t][j]) / st.xMax * H, wh = (j, o) => Math.abs(st.Wh[j][o]) / st.wMax * H;
    const keep = j => s8 && s8.out.has(j);
    const xCol = j => keep(j) ? PE3.mix(heat(st.xErr[j]), pal.green, S.m8) : s8 ? PE3.mix(heat(st.xErr[j]), heat(s8.xErr[j]), S.m8) : heat(st.xErr[j]);
    const wCol = j => keep(j) ? PE3.mix(heat(st.wErr[j]), pal.green, S.m8) : s8 ? PE3.mix(heat(st.wErr[j]), heat(s8.wErr[j]), S.m8) : heat(st.wErr[j]);
    for (let t = 0; t < A.T - 1; t++) for (let j = 0; j < A.C - 1; j++) {
      const c = Math.max(xh(t, j), xh(t, j + 1)) > .25 ? (xh(t, j) > xh(t, j + 1) ? j : j + 1) : j;
      it.push({ type: 'quad', pts: [[X(j), xh(t, j), ZX(t)], [X(j + 1), xh(t, j + 1), ZX(t)], [X(j + 1), xh(t + 1, j + 1), ZX(t + 1)], [X(j), xh(t + 1, j), ZX(t + 1)]],
        fill: xCol(c), link: 'act' });
    }
    for (let o = 0; o < A.CO - 1; o++) for (let j = 0; j < A.C - 1; j++) {
      const c = Math.max(wh(j, o), wh(j + 1, o)) > .25 ? (wh(j, o) > wh(j + 1, o) ? j : j + 1) : j;
      it.push({ type: 'quad', pts: [[X(j), wh(j, o), ZW(o)], [X(j + 1), wh(j + 1, o), ZW(o)], [X(j + 1), wh(j + 1, o + 1), ZW(o + 1)], [X(j), wh(j, o + 1), ZW(o + 1)]],
        fill: wCol(c), link: 'wt' });
    }
    // crest lines on the outlier channels: faint, lit by the 'outlier' link, green once LLM.int8() keeps them in FP16
    Object.keys(A.ridges).forEach(k => { const j = +k;
      it.push({ type: 'line', pts: Array.from({ length: A.T }, (_, t) => [X(j), xh(t, j) + .01, ZX(t)]), color: S.m8 > .5 ? '--green' : '--fg', width: 2, alpha: .12 + .6 * S.m8, link: 'outlier' }); });
    // the smoothing scale s_j, drawn along the channel axis between the two patches (log scale, 0 = no scaling)
    const lmax = Math.max(1e-9, ...S.lg.map(Math.abs));
    const sp = S.lg.map((l, j) => [X(j), .03 + .3 * Math.abs(l) / Math.max(lmax, Math.log(4)), 0]);
    it.push({ type: 'line', pts: sp, color: '--fg', width: 1.6, alpha: S.mode === 'smooth' ? .85 : .25, link: 's' });
    const lab = (at, text, color, align, size) => it.push({ type: 'label', at, text, color, align: align || 'center', font: `600 ${size || 12}px system-ui, sans-serif` });
    const n = narrow();
    const fx = st.xMax < 10 ? 1 : 0, fw = st.wMax < 1 ? 2 : 1;
    lab([X(0), 0, ZX(A.T - 1) + .15], (n ? '|X|, max ' : 'activations |X|, max ') + KIT.fmt(st.xMax, fx), '--fg', 'left');
    lab([X(0), H + .2, ZW(A.CO - 1)], (n ? '|W|, max ' : 'weights |W|, max ') + KIT.fmt(st.wMax, fw), '--fg', 'left');
    const rj = 41, ry = Math.abs(st.Xh[A.T >> 1][rj]) / st.xMax * H;
    lab([X(rj), ry + .32, ZX(A.T / 2)], S.mode === 'int8' ? 'kept in FP16' : n ? 'outlier' : 'outlier channel', S.mode === 'int8' ? '--green' : '--fg', 'center', 11);
    if (S.mode === 'smooth') lab([X(14), .5, 0], 'scale s', '--fg', 'center', 11);
    const Y0 = opts.lift || 0;
    if (Y0) it.forEach(o => { if (o.pts) o.pts = o.pts.map(q => [q[0], q[1] + Y0, q[2]]); if (o.at) o.at = [o.at[0], o.at[1] + Y0, o.at[2]]; });
    s3.items = it; s3.redraw();
  }
  build();
  const offTheme = PE.onTheme(build);
  let stop = () => {};
  function set(next, ms) {
    if (next.alpha != null) S.alpha = next.alpha;
    if (next.mode) S.mode = next.mode;
    const from = { lg: S.lg.slice(), m8: S.m8 }, to = { lg: target(), m8: S.mode === 'int8' ? 1 : 0 };
    stop();
    const step = u => { S.lg = from.lg.map((l, j) => PE.lerp(l, to.lg[j], u)); S.m8 = PE.lerp(from.m8, to.m8, u); build(); };
    if (ms === 0) { step(1); stop = () => {}; } else stop = s3.tween(ms || 700, step);
  }
  const resize = () => { s3.setView({ dist: dist() }); el.style.width = 'calc(100% - 1px)'; requestAnimationFrame(() => { el.style.width = ''; }); };
  return { set, state: S, scene: s3, info: () => Object.assign({ mode: S.mode, alpha: S.alpha }, S.mode === 'int8' ? KIT.actStats('int8') : KIT.actStats('tensor', target().map(Math.exp))),
    redraw: () => s3.redraw(), resize, destroy: () => { stop(); offTheme(); s3.destroy(); } };
};
KIT.actsurfText = i => {
  const p = v => (100 * v).toFixed(v < .1 ? 1 : 0) + '%', f = (v, d) => KIT.fmt(v, d);
  const tail = ` Output <i>XW</i> is off by ${(100 * i.yErr).toFixed(2)}%.`;
  if (i.mode === 'int8') return `<b class="c-green">LLM.int8()</b>: the ${i.out.size} channels with a value above <span data-claim="dettmers-outlier-threshold">6.0</span> stay in FP16; ` +
    `the rest use one INT8 scale per token row of X and per output column of W. A typical activation channel is off by ${p(i.xMed)} of its own size, a typical weight row by ${p(i.wMed)}.` + tail + ' The price is a second, FP16 matrix multiply.';
  if (i.mode === 'orig') return `<b>One INT8 scale per tensor, no smoothing</b>: one step is max|X| / 127 = ${f(i.xMax / 127, 2)}. ` +
    `A typical activation channel spans about ${f(i.levels, i.levels < 10 ? 1 : 0)} of the 256 levels, so it is off by ${p(i.xMed)} of its own size. A typical weight row is off by ${p(i.wMed)}.` + tail;
  return `<b>SmoothQuant, alpha = ${i.alpha.toFixed(2)}</b>: max|X| is now ${f(i.xMax, 1)} and max|W| is ${f(i.wMax, 2)}. ` +
    `A typical activation channel spans about ${f(i.levels, i.levels < 10 ? 1 : 0)} of the 256 levels and is off by ${p(i.xMed)}; a typical weight row is off by ${p(i.wMed)}.` + tail;
};
