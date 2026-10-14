/* CoreState3d: an N = 3 state h(t) driven by a 1-D input, drawn as a 3D trajectory. Shared by ch-continuous,
   ch-mamba and atlas stages 3 (continuous) and 6 (select). Pure math (3x3 matrix exponential by scaling and squaring,
   zero-order hold) plus pe3 items. Two presets:
   - lti: a fixed A with one real decay mode and one rotating pair. The input is held constant for one time unit per
     value, so ZOH with any step Delta = 1/n lands exactly on the continuous curve (checked against RK4 in node).
   - select: Mamba's real initialization A_n = -(n+1) at N = 3; each token sets its own Delta (softplus of a linear
     map of its importance flag) and its own B gain; C is shared. A dashed LTI path (one Delta for all) is the baseline. */
const CoreState3d = (() => {
  const I = () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const mm = (P, Q) => P.map(r => [0, 1, 2].map(j => r[0] * Q[0][j] + r[1] * Q[1][j] + r[2] * Q[2][j]));
  const mv = (M, v) => M.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
  const msc = (M, s) => M.map(r => r.map(x => x * s));
  const madd = (P, Q, s = 1) => P.map((r, i) => r.map((x, j) => x + s * Q[i][j]));
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function inv3(m) {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C;
    return [[A, c * h - b * i, b * f - c * e], [B, a * i - c * g, c * d - a * f], [C, b * g - a * h, a * e - b * d]].map(r => r.map(x => x / det));
  }
  // exp(M): scale until the norm is below 1/4, sum 16 Taylor terms, square back. Error near 1e-16 for these sizes.
  function expm(M) {
    const n = Math.max(...M.map(r => Math.abs(r[0]) + Math.abs(r[1]) + Math.abs(r[2])));
    let k = n > .25 ? Math.ceil(Math.log2(n / .25)) : 0, E = I(), T = I();
    const S = msc(M, Math.pow(2, -k));
    for (let i = 1; i <= 16; i++) { T = msc(mm(T, S), 1 / i); E = madd(E, T); }
    for (; k > 0; k--) E = mm(E, E);
    return E;
  }
  // Zero-order hold (Mamba Eq. 4): Abar = exp(Delta A), Bbar = (Delta A)^-1 (exp(Delta A) - I) Delta B = A^-1 (Abar - I) B.
  function zoh(A, Ainv, dt, B) { const Ab = expm(msc(A, dt)); return { Ab, Bb: mv(mm(Ainv, madd(Ab, I(), -1)), B) }; }
  const step = (d, h, x) => mv(d.Ab, h).map((v, i) => v + d.Bb[i] * x);
  const softplus = z => Math.log1p(Math.exp(z));
  const lerp = (a, b, u) => a + (b - a) * u, lerp3 = (p, q, u) => [lerp(p[0], q[0], u), lerp(p[1], q[1], u), lerp(p[2], q[2], u)];

  /* ---------- lti preset ---------- */
  const R = .5, AL = .25, W = 1;
  const LTI = { A: [[-R, 0, 0], [0, -AL, -W], [0, W, -AL]], B: [1, 1, .4], C: [.6, .5, .5] };
  LTI.Ainv = inv3(LTI.A);
  const SIGNAL = [1, .6, -.2, -.8, -.5, .4, .9, .3, 0, 0, 0, 0, 0, 0, 0, 0];   // held one time unit each; silent from t = 8
  const DTS = [1 / 20, 1 / 16, 1 / 12, 1 / 10, 1 / 8, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1];
  const SUB = 24;   // curve points per held value
  const cache = {};
  // mode 'signal': the same 16-unit signal sampled every dt (16/dt steps). mode 'tokens': the same 16 values, one per
  // step, each held for dt (the span shrinks to 16 dt).
  function runLTI(mode, dt) {
    const key = mode + dt; if (cache[key]) return cache[key];
    const hold = mode === 'signal' ? 1 : dt, per = mode === 'signal' ? Math.round(1 / dt) : 1, span = SIGNAL.length * hold;
    const d = zoh(LTI.A, LTI.Ainv, dt, LTI.B), samples = [{ t: 0, x: SIGNAL[0], h: [0, 0, 0], y: 0 }], curve = [{ t: 0, h: [0, 0, 0] }];
    let h = [0, 0, 0];
    SIGNAL.forEach((x, j) => {
      const h0 = h;
      for (let i = 1; i <= SUB; i++) { const s = hold * i / SUB, e = zoh(LTI.A, LTI.Ainv, s, LTI.B); curve.push({ t: j * hold + s, h: step(e, h0, x) }); }
      for (let k = 0; k < per; k++) { h = step(d, h, x); samples.push({ t: j * hold + (k + 1) * dt, x, h, y: dot3(LTI.C, h) }); }
    });
    curve.forEach(c => { c.y = dot3(LTI.C, c.h); });
    return (cache[key] = { mode, dt, hold, span, samples, curve, Ab: d.Ab, Bb: d.Bb });
  }
  function readLTI(mode, dt) {
    const r = runLTI(mode, dt), n = r.samples.length - 1, keep = Math.exp(-AL * dt), half = Math.LN2 / (AL * dt);
    const f = v => v >= 10 ? v.toFixed(0) : v.toFixed(v >= 1 ? 1 : 2);
    const head = mode === 'signal'
      ? `Same signal, \u0394 = ${fmtDt(dt)}: ${n} steps cover the 16 time units, and every step lands on the continuous curve.`
      : `Same 16 tokens, \u0394 = ${fmtDt(dt)}: the tokens now span ${f(r.span)} time units.`;
    return `${head} Per step the slow pair keeps ${(100 * keep).toFixed(1)}% of itself and turns ${(W * dt * 180 / Math.PI).toFixed(1)} degrees; it halves every ${f(half)} steps (${f(half * dt)} time units). One step costs the same at any length: a 3 x 3 update, no matter how many tokens came before.`;
  }
  const fmtDt = dt => { const n = Math.round(1 / dt); return n === 1 ? '1' : Math.abs(1 / n - dt) < 1e-9 ? `1/${n}` : dt.toFixed(2); };

  /* ---------- select preset ---------- */
  const SEL = { A: [[-1, 0, 0], [0, -2, 0], [0, 0, -3]], B0: [1, 1, 1], C: [6 / 11, 6 / 11, 6 / 11] };
  SEL.Ainv = inv3(SEL.A);
  const VAL = [.2, .9, -.4, .3, -.5, -.8, .4, -.2, .5, .6, -.3, .1], KEYS = [1, 5, 8];
  const DT_BIAS = -3, DT_W = 5, LTI_DT = .5, KSUB = 14;
  const selDelta = m => softplus(DT_BIAS + DT_W * m);     // filler 0.049, important 2.13
  const selGain = m => .3 + .7 * m;                       // B_t = gain * B0: filler writes at 30%
  function runSel(marks, lti = false) {
    let h = [0, 0, 0];
    const tokens = [], curve = [{ h }], samples = [h];
    VAL.forEach((x, t) => {
      const m = marks[t] ? 1 : 0, dt = lti ? LTI_DT : selDelta(m), B = SEL.B0.map(b => b * (lti ? 1 : selGain(m))), h0 = h;
      for (let i = 1; i <= KSUB; i++) curve.push({ h: step(zoh(SEL.A, SEL.Ainv, dt * i / KSUB, B), h0, x) });
      h = step(zoh(SEL.A, SEL.Ainv, dt, B), h0, x); samples.push(h);
      tokens.push({ x, m, dt, B, h, y: dot3(SEL.C, h), gate: 1 - Math.exp(-dt) });
    });
    return { tokens, curve, samples, lti };
  }
  const keyMarks = () => VAL.map((_, t) => KEYS.includes(t));
  function readSel(marks) {
    const r = runSel(marks), last = r.tokens[r.tokens.length - 1], nI = marks.filter(Boolean).length;
    const lastKey = r.tokens.map((k, i) => k.m ? i : -1).filter(i => i >= 0).pop();
    const mem = lastKey == null ? 'No token is marked, so every \u0394 is small and the state hardly moves.'
      : nI === VAL.length ? 'Every token is marked, so the state jumps to each new token and remembers none of the earlier ones.'
      : `The last important token was ${lastKey + 1} (x = ${VAL[lastKey].toFixed(1)}); after the filler that follows, the output still reads ${last.y.toFixed(2)}.`;
    return `${nI} of ${VAL.length} tokens marked important (\u0394 = ${selDelta(1).toFixed(2)}, slowest-mode gate 1 - exp(-\u0394) = ${(1 - Math.exp(-selDelta(1))).toFixed(2)}); filler gets \u0394 = ${selDelta(0).toFixed(3)} (gate ${(1 - Math.exp(-selDelta(0))).toFixed(3)}). ${mem}`;
  }

  /* ---------- world mapping and pe3 items ---------- */
  // lti: the rotating pair (h2, h3) faces the camera and the decay mode h1 points toward the viewer.
  // select: h1 (the slowest mode, A = -1) runs left to right. Strips sit below the state on a shared time axis.
  const LAY = {
    lti: { o: [0, .72, 0], map: h => [h[1] * .76, .72 + h[2] * .76, h[0] * .95], YS: -1.62, ZS: .7, KS: .3, X0: -2.3, X1: 2.3 },
    select: { o: [-.9, .62, 0], map: h => [-.9 + h[0] * 2.5, .62 + h[1] * 2.9, h[2] * 3.2], YS: -1.35, ZS: .6, KS: .42, X0: -2.3, X1: 2.3 },
  };
  const SX = (lay, u) => lay.X0 + (lay.X1 - lay.X0) * u;
  const M = 321;   // curve resample count, so any two frames interpolate point by point
  function resample(arr, key) {
    const out = [];
    for (let i = 0; i < M; i++) { const f = i / (M - 1) * (arr.length - 1), j = Math.min(arr.length - 2, Math.floor(f)), u = f - j; out.push(key(arr[j], arr[j + 1], u)); }
    return out;
  }
  // A frame holds world coordinates only; items() interpolates two frames.
  function frameLTI(mode, dt) {
    const r = runLTI(mode, dt), sp = r.span, Y = LAY.lti, X = u => SX(Y, u);
    // With few, short steps the state stays near zero; zoom it (and y) by a whole factor and say so on the figure.
    const big = rr => Math.max(...rr.curve.map(c => Math.hypot(...c.h)));
    const zr = big(runLTI('signal', 1)) / big(r), zoom = zr > 1.5 ? Math.min(12, Math.floor(zr)) : 1;
    const map = h => Y.map(h.map(v => v * zoom)), Ys = v => Y.YS + Y.KS * v, Yz = v => Ys(v * zoom);
    const curve = resample(r.curve, (a, b, u) => map(lerp3(a.h, b.h, u)));
    const out = resample(r.curve, (a, b, u) => [X(lerp(a.t, b.t, u) / sp), Yz(lerp(a.y, b.y, u)), Y.ZS]);
    const inp = [];
    SIGNAL.forEach((x, j) => { inp.push([X(j / SIGNAL.length), Ys(x), Y.ZS], [X((j + 1) / SIGNAL.length), Ys(x), Y.ZS]); });
    const samples = r.samples.map(s => map(s.h));
    const sDots = r.samples.slice(1).map(s => [X(s.t / sp), Yz(s.y), Y.ZS]);
    const xDots = r.samples.slice(1).map(s => [X((s.t - dt) / sp), Ys(s.x), Y.ZS]);
    let rim = 0; curve.forEach((p, i) => { if (p[0] > curve[rim][0]) rim = i; });
    return { kind: 'lti', mode, dt, zoom, n: r.samples.length - 1, curve, out, inp, samples, sDots, xDots, stopX: X(8 * r.hold / sp), rim };
  }
  function frameSel(marks) {
    const r = runSel(marks), g = runSel(marks, true), L = VAL.length, Y = LAY.select, Ys = v => Y.YS + Y.KS * v;
    const tx = t => SX(Y, (t + .5) / L);
    return {
      kind: 'select', marks: marks.slice(), curve: resample(r.curve, (a, b, u) => Y.map(lerp3(a.h, b.h, u))),
      ghost: resample(g.curve, (a, b, u) => Y.map(lerp3(a.h, b.h, u))),
      samples: r.samples.map(h => Y.map(h)), imp: r.tokens.map(k => k.m),
      bars: r.tokens.map((k, t) => ({ x: tx(t), v: k.x, m: k.m, dt: k.dt })),
      outDots: r.tokens.map((k, t) => [tx(t), Ys(k.y), Y.ZS]), ghostOut: g.tokens.map((k, t) => [tx(t), Ys(k.y), Y.ZS]),
    };
  }
  const L3 = (a, b, u) => a.map((p, i) => lerp3(p, b[i], u));
  const F11 = '600 11px system-ui, sans-serif';
  function axesItems(kind, narrow) {
    const o = LAY[kind].o, it = [];
    const lab = kind === 'lti' ? [['h2', [1.6, 0, 0]], ['h3', [0, 1.5, 0]], ['h1, only shrinks', [0, 0, 1.25]]]
      : [['h1, A = -1', [2.6, 0, 0]], ['h2, A = -2', [0, 1.45, 0]], ['h3, A = -3', [0, 0, 1.25]]];
    lab.forEach(([t, e]) => {
      const tip = [o[0] + e[0], o[1] + e[1], o[2] + e[2]], neg = [o[0] - e[0] * .5, o[1] - e[1] * .5, o[2] - e[2] * .5];
      it.push({ type: 'line', pts: [neg, tip], color: '--muted', width: 1, alpha: .55, arrow: true });
      it.push({ type: 'label', at: tip, text: narrow ? t.split(',')[0] : t, color: '--muted', dy: t.includes('shrinks') ? 12 : -9, dx: t.includes('shrinks') ? -6 : 0, align: t.includes('shrinks') ? 'right' : 'center', font: F11, link: t.includes('shrinks') ? 'A' : undefined });
    });
    return it;
  }
  function stripBase(it, lay, text) {
    it.push({ type: 'line', pts: [[lay.X0 - .1, lay.YS, lay.ZS], [lay.X1 + .25, lay.YS, lay.ZS]], color: '--muted', width: 1.1, alpha: .7, arrow: true });
    it.push({ type: 'label', at: [lay.X1 + .2, lay.YS, lay.ZS], text, color: '--muted', dy: 13, font: F11 });
  }
  // st = { from?, to, u, narrow, ghost?, lift? }; frames come from frameLTI or frameSel. lift moves the scene up (atlas).
  function items(st) {
    const it = draw(st), dy = st.lift || 0;
    if (!dy) return it;
    const up = p => [p[0], p[1] + dy, p[2]];
    return it.map(o => o.pts ? { ...o, pts: o.pts.map(up) } : o.at ? { ...o, at: up(o.at) } : o);
  }
  function draw(st) {
    const a = st.from || st.to, b = st.to, u = st.from ? st.u : 1, nar = !!st.narrow, it = [];
    if (b.kind === 'lti') {
      const Y = LAY.lti, o = Y.o;
      it.push(...axesItems('lti', nar));
      // A at work: a turn around the pair's plane and a pull toward zero along h1.
      const arc = []; for (let i = 0; i <= 24; i++) { const q = -.55 + 1.1 * i / 24; arc.push([o[0] + 1.95 * Math.cos(q), o[1] + 1.95 * Math.sin(q), 0]); }
      it.push({ type: 'line', pts: arc, color: '--muted', width: 1.4, arrow: true, link: 'A', alpha: .8 });
      if (!nar) it.push({ type: 'label', at: arc[4], text: 'A turns and shrinks (h2, h3)', color: '--muted', dx: 8, align: 'left', link: 'A', font: F11 });
      const z = lerp(a.zoom, b.zoom, u);
      if (z > 1.05) it.push({ type: 'label', at: [o[0] - 2.3, o[1] + 1.3, 0], text: `h and y drawn ${b.zoom} times larger`, color: '--fg', align: 'left', alpha: Math.min(1, z - 1), font: F11 });
      const cv = L3(a.curve, b.curve, u), seg = 16, per = (M - 1) / seg;
      for (let s = 0; s < seg; s++) it.push({ type: 'line', pts: cv.slice(s * per, (s + 1) * per + 1), color: '--yellow', width: 2.6, alpha: .3 + .7 * (s + 1) / seg, link: 'state' });
      const sz = n => n > 100 ? 2.4 : n > 40 ? 3.2 : 4.6;
      const pts = (f, al) => {
        if (al <= .02) return;
        it.push({ type: 'points', pts: f.samples, color: '--yellow', size: sz(f.n), alpha: al, link: 'samples' });
        it.push({ type: 'points', pts: f.sDots, color: '--green', size: sz(f.n) * .75, alpha: al, link: 'output' });
        it.push({ type: 'points', pts: f.xDots, color: '--blue', size: sz(f.n) * .75, alpha: al, link: 'samples' });
      };
      if (st.from && (a.mode !== b.mode || a.dt !== b.dt)) { pts(a, 1 - u); pts(b, u); } else pts(b, 1);
      stripBase(it, Y, 'time');
      it.push({ type: 'line', pts: b.inp, color: '--blue', width: 2, link: 'input' });
      it.push({ type: 'line', pts: L3(a.out, b.out, u), color: '--green', width: 2.2, link: 'output' });
      const sx = lerp(a.stopX, b.stopX, u);
      it.push({ type: 'line', pts: [[sx, Y.YS - .25, Y.ZS], [sx, Y.YS + .45, Y.ZS]], color: '--muted', width: 1, dash: [3, 3], alpha: .6 });
      it.push({ type: 'label', at: [sx, Y.YS - .25, Y.ZS], text: 'input stops', color: '--muted', dy: 10, font: F11 });
      it.push({ type: 'label', at: [Y.X0, Y.YS + Y.KS * 1.1, Y.ZS], text: nar ? 'x(t)' : 'x(t), input', color: '--blue', dy: -10, align: 'left', link: 'input' });
      it.push({ type: 'label', at: [SX(Y, .62), Y.YS + Y.KS * 1.6, Y.ZS], text: nar ? 'y(t)' : 'y(t) = C h(t), output', color: '--green', dy: -8, dx: 6, align: 'left', link: 'output' });
      it.push({ type: 'label', at: cv[b.rim], text: nar ? 'h(t)' : 'h(t), the state', color: '--yellow', dx: 12, align: 'left', link: 'state' });
      return it;
    }
    const Y = LAY.select;
    it.push(...axesItems('select', nar));
    if (st.ghost !== false) {
      it.push({ type: 'line', pts: b.ghost, color: '--muted', width: 1.6, dash: [4, 4], alpha: .8, link: 'lti' });
      if (!nar) it.push({ type: 'label', at: b.ghost[Math.round(M * .45)], text: 'one \u0394 for every token (LTI)', color: '--muted', dx: 10, dy: 12, align: 'left', link: 'lti', font: F11 });
    }
    const cv = L3(a.curve, b.curve, u), seg = VAL.length, per = (M - 1) / seg;
    for (let s = 0; s < seg; s++) it.push({ type: 'line', pts: cv.slice(Math.round(s * per), Math.round((s + 1) * per) + 1), color: '--yellow', width: 2.8, alpha: .35 + .65 * (s + 1) / seg, link: 'state' });
    const sm = L3(a.samples, b.samples, u), big = [], small = [];
    sm.slice(1).forEach((p, t) => (b.imp[t] ? big : small).push(p));
    it.push({ type: 'points', pts: [sm[0]], color: '--muted', size: 4 });
    if (small.length) it.push({ type: 'points', pts: small, color: '--yellow', size: nar ? 3 : 3.6, alpha: .8, link: 'state' });
    if (big.length) it.push({ type: 'points', pts: big, color: '--yellow', size: nar ? 5.5 : 7.5, link: 'write' });
    stripBase(it, Y, 'tokens');
    b.bars.forEach((k, t) => {
      const v = lerp(a.bars[t].v, k.v, u), m = lerp(a.bars[t].m, k.m, u), d = lerp(a.bars[t].dt, k.dt, u);
      it.push({ type: 'line', pts: [[k.x, Y.YS, Y.ZS], [k.x, Y.YS + Y.KS * v, Y.ZS]], color: '--blue', width: nar ? 4 : 7, alpha: .28 + .72 * m, link: 'input' });
      it.push({ type: 'line', pts: [[k.x, Y.YS - .1, Y.ZS], [k.x, Y.YS - .1 - .24 * d, Y.ZS]], color: '--fg', width: nar ? 2.5 : 4, alpha: .6, link: 'delta' });
    });
    it.push({ type: 'line', pts: b.ghostOut, color: '--muted', width: 1.2, dash: [3, 3], alpha: .75, link: 'lti' });
    const od = L3(a.outDots, b.outDots, u);
    it.push({ type: 'line', pts: od, color: '--green', width: 1.8, link: 'output' }, { type: 'points', pts: od, color: '--green', size: nar ? 3 : 4, link: 'output' });
    it.push({ type: 'label', at: [Y.X0, Y.YS + Y.KS * 1.1, Y.ZS], text: nar ? 'x_t' : 'tokens x_t', color: '--blue', dy: -8, align: 'left', link: 'input' });
    it.push({ type: 'label', at: [Y.X0, Y.YS - .35, Y.ZS], text: '\u0394_t', color: '--fg', dx: -8, align: 'right', link: 'delta' });
    it.push({ type: 'label', at: od[od.length - 1], text: nar ? 'y_t' : 'y_t, output', color: '--green', dx: 10, align: 'left', link: 'output' });
    it.push({ type: 'label', at: sm[sm.length - 1], text: nar ? 'h' : 'h_t, the state', color: '--yellow', dx: 10, dy: -10, align: 'left', link: 'state' });
    return it;
  }
  // Camera that fits the width: pe3's field of view is vertical, so a squarer stage needs the camera farther back.
  function view(W, H, kind = 'lti') {
    const narrow = W < 620 || W / H < 1.05, r = Math.max(.3, W / Math.max(H, 1));
    const v = kind === 'lti' ? { yaw: narrow ? .34 : .24, pitch: .18 } : { yaw: narrow ? .4 : .3, pitch: .22 };
    return { narrow, ...v, dist: Math.max(8.2, 2.9 / (.306 * r)) };
  }
  return { expm, zoh, inv3, softplus, LTI, SIGNAL, DTS, runLTI, readLTI, fmtDt, SEL, VAL, KEYS, selDelta, selGain, runSel, readSel, keyMarks, frameLTI, frameSel, items, view, LTI_DT };
})();
