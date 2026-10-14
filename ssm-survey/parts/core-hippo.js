/* CoreHippo: HiPPO-LegS online function approximation, computed. Shared by viz-hippo, atlas stage hippo and CoreS4.
   Signal: 1000 samples on [0, 100] (the paper's App. F.8 setup). The state c(t) holds the Legendre coefficients of
   the best polynomial fit of degree < N to the whole history on [0, t].
   Sign trap: Theorem 2 writes dc/dt = -(1/t) A c + (1/t) B f with a POSITIVE A; App. B.3's GBT paragraph drops the minus.
   With Theorem 2's A the bilinear step is (our reading, checked against exact quadrature):
     c_{k+1} = (I + A/(2(k+1)))^-1 [ (I - A/(2k)) c_k + B f_k / k ]
   Start trap: the paper divides by k and never says how to start. We start at k = 1 with c_1 = f_0 e_0 (a constant is
   fit exactly by P_0) and read sample j as past time j (in steps), matching the notes' check. A is lower triangular, so
   degree n only listens to degrees <= n: one run at N = 64 holds every smaller N as its first rows. */
const CoreHippo = (() => {
  const NMAX = 64, L = 1000, TMAX = 100;
  function A(N) {
    const M = [];
    for (let n = 0; n < N; n++) { M.push(new Float64Array(N)); for (let k = 0; k <= n; k++) M[n][k] = n > k ? Math.sqrt((2 * n + 1) * (2 * k + 1)) : n + 1; }
    return M;
  }
  const B = N => Float64Array.from({ length: N }, (_, n) => Math.sqrt(2 * n + 1));
  const AM = A(NMAX), BM = B(NMAX);
  let noiseF = null;
  function noise() {
    if (noiseF) return noiseF;
    // Band-limited noise, as in the paper's Fig. 3: a fixed sum of sines below a band limit (seeded, so it is the same every visit).
    const r = PE.rng(11), g = PE.gauss(r), parts = [];
    for (let i = 0; i < 28; i++) parts.push([g() / Math.sqrt(28) * 1.5, .05 + 1.05 * r(), 6.283 * r()]);
    return (noiseF = x => parts.reduce((s, [a, w, p]) => s + a * Math.sin(w * x + p), 0));
  }
  const SIGNALS = {
    paper: { name: 'test function', f: x => .25 * Math.sin(x) + .5 * Math.sin(x / 3) + Math.sin(x / 7) },
    steps: { name: 'steps', f: x => [.3, 1.2, -.5, .7, -1.1, .4, -.2][Math.min(6, Math.floor(x / 14.3))] },
    chirp: { name: 'chirp', f: x => Math.sin(x * x / 180) },
    noise: { name: 'noise', f: x => noise()(x) },
  };
  const runs = {};
  // One pass at N = 64 for a signal and a discretization; c[(k - 1) * NMAX + n] = c_n at step k (time k * TMAX / L).
  function run(sig, method = 'bilinear') {
    const key = sig + '|' + method; if (runs[key]) return runs[key];
    const f = Float64Array.from({ length: L }, (_, j) => SIGNALS[sig].f(j * TMAX / L));
    const N = NMAX, c = new Float64Array(L * N), r = new Float64Array(N);
    c[0] = f[0];
    for (let k = 1; k < L; k++) {
      const o = (k - 1) * N, w = k * N, fk = f[k];
      if (method === 'euler') {
        for (let n = 0; n < N; n++) { let s = c[o + n]; for (let j = 0; j <= n; j++) s -= AM[n][j] * c[o + j] / k; c[w + n] = s + BM[n] * fk / k; }
      } else {
        for (let n = 0; n < N; n++) { let s = c[o + n]; for (let j = 0; j <= n; j++) s -= AM[n][j] * c[o + j] / (2 * k); r[n] = s + BM[n] * fk / k; }
        const d = 2 * (k + 1);   // forward substitution: (I + A/d) is lower triangular
        for (let n = 0; n < N; n++) { let s = r[n]; for (let j = 0; j < n; j++) s -= AM[n][j] / d * c[w + j]; c[w + n] = s / (1 + AM[n][n] / d); }
      }
    }
    let fmax = 0; for (let j = 0; j < L; j++) fmax = Math.max(fmax, Math.abs(f[j]));
    let vmax = 0; if (method !== 'euler') for (let i = 0; i < c.length; i++) vmax = Math.max(vmax, Math.abs(c[i]));
    return (runs[key] = { c, f, fmax, vmax: vmax || run(sig).vmax });
  }
  function legendre(N, x, P = new Float64Array(N)) {
    P[0] = 1; if (N > 1) P[1] = x;
    for (let n = 1; n < N - 1; n++) P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1);
    return P;
  }
  // Reconstruction from c at step k (window [0, k] in steps): f(x) ~ sum_n c_n sqrt(2n+1) P_n(2x/k - 1) (App. D.3).
  function recon(cv, k, x) {
    const N = cv.length, P = legendre(N, 2 * x / k - 1); let g = 0;
    for (let n = 0; n < N; n++) g += cv[n] * Math.sqrt(2 * n + 1) * P[n];
    return g;
  }
  // A view state mixes two sources {sig, method} with weight u, so preset changes can tween.
  const src = st => [run(st.sig0 || st.sig, st.m0 || st.method), run(st.sig, st.method)];
  function coef(st, k) {
    const [a, b] = src(st), u = st.u ?? 1, o = (k - 1) * NMAX, out = new Float64Array(st.N);
    for (let n = 0; n < st.N; n++) out[n] = a.c[o + n] + (b.c[o + n] - a.c[o + n]) * u;
    return out;
  }
  const sample = (st, j) => { const [a, b] = src(st), u = st.u ?? 1; return a.f[j] + (b.f[j] - a.f[j]) * u; };
  const vmaxOf = st => { const [a, b] = src(st), u = st.u ?? 1; return a.vmax + (b.vmax - a.vmax) * u; };
  const fmaxOf = st => { const [a, b] = src(st), u = st.u ?? 1; return a.fmax + (b.fmax - a.fmax) * u; };
  function rmse(st, k = st.k) {
    const cv = coef(st, k); let se = 0;
    for (let j = 0; j < k; j++) { const e = recon(cv, k, j) - sample(st, j); se += e * e; }
    return Math.sqrt(se / k);
  }
  const fmt = v => !isFinite(v) ? 'inf' : v >= 1e4 ? v.toExponential(1).replace('e+', 'e') : v >= 10 ? v.toFixed(1) : v.toFixed(v >= 1 ? 2 : v >= .1 ? 3 : 4);

  // 3D: the coefficients as a surface over (time, degree); the current state c(t) as a yellow slice; input in front.
  function items(st) {
    const css = PE.css, mix = PE3.mix, N = st.N, vmax = vmaxOf(st), fr = fmaxOf(st);
    const X = t => -2.1 + 4.2 * t / TMAX, D = .8 + 1.4 * (N - 4) / 60, Zf = .5, Z = n => Zf - (N > 1 ? n * D / (N - 1) : 0);
    const Y0 = -.25, HH = .82;
    const soft = v => { const z = v / vmax, a = Math.abs(z); return Math.sign(z) * (a <= 1 ? a : 1 + .3 * Math.tanh((a - 1) / .3)); };
    const tk = k => k * TMAX / L, kNow = st.k;
    const nx = Math.round(PE.clamp(2000 / Math.max(1, N - 1), 24, 64));
    // Columns denser early, where the window is short and the coefficients move fast; each sits at its true time.
    const ks = []; for (let i = 0; i <= nx; i++) ks.push(Math.max(1, Math.round(1 + (L - 1) * Math.pow(i / nx, 1.6))));
    const cols = ks.map(k => coef(st, k));
    const bgc = css('--panel2'), yel = css('--yellow'), red = css('--red');
    const out = [];
    for (let i = 0; i < nx; i++) {
      const future = ks[i] >= kNow, al = future ? .2 : .93;
      for (let n = 0; n < N - 1; n++) {
        const v = [cols[i][n], cols[i + 1][n], cols[i + 1][n + 1], cols[i][n + 1]];
        const pts = [[X(tk(ks[i])), Y0 + HH * soft(v[0]), Z(n)], [X(tk(ks[i + 1])), Y0 + HH * soft(v[1]), Z(n)], [X(tk(ks[i + 1])), Y0 + HH * soft(v[2]), Z(n + 1)], [X(tk(ks[i])), Y0 + HH * soft(v[3]), Z(n + 1)]];
        const m = (Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]) + Math.abs(v[3])) / 4;
        const fill = m > 1.05 * vmax ? red : mix(bgc, yel, .14 + .6 * Math.min(1, Math.sqrt(m / vmax)));
        out.push({ type: 'quad', pts, fill, stroke: mix(bgc, yel, .3), alpha: al, link: 'surface' });
      }
    }
    if (N === 1) out.push({ type: 'line', pts: ks.map((k, i) => [X(tk(k)), Y0 + HH * soft(cols[i][0]), Z(0)]), color: '--yellow', width: 2 });
    // Axes on the floor: time along the front, degree up the left side.
    const yf = Y0 - .02;
    out.push({ type: 'line', pts: [[X(0), yf, Zf + .22], [X(TMAX) + .25, yf, Zf + .22]], color: '--muted', width: 1.3, arrow: true, link: 'time' });
    out.push({ type: 'line', pts: [[X(0) - .12, yf, Zf], [X(0) - .12, yf, Z(N - 1) - .05]], color: '--muted', width: 1.3, arrow: true, link: 'A' });
    out.push({ type: 'label', at: [X(TMAX) + .3, yf, Zf + .22], text: 'time t', color: '--muted', dx: 4, dy: 14, link: 'time' });
    out.push({ type: 'label', at: [X(0) - .12, yf, Zf], text: 'n = 0', color: '--muted', dx: -6, align: 'right', link: 'A' });
    out.push({ type: 'label', at: [X(0) - .12, yf, Z(N - 1)], text: st.narrow ? `n = ${N - 1}` : `degree n = ${N - 1}`, color: '--muted', dx: -6, align: 'right', link: 'A' });
    // Input f along the front edge, bright up to now.
    const fin = [], fut = [], zi = Zf + .5;
    for (let j = 0; j < L; j += 4) (j <= kNow ? fin : fut).push([X(tk(j)), Y0 + .34 * sample(st, j) / fr, zi]);
    if (fut.length && fin.length) fut.unshift(fin[fin.length - 1]);
    if (fin.length > 1) out.push({ type: 'line', pts: fin, color: '--blue', width: 2, link: 'input' });
    if (fut.length > 1) out.push({ type: 'line', pts: fut, color: '--blue', width: 1.2, alpha: .3, link: 'input' });
    out.push({ type: 'label', at: [X(0), Y0 + .34 * sample(st, 0) / fr, zi], text: 'input f', color: '--blue', dx: -8, align: 'right', link: 'input' });
    // The state now: one value per degree at the current time.
    const cn = coef(st, kNow), xs = X(tk(kNow)), sl = [];
    for (let n = 0; n < N; n++) sl.push([xs, Y0 + HH * soft(cn[n]) + .012, Z(n)]);
    out.push({ type: 'line', pts: sl, color: '--yellow', width: 3.2, link: 'state' });
    out.push({ type: 'points', pts: sl, color: '--yellow', size: N > 24 ? 2.6 : 4.5, link: 'state' });
    out.push({ type: 'line', pts: [[xs, yf, Zf + .22], [xs, yf, Z(N - 1)]], color: '--yellow', width: 1, dash: [3, 3], alpha: .6, link: 'state' });
    out.push({ type: 'label', at: sl[0], text: st.narrow ? 'c(t)' : 'state now: c(t)', color: '--yellow', dy: -14, link: 'state' });
    return out;
  }

  // 2D: the true history vs its reconstruction from the N numbers in c(t), plus the state register, A and B.
  function draw(ctx, W, H, st, glow = () => 0) {
    const C = n => PE.css(n), narrow = W < 520, N = st.N, k = st.k, t = k * TMAX / L;
    ctx.clearRect(0, 0, W, H);
    const b = narrow ? { x: 38, y: 30, w: W - 50, h: Math.round(H * .56) } : { x: 48, y: 24, w: W - 48 - 214, h: H - 62 };
    const fr = fmaxOf(st) * 1.25;
    const ax = PE.axes(ctx, b, [0, t], [-fr, fr], { xt: [0, t / 2, t], xf: v => v.toFixed(v < 10 && v > 0 ? 1 : 0), yt: [-1, 0, 1].map(v => +(v * fr * .8).toFixed(1)), xl: narrow ? 'past time x in [0, t]' : 'past time x, over the whole window [0, t]' });
    const X = ax.X, Y = ax.Y, cv = coef(st, k);
    const M = Math.min(360, k), xsS = [], fpts = [], gpts = [];
    for (let i = 0; i < M; i++) { const j = Math.round(i * (k - 1) / Math.max(1, M - 1)); xsS.push(j); fpts.push([X(j * TMAX / L), Y(sample(st, j))]); gpts.push([X(j * TMAX / L), Y(PE.clamp(recon(cv, k, j), -fr * 1.6, fr * 1.6))]); }
    ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
    // Red: the error between history and reconstruction.
    ctx.globalAlpha = .22 + .2 * glow('recon'); ctx.fillStyle = C('--red'); ctx.beginPath();
    fpts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); for (let i = gpts.length - 1; i >= 0; i--) ctx.lineTo(gpts[i][0], gpts[i][1]); ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    PE.polyline(ctx, fpts, C('--blue'), 1.8 + 2.5 * glow('input'));
    PE.polyline(ctx, gpts, C('--yellow'), 2.2 + 2.5 * glow('recon') + 1.5 * glow('state'));
    ctx.restore();
    // The window bracket: LegS weighs all of [0, t] equally.
    const wy = b.y - 8, gw = glow('time');
    ctx.save(); ctx.strokeStyle = C('--muted'); ctx.globalAlpha = .7 + .3 * gw; ctx.lineWidth = 1 + 2 * gw; ctx.beginPath();
    ctx.moveTo(b.x, wy + 5); ctx.lineTo(b.x, wy); ctx.lineTo(b.x + b.w, wy); ctx.lineTo(b.x + b.w, wy + 5); ctx.stroke(); ctx.restore();
    PE.label(ctx, `window [0, ${t.toFixed(t < 10 ? 1 : 0)}]: every past moment weighs the same`, b.x + 4, wy - 4, C('--muted'), narrow ? 10 : 11);
    const e = rmse(st), m = st.method === 'euler';
    PE.label(ctx, `RMSE ${fmt(e)}`, b.x + b.w - 6, b.y + b.h - 8, C('--red'), 12, 'right');
    if (!narrow) { PE.label(ctx, 'history f', b.x + 6, b.y + 14, C('--blue'), 11); PE.label(ctx, `rebuilt from ${N} numbers`, b.x + 70, b.y + 14, C('--yellow'), 11); }
    // State register, A and B: the right column on desktop, a row below on phones.
    const vmax = vmaxOf(st), panel = C('--panel2'), yel = C('--yellow');
    const rx = narrow ? b.x : W - 196, ry = narrow ? b.y + b.h + 56 : 30;
    const cols = Math.min(N, 16), rows = Math.ceil(N / cols), cs = Math.min(narrow ? (W * .5 - 24) / cols : 172 / cols, 22);
    PE.label(ctx, `state c(t): ${N} numbers`, rx, ry - 8, yel, 11);
    for (let n = 0; n < N; n++) {
      const cx = rx + (n % cols) * cs, cy = ry + Math.floor(n / cols) * cs, a = Math.min(1, Math.abs(cv[n]) / vmax);
      ctx.fillStyle = Math.abs(cv[n]) > 1.05 * vmax ? C('--red') : PE.mix(panel, yel, .12 + .88 * Math.sqrt(a)); ctx.fillRect(cx + .5, cy + .5, cs - 1.5, cs - 1.5);
    }
    if (glow('state') > 0) { ctx.save(); ctx.strokeStyle = yel; ctx.lineWidth = 2.5 * glow('state'); ctx.strokeRect(rx - 3, ry - 3, cols * cs + 4, rows * cs + 4); ctx.restore(); }
    // A: lower triangular, so degree n reads only degrees 0..n. B: one input weight per degree.
    const ms = narrow ? Math.min(64, W * .3) : 96, axx = narrow ? W - ms - 34 : rx, ayy = narrow ? ry : ry + rows * cs + 34, s = ms / N, gA = glow('A'), gB = glow('B');
    PE.label(ctx, 'A', axx + ms / 2, ayy - 6, C('--fg'), 11 + 3 * gA, 'center');
    ctx.fillStyle = panel; ctx.fillRect(axx, ayy, ms, ms);
    const amax = Math.log(1 + 2 * NMAX);
    for (let n = 0; n < N; n++) for (let j = 0; j <= n; j++) {
      ctx.fillStyle = PE.mix(panel, C('--fg'), Math.min(1, .25 + .45 * Math.log(1 + AM[n][j]) / amax + .3 * gA)); ctx.fillRect(axx + j * s, ayy + n * s, Math.max(s - (N > 24 ? 0 : 1), .8), Math.max(s - (N > 24 ? 0 : 1), .8));
    }
    const bx = axx + ms + 12;
    PE.label(ctx, 'B', bx + 5, ayy - 6, C('--fg'), 11 + 3 * gB, 'center');
    for (let n = 0; n < N; n++) { ctx.fillStyle = PE.mix(panel, C('--blue'), Math.min(1, .3 + .55 * Math.sqrt((2 * n + 1) / (2 * N - 1)) + .3 * gB)); ctx.fillRect(bx, ayy + n * s, 10, Math.max(s - (N > 24 ? 0 : 1), .8)); }
    if (!narrow) PE.label(ctx, 'zero above the diagonal', axx, ayy + ms + 14, C('--muted'), 10);
    if (m) PE.label(ctx, 'forward Euler', b.x + b.w - 6, b.y + b.h - 24, C('--red'), 11, 'right');
  }

  function peakOf(st) {
    const r = run(st.sig, st.method); let peak = 0, at = 1;
    for (let k = 1; k <= L; k++) for (let n = 0; n < st.N; n++) { const a = Math.abs(r.c[(k - 1) * NMAX + n]); if (a > peak) { peak = a; at = k; } }
    return { peak, at };
  }
  function readout(st) {
    const e = rmse(st), r = peakOf(st), t = st.k * TMAX / L;
    let s = `t = ${t.toFixed(t < 10 ? 1 : 0)}, N = ${st.N}, ${st.method === 'euler' ? 'forward Euler' : 'bilinear'} update: ${st.N} numbers rebuild the ${st.k.toLocaleString('en-US')} samples seen so far with RMSE ${fmt(e)}. A KV cache would keep all ${st.k.toLocaleString('en-US')}.`;
    if (st.method === 'euler' && r.peak > 1e3) s += ` On the way a coefficient reached ${fmt(r.peak)} (step ${r.at}) before shrinking back.`;
    return s;
  }
  return { NMAX, L, TMAX, A, B, SIGNALS, run, legendre, recon, coef, rmse, peakOf, items, draw, readout, fmt };
})();
