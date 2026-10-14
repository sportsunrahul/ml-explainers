/* ch-sphere toy: 90 "images" in 6 hidden classes, a linear encoder z = normalize(W x + b) onto the sphere S^2.
   Input x (6 dims) = [class direction * MU + small within-class noise (3 dims), instance-specific detail (3 dims)].
   Augmentation adds noise: small on the class dims (AUG), larger on the instance dims (AUGI), so class features are
   the more stable ones. 'align' trains on L_align only; 'both' trains InfoNCE (all 180 views, in-batch negatives) at
   temperature tau. Adam, 300 steps. Pure math in KIT.sph*; settings and the numbers the reader sees: notes/sim-sphere.txt. */
KIT.SPH = { C: 6, NPER: 15, MU: 2.5, WIN: .25, INST: 1.5, AUG: .15, AUGI: .3, LR: .05, STEPS: 300, SEED: 6, BI: 1, T: 2 };
KIT.SPH_TAUS = [.1, .5, 1];
KIT.sphData = () => {
  const P = KIT.SPH, g = PE.gauss(PE.rng(P.SEED)), X = [], cls = [];
  const mu = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  for (let c = 0; c < P.C; c++) for (let i = 0; i < P.NPER; i++) {
    X.push(mu[c].map(v => P.MU * v + P.WIN * g()).concat([0, 0, 0].map(() => P.INST * g()))); cls.push(c); }
  const W = [0, 1, 2].map(() => Array.from({ length: 6 }, () => .3 * g()));
  const b = [0, 0, 0].map(() => P.BI * g());
  return { X, cls, W, b };
};
KIT.sphEnc = (W, b, x) => { const h = W.map((r, k) => r.reduce((s, w, d) => s + w * x[d], b[k])), n = Math.hypot(h[0], h[1], h[2]); return { n, z: h.map(v => v / n) }; };
KIT.sphAug = (x, g) => x.map((v, d) => v + (d < 3 ? KIT.SPH.AUG : KIT.SPH.AUGI) * g());
// One Adam step on (W, b). The augmentation noise comes from st.g, reseeded per run, so every run is identical.
KIT.sphStep = (st, mode, tau) => {
  const { X, W, b } = st, N = X.length, M = 2 * N;
  const V = []; for (let k = 0; k < 2; k++) for (let i = 0; i < N; i++) V.push(KIT.sphAug(X[i], st.g));
  const E = V.map(x => KIT.sphEnc(W, b, x)), gz = E.map(() => [0, 0, 0]);
  if (mode === 'align') {
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) { const d = E[i].z[k] - E[i + N].z[k]; gz[i][k] += 2 * d / N; gz[i + N][k] -= 2 * d / N; }
  } else {
    for (let a = 0; a < M; a++) {
      const p = a < N ? a + N : a - N, s = []; let mx = -Infinity;
      for (let k = 0; k < M; k++) { s[k] = k === a ? -Infinity : (E[a].z[0] * E[k].z[0] + E[a].z[1] * E[k].z[1] + E[a].z[2] * E[k].z[2]) / tau; if (s[k] > mx) mx = s[k]; }
      let Z = 0; for (let k = 0; k < M; k++) if (k !== a) Z += Math.exp(s[k] - mx);
      for (let k = 0; k < M; k++) { if (k === a) continue; const c = (Math.exp(s[k] - mx) / Z - (k === p ? 1 : 0)) / tau / M;
        for (let j = 0; j < 3; j++) { gz[a][j] += c * E[k].z[j]; gz[k][j] += c * E[a].z[j]; } }
    }
  }
  const gW = W.map(r => r.map(() => 0)), gb = [0, 0, 0];
  E.forEach((e, i) => { const zd = e.z[0] * gz[i][0] + e.z[1] * gz[i][1] + e.z[2] * gz[i][2];
    for (let k = 0; k < 3; k++) { const gh = (gz[i][k] - e.z[k] * zd) / e.n; gb[k] += gh; for (let d = 0; d < 6; d++) gW[k][d] += gh * V[i][d]; } });
  st.t++;
  const adam = (key, grad, get, set) => { const m = st.m[key] = .9 * (st.m[key] || 0) + .1 * grad, v = st.v[key] = .999 * (st.v[key] || 0) + .001 * grad * grad;
    set(get() - KIT.SPH.LR * (m / (1 - .9 ** st.t)) / (Math.sqrt(v / (1 - .999 ** st.t)) + 1e-8)); };
  for (let k = 0; k < 3; k++) { for (let d = 0; d < 6; d++) adam(k * 6 + d, gW[k][d], () => W[k][d], v => { W[k][d] = v; }); adam(18 + k, gb[k], () => b[k], v => { b[k] = v; }); }
};
KIT.sphRun = () => Object.assign(KIT.sphData(), { g: PE.gauss(PE.rng(KIT.SPH.SEED + 100)), m: {}, v: {}, t: 0, step: 0 });
// L_align (alpha = 2) on fresh views, L_uniform (t = 2) and tolerance (mean same-class cosine, Wang & Liu Eq. 11) on clean inputs.
KIT.sphMetrics = st => {
  const { X, cls, W, b } = st, N = X.length, g = PE.gauss(PE.rng(99)), d2 = (a, c) => (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2 + (a[2] - c[2]) ** 2;
  const Z = X.map(x => KIT.sphEnc(W, b, x).z);
  let la = 0; X.forEach(x => { la += d2(KIT.sphEnc(W, b, KIT.sphAug(x, g)).z, KIT.sphEnc(W, b, KIT.sphAug(x, g)).z) / N; });
  let gs = 0, np = 0, tol = 0, nt = 0;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { gs += Math.exp(-KIT.SPH.T * d2(Z[i], Z[j])); np++;
    if (cls[i] === cls[j]) { tol += Z[i][0] * Z[j][0] + Z[i][1] * Z[j][1] + Z[i][2] * Z[j][2]; nt++; } }
  return { align: la, unif: Math.log(gs / np), tol: tol / nt };
};
KIT.sphTrain = (mode, tau) => { const st = KIT.sphRun(); if (mode) while (st.step < KIT.SPH.STEPS) { KIT.sphStep(st, mode, tau); st.step++; } return st; };

// Drawing (PE3): blue dot = view A of each image, yellow dot = view B, green line = the pair; class 1 drawn larger.
KIT.sphere = (canvas, opts = {}) => {
  const STEPS = KIT.SPH.STEPS;
  const state = { mode: null, tau: .5, st: KIT.sphRun() };
  const s3 = PE3.scene(canvas, { aspect: w => KIT.narrow(w) ? 1 : .62, fov: 34, dist: 4.4, yaw: .7, pitch: .38 });
  const ring = (f, n = 64) => Array.from({ length: n + 1 }, (_, i) => f(2 * Math.PI * i / n));
  const wire = [
    ring(a => [Math.cos(a), 0, Math.sin(a)]),
    ...[0, Math.PI / 3, 2 * Math.PI / 3].map(p => ring(a => [Math.cos(a) * Math.cos(p), Math.sin(a), Math.cos(a) * Math.sin(p)])),
    ...[-.6, .6].map(y => ring(a => [Math.cos(a) * Math.sqrt(1 - y * y), y, Math.sin(a) * Math.sqrt(1 - y * y)])),
  ].map(pts => ({ type: 'line', pts, color: '--rule', width: 1 }));
  let cancel = () => {};
  const build = () => {
    const { X, cls, W, b } = state.st, g = PE.gauss(PE.rng(77)), small = KIT.narrow(s3.w || canvas.clientWidth || 600);
    const A = X.map(x => KIT.sphEnc(W, b, KIT.sphAug(x, g)).z), B = X.map(x => KIT.sphEnc(W, b, KIT.sphAug(x, g)).z);
    const items = wire.slice();
    A.forEach((p, i) => items.push({ type: 'line', pts: [p, B[i]], color: '--green', width: 2, link: 'align' }));
    const c1 = A.filter((_, i) => cls[i] === 0), rest = A.filter((_, i) => cls[i] !== 0);
    items.push({ type: 'points', pts: B, color: '--yellow', size: 3, alpha: .9, link: 'align' });
    items.push({ type: 'points', pts: rest, color: '--blue', size: 4.2, link: 'uniform' });
    items.push({ type: 'points', pts: c1, color: '--blue', size: 8.5, link: 'uniform' });
    const cm = [0, 1, 2].map(k => c1.reduce((s, p) => s + p[k], 0)), cn = Math.hypot(...cm) || 1;
    const lbl = state.mode === 'align' ? (small ? 'all images here' : 'every image lands here') : (small ? 'one class' : 'big dots: one hidden class');
    items.push({ type: 'label', at: cm.map(v => v / cn * 1.3), text: lbl, color: '--fg', font: (small ? 11 : 12) + 'px system-ui, sans-serif', align: 'center', dy: -6 });
    const head = state.mode === 'align' ? 'alignment only' : state.mode === 'both' ? 'InfoNCE, \u03c4 = ' + state.tau.toFixed(1) : 'before training (\u03c4 = ' + state.tau.toFixed(1) + ')';
    items.push({ type: 'label', at: [0, 1.12, 0], text: head, color: '--muted', font: (small ? 11 : 13) + 'px system-ui, sans-serif', align: 'center', link: 'tau' });
    s3.items = items; s3.redraw();
  };
  const set = (mode, tau, done) => {
    cancel(); state.mode = mode; state.tau = tau; state.st = KIT.sphRun();
    if (!mode) { build(); done && done(); return; }
    const run = n => { while (state.st.step < n) { KIT.sphStep(state.st, mode, tau); state.st.step++; } };
    cancel = PE.animate(3000, u => { run(Math.round(u * STEPS)); build(); }, done);
  };
  build();
  const off = PE.onTheme(() => s3.redraw());
  return { set, state, redraw: build, resize: () => s3.redraw(), destroy: () => { cancel(); off(); s3.destroy(); } };
};
// m = KIT.sphMetrics(st) of the finished run, so the text is written once per click.
KIT.sphereText = (mode, tau, m) => {
  if (!mode) return 'Before training: a random encoder. Press a button to train it for ' + KIT.SPH.STEPS + ' steps.';
  const f = x => (Math.abs(x) < .005 ? 0 : x).toFixed(2);
  const nums = 'L_align ' + f(m.align) + ', L_uniform ' + f(m.unif) + ', same-class similarity ' + f(m.tol) + '.';
  return mode === 'align'
    ? 'Alignment only, after ' + KIT.SPH.STEPS + ' steps: ' + nums + ' L_uniform = 0 is its worst value: every image sits on one point. (\u03c4 plays no part in this loss.)'
    : 'InfoNCE at \u03c4 = ' + tau.toFixed(1) + ', after ' + KIT.SPH.STEPS + ' steps: ' + nums + ' ' + (tau < .3 ? 'Very uniform, but each hidden class is spread far apart.' : tau < .8 ? 'In between: fairly uniform, classes loosely grouped.' : 'Less uniform, but each hidden class stays in a tight group.');
};
