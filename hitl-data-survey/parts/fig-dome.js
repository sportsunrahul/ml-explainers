/* FIG.dome: BALD as a gap on the entropy dome (ch-bald; atlas "disagree").
   The 3-class probability triangle lies flat; the height above each point is its entropy. T dropout samples sit on the
   dome. Their mean (one point on the triangle) has height H[mean]; the samples' average height is E[H]. BALD = the gap.
   Exact math; the samples are a toy. */
FIG.dome = (el, o = {}) => {
  const LOG3 = Math.log(3), HS = 1.2;   // height scale: max entropy -> 1.5 world units
  const corner = [[-1.25, -.72], [1.25, -.72], [0, 1.44]];   // (x, z) of the triangle corners
  const toXZ = p => [p[0] * corner[0][0] + p[1] * corner[1][0] + p[2] * corner[2][0], p[0] * corner[0][1] + p[1] * corner[1][1] + p[2] * corner[2][1]];
  const H = p => KIT.entropy(p) / LOG3;
  const pt = p => { const [x, z] = toXZ(p); return [x, H(p) * HS - .75, z]; };
  const base = p => { const [x, z] = toXZ(p); return [x, -.75, z]; };
  const PRESET = { ambiguous: { m: [.36, .33, .31], s: .12 }, unfamiliar: { m: [.34, .33, .33], s: .92 }, confident: { m: [.9, .06, .04], s: .1 } };
  const S = { preset: 'unfamiliar', T: 12, spread: .92 };
  // Fixed Dirichlet(0.5)-like directions so a preset always draws the same samples.
  const r = PE.rng(41), dirs = [];
  for (let t = 0; t < 40; t++) { const g = [0, 1, 2].map(() => Math.pow(r(), 2.2)), s = g.reduce((a, b) => a + b, 0); dirs.push(g.map(v => v / s)); }
  function samples() {
    const m = PRESET[S.preset].m;
    return dirs.slice(0, S.T).map(d => { const q = m.map((v, k) => (1 - S.spread) * v + S.spread * d[k]); const s = q.reduce((a, b) => a + b, 0); return q.map(v => Math.max(v / s, 1e-6)); });
  }
  // the dome: a triangular mesh in barycentric coordinates
  const N = 22, dome = [];
  const bc = (i, j) => [i / N, j / N, 1 - i / N - j / N];
  for (let i = 0; i < N; i++) for (let j = 0; j < N - i; j++) {
    const a = bc(i, j), b = bc(i + 1, j), c2 = bc(i, j + 1);
    dome.push({ type: 'quad', pts: [pt(a), pt(b), pt(c2), pt(c2)], fill: () => PE.mix(PE.css('--bg'), PE.css('--blue'), .25 + .45 * H(a)), alpha: .34 });
    if (i + j + 2 <= N) { const d = bc(i + 1, j + 1); dome.push({ type: 'quad', pts: [pt(b), pt(d), pt(c2), pt(c2)], fill: () => PE.mix(PE.css('--bg'), PE.css('--blue'), .25 + .45 * H(b)), alpha: .34 }); }
  }
  const s3 = PE3.scene(el, { aspect: o.aspect || (w => w < 620 ? 1.05 : .56), fov: 36, dist: 5.3, yaw: .55, pitch: .5 });
  function stats() { const P = samples(), m = [0, 1, 2].map(k => P.reduce((a, p) => a + p[k], 0) / P.length);
    const hm = KIT.entropy(m), eh = P.reduce((a, p) => a + KIT.entropy(p), 0) / P.length; return { P, m, hm, eh, bald: hm - eh }; }
  function build() {
    const st = stats(), items = dome.slice();
    items.push({ type: 'line', pts: [base([1, 0, 0]), base([0, 1, 0]), base([0, 0, 1]), base([1, 0, 0])], color: '--muted', width: 1 });
    ['circle', 'square', 'triangle'].forEach((n, k) => { const p = [0, 0, 0]; p[k] = 1; items.push({ type: 'label', at: base(p), text: 'all ' + n, color: '--muted', dy: k === 2 ? -10 : 16 }); });
    items.push({ type: 'points', pts: st.P.map(pt), color: '--blue', size: 6, link: 'eh' });
    st.P.forEach(p => items.push({ type: 'line', pts: [base(p), pt(p)], color: '--blue', width: 1, alpha: .3 }));
    const [mx, mz] = toXZ(st.m), yTop = H(st.m) * HS - .75, yAvg = st.eh / LOG3 * HS - .75;
    items.push({ type: 'line', pts: [[mx, -.75, mz], [mx, yAvg, mz]], color: '--muted', width: 1.5, dash: [3, 3] });
    items.push({ type: 'points', pts: [[mx, -.75, mz]], color: '--fg', size: 6 });
    items.push({ type: 'line', pts: [[mx, yAvg, mz], [mx, yTop, mz]], color: '--yellow', width: 5, link: 'gap' });
    items.push({ type: 'points', pts: [[mx, yTop, mz]], color: '--fg', size: 8, link: 'hmean' });
    items.push({ type: 'points', pts: [[mx, yAvg, mz]], color: '--blue', size: 8, link: 'eh' });
    items.push({ type: 'label', at: [mx, yTop, mz], text: 'H[mean]', color: '--fg', dx: 10, dy: -8, link: 'hmean' });
    items.push({ type: 'label', at: [mx, yAvg, mz], text: 'average H', color: '--blue', dx: 10, dy: 14, link: 'eh' });
    if (st.bald > .05) items.push({ type: 'label', at: [mx, (yAvg + yTop) / 2, mz], text: 'BALD', color: '--yellow', dx: -46, link: 'gap' });
    s3.items = items; s3.redraw();
    o.say && o.say(`${S.T} dropout passes. Entropy of the mean prediction ${st.hm.toFixed(2)} nats, average entropy of the passes ${st.eh.toFixed(2)} nats, BALD = gap = ${st.bald.toFixed(2)} nats (maximum ${LOG3.toFixed(2)}).`);
  }
  build(); PE.onTheme(() => s3.redraw());
  return {
    preset(p) { S.preset = p; S.spread = PRESET[p].s; build(); return S.spread; },
    T(t) { S.T = t; build(); }, spread(v) { S.spread = v; build(); },
    reset() { S.preset = 'unfamiliar'; S.spread = .92; S.T = 12; s3.setView({ yaw: .55, pitch: .5, dist: 5.3 }, 0); build(); },
    redraw: () => s3.redraw(), destroy: () => s3.destroy(), state: S };
};
