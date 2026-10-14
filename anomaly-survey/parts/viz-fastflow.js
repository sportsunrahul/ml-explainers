<script>
(() => {
  // ch-fastflow: an exact toy 2D flow (parts/lib-fastflow.js) drawn as a density surface over the feature plane.
  // Checked in build/fastflow-sim.mjs: mass in the +-3.4 box 0.985-0.999 at every stage; log pX(x) of each test point is
  // identical at stages 0, .3, 1, 1.7, 2, 3, 3.5, 4 (anomaly -7.5936, normal -0.9841); the anomaly ends at z = (0.74, 3.50),
  // with 0.25% of training samples scoring lower.
  const F = FASTFLOW_KIT, fig = document.getElementById('viz-fastflow'), el = document.getElementById('cv-fastflow');
  const out = fig.querySelector('.readout');
  const EXT = 3.6, SC = 3 / EXT, HS = 5.5;
  const train = F.sample(260, 7), ref = F.sample(2000, 11).map(x => F.logDensity(x, 0));
  const peakIdx = train.reduce((b, x, i) => F.logDensity(x, 0) > F.logDensity(train[b], 0) ? i : b, 0);
  const circle = c => Array.from({ length: 25 }, (_, i) => [c[0] + .2 * Math.cos(i * Math.PI / 12), c[1] + .2 * Math.sin(i * Math.PI / 12)]);
  const DEF = { stage: 0, test: 'anomaly' };
  const S = { ...DEF };
  const W3 = h => [h[0] * SC, 0, -h[1] * SC];
  let stopT = () => {};

  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .95 : .56, fov: 36, dist: 10.5, yaw: .55, pitch: .62, items: [] });
  function items() {
    const blue = PE.css('--blue'), panel = PE.css('--panel'), st = S.stage;
    const ht = h => HS * Math.exp(F.logDensity(h, st));
    let ymax = 0; for (let i = 0; i <= 34; i++) for (let j = 0; j <= 34; j++) ymax = Math.max(ymax, ht([(-3 + 6 * i / 34) / SC, -(-3 + 6 * j / 34) / SC]));
    const surf = PE3.surface((x, z) => ht([x / SC, -z / SC]),
      { x0: -3, x1: 3, z0: -3, z1: 3, nx: 34, nz: 34, link: 'base', fill: y => PE3.mix(panel, blue, Math.min(1, .1 + .9 * y / ymax)) });
    // Training features sit on the surface, so the opaque surface does not hide them.
    const pts = train.map(x => { const h = F.push(x, st).h; return [h[0] * SC, ht(h) + .03, -h[1] * SC]; });
    const x0 = F.TEST[S.test], p = F.push(x0, st), h = p.h;
    const top = [h[0] * SC, ht(h) + .03, -h[1] * SC], base = W3(h);
    const ring = circle(x0).map(c => W3(F.push(c, st).h));
    const lab = W3(F.push(train[peakIdx], st).h);
    return [
      { type: 'grid', y: 0, size: 6, step: 1, color: '--rule' },
      ...surf,
      { type: 'points', pts, color: '--fg', size: 2, alpha: .7 },
      { type: 'line', pts: ring, color: '--yellow', width: 1.6, link: 'logdet' },
      { type: 'line', pts: [base, top], color: '--yellow', width: 1.4, dash: [3, 3] },
      { type: 'points', pts: [top], color: '--yellow', size: 7, link: 'test' },
      { type: 'label', at: top, text: S.test === 'anomaly' ? 'feature from a scratch' : 'feature from a good part', color: '--yellow', dy: -12, align: 'center' },
      { type: 'label', at: lab, text: 'normal features', color: '--blue', dy: 22, align: 'center' },
    ];
  }
  const redraw = () => { s3.items = items(); s3.redraw(); };
  PE.onTheme(redraw);

  function say() {
    const x0 = F.TEST[S.test], k = Math.round(S.stage), p = F.push(x0, k), here = F.logDensity(p.h, k), tot = here + p.logdet;
    const lower = ref.filter(v => v < tot).length / ref.length * 100, who = S.test === 'anomaly' ? 'Scratch feature' : 'Good-part feature';
    const f = v => (v >= 0 ? '+' : '') + v.toFixed(2);
    out.textContent = k === F.K
      ? `All ${F.K} layers applied: normal is now a standard Gaussian. ${who}: log N(z) ${here.toFixed(2)} ${f(p.logdet)} log-det = log likelihood ${tot.toFixed(2)}. ${lower.toFixed(lower < 1 ? 2 : 0)}% of normal training features score lower.`
      : `${k} of ${F.K} layers applied. ${who}: log density here ${here.toFixed(2)} ${f(p.logdet)} log-det so far = ${tot.toFixed(2)}. The total is the same at every stage; only the split changes.`;
  }
  function pressed() { for (let k = 0; k <= F.K; k++) document.getElementById('btn-fastflow-s' + k).setAttribute('aria-pressed', Math.round(S.stage) === k && S.target === k);
    document.getElementById('btn-fastflow-anomaly').setAttribute('aria-pressed', S.test === 'anomaly'); document.getElementById('btn-fastflow-normal').setAttribute('aria-pressed', S.test === 'normal'); }
  function goTo(k, from) {
    stopT(); const a = from ?? S.stage; S.target = k; pressed();
    if (a === k) { S.stage = k; redraw(); say(); pressed(); return; }
    stopT = s3.tween(Math.min(2600, 650 * Math.abs(k - a)), u => { S.stage = a + (k - a) * u; redraw(); }, () => { S.stage = k; redraw(); say(); pressed(); });
  }
  for (let k = 0; k <= F.K; k++) document.getElementById('btn-fastflow-s' + k).addEventListener('click', () => goTo(k));
  document.getElementById('btn-fastflow-play').addEventListener('click', () => { S.stage = 0; goTo(F.K, 0); });
  document.getElementById('btn-fastflow-anomaly').addEventListener('click', () => { S.test = 'anomaly'; pressed(); redraw(); say(); });
  document.getElementById('btn-fastflow-normal').addEventListener('click', () => { S.test = 'normal'; pressed(); redraw(); say(); });
  document.getElementById('btn-fastflow-reset').addEventListener('click', () => { stopT(); Object.assign(S, DEF); S.target = 0; s3.setView({ yaw: .55, pitch: .62, dist: 10.5 }, 500); redraw(); say(); pressed(); });
  S.target = 0; redraw(); say(); pressed();
})();
</script>
