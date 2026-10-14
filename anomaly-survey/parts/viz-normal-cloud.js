<script>
(() => {
  // ch-problem: nearest-neighbour anomaly scoring on toy cap features (axes: diagonal edges, brightness).
  // Bank sizes, the default threshold and the test seeds were checked in build/toy-ch1.mjs: at 0.30, one cap in
  // memory flags every good test cap (0.33-0.40), five or twenty pass them (<= 0.26), the scratched cap scores 0.81-0.94.
  const fig = document.getElementById('viz-normal-cloud'), el = document.getElementById('cv-normal-cloud');
  const out = fig.querySelector('.readout'), slider = document.getElementById('rng-cloud-tau'), sOut = document.getElementById('out-cloud-tau');
  const f2 = f => { const o = []; for (let k = 0; k < f.length / 3; k++) o.push([f[k * 3 + 1], f[k * 3 + 2]]); return o; };
  const pool = Array.from({ length: 20 }, (_, i) => f2(TOY.features(TOY.make('cap', 1 + i))));
  const tests = { good: TOY.make('cap', 104), scratch: TOY.defect(TOY.make('cap', 200), 'scratch', 200) };
  const tf = { good: f2(TOY.features(tests.good)), scratch: f2(TOY.features(tests.scratch)) };
  const all = pool.flat().concat(tf.scratch);
  const lo = [Math.min(...all.map(p => p[0])) - .2, Math.min(...all.map(p => p[1])) - .2], hi = [Math.max(...all.map(p => p[0])) + .2, Math.max(...all.map(p => p[1])) + .2];
  const DEF = { n: 20, test: 'good', tau: .3 };
  const S = { ...DEF, probe: null, shown: 20, alpha: 1 };
  const cache = {};
  function scored(n, test) {
    const key = n + test; if (cache[key]) return cache[key];
    const bank = pool.slice(0, n).flat();
    const s = tf[test].map(p => { let b = Infinity, bi = 0; bank.forEach((m, i) => { const d = Math.hypot(m[0] - p[0], m[1] - p[1]); if (d < b) { b = d; bi = i; } }); return [b, bi]; });
    let w = 0; s.forEach((v, q) => { if (v[0] > s[w][0]) w = q; });
    return cache[key] = { bank, s, worst: w, map: TOY.toPixels(Float32Array.from(s.map(v => v[0]))) };
  }
  const nearest = (bank, p) => { let b = Infinity, bi = 0; bank.forEach((m, i) => { const d = Math.hypot(m[0] - p[0], m[1] - p[1]); if (d < b) { b = d; bi = i; } }); return [b, bi]; };
  let L = null, heat = null, heatKey = '', cover = null, coverKey = '';
  const glow = PE.linkPulses(fig, () => view.redraw());
  const view = PE.canvas(el, w => w < 560 ? 1.3 : .5, (ctx, W, H) => draw(ctx, W, H));

  function layout(W, H) {
    const narrow = W < 560;
    const img = narrow ? { x: W * .26, y: 34, s: Math.min(W * .48, H * .33) } : { x: 26, y: 46, s: Math.min(W * .3, H * .62) };
    const fp = narrow ? { x: 46, y: img.y + img.s + 64, w: W - 66, h: H - (img.y + img.s + 64) - 44 } : { x: W * .45, y: 28, w: W * .52, h: H - 76 };
    return { narrow, img, fp };
  }
  const X = p => L.fp.x + (p[0] - lo[0]) / (hi[0] - lo[0]) * L.fp.w, Y = p => L.fp.y + L.fp.h - (p[1] - lo[1]) / (hi[1] - lo[1]) * L.fp.h;
  const unX = x => lo[0] + (x - L.fp.x) / L.fp.w * (hi[0] - lo[0]), unY = y => lo[1] + (L.fp.y + L.fp.h - y) / L.fp.h * (hi[1] - lo[1]);

  function coverage(W, H, bank, tau, color) {
    const key = [W, H, bank.length, tau, color].join();
    if (key === coverKey) return cover;
    const c = document.createElement('canvas'), r = devicePixelRatio || 1; c.width = W * r; c.height = H * r;
    const g = c.getContext('2d'); g.setTransform(r, 0, 0, r, 0, 0); g.fillStyle = color;
    const rx = tau / (hi[0] - lo[0]) * L.fp.w, ry = tau / (hi[1] - lo[1]) * L.fp.h;
    g.beginPath(); bank.forEach(m => { g.moveTo(X(m) + rx, Y(m)); g.ellipse(X(m), Y(m), rx, ry, 0, 0, 7); }); g.fill();
    coverKey = key; return cover = c;
  }

  function draw(ctx, W, H) {
    if (!W) return;
    L = layout(W, H);
    if (L.fp.w <= 0 || L.fp.h <= 0 || L.img.s <= 0) return;   // canvas not laid out yet
    const C = { blue: PE.css('--blue'), red: PE.css('--red'), yellow: PE.css('--yellow'), muted: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg') };
    const R = scored(S.n, S.test), cur = scored(S.shown, S.test);
    ctx.clearRect(0, 0, W, H);
    // camera view of the cap under inspection, heat where patches exceed the threshold
    const hk = S.n + S.test + S.tau + C.red;
    if (hk !== heatKey) { heat = TOY.heatBitmap(R.map, S.tau, S.tau + .45, C.red); heatKey = hk; }
    TOY.draw(ctx, tests[S.test], L.img.x, L.img.y, L.img.s, { heat });
    const wsz = L.img.s / TOY.N, w = R.worst, wx = L.img.x + (w % TOY.G) * TOY.S * wsz, wy = L.img.y + Math.floor(w / TOY.G) * TOY.S * wsz;
    ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2 + 2 * glow('max'); ctx.strokeRect(wx, wy, TOY.P * wsz, TOY.P * wsz); ctx.restore();
    const score = R.s[w][0], reject = score > S.tau;
    ctx.save(); ctx.font = '600 13px system-ui, sans-serif'; const word = reject ? 'reject' : 'pass', tw = ctx.measureText(word).width + 18;
    const tx = L.img.x, ty = L.img.y + L.img.s + 14;
    ctx.strokeStyle = reject ? C.red : C.blue; ctx.lineWidth = 1.3 + 1.5 * glow('max'); ctx.beginPath(); ctx.roundRect(tx, ty, tw, 22, 11); ctx.stroke();
    ctx.fillStyle = reject ? C.red : C.blue; ctx.fillText(word, tx + 9, ty + 15); ctx.restore();
    PE.label(ctx, `image score ${score.toFixed(2)}`, tx + tw + 10, ty + 15, C.muted, 12);
    PE.label(ctx, 'camera view', L.img.x, L.img.y - 12, C.muted, 12);
    // feature space
    ctx.save(); ctx.strokeStyle = C.rule; ctx.lineWidth = 1; ctx.strokeRect(L.fp.x, L.fp.y, L.fp.w, L.fp.h); ctx.restore();
    PE.label(ctx, 'diagonal edges', L.fp.x + L.fp.w / 2, L.fp.y + L.fp.h + 18, C.muted, 11, 'center');
    ctx.save(); ctx.translate(L.fp.x - 10, L.fp.y + L.fp.h / 2); ctx.rotate(-Math.PI / 2); PE.label(ctx, 'brightness', 0, 0, C.muted, 11, 'center'); ctx.restore();
    ctx.save(); ctx.beginPath(); ctx.rect(L.fp.x, L.fp.y, L.fp.w, L.fp.h); ctx.clip();
    ctx.globalAlpha = .16; ctx.drawImage(coverage(W, H, cur.bank, S.tau, C.blue), 0, 0, W, H); ctx.globalAlpha = 1;
    const gb = glow('bank');
    cur.bank.forEach(m => PE.dot(ctx, X(m), Y(m), 1.5 + gb, C.blue, .6 * S.alpha + .3 * gb));
    if (S.shown !== S.n) R.bank.forEach(m => PE.dot(ctx, X(m), Y(m), 1.5, C.blue, .6 * (1 - S.alpha)));
    const gt = glow('test');
    tf[S.test].forEach((p, q) => { const far = R.s[q][0] > S.tau;
      PE.dot(ctx, X(p), Y(p), 2 + gt, C.yellow, .9);
      if (far) { ctx.save(); ctx.strokeStyle = C.red; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(X(p), Y(p), 4.5, 0, 7); ctx.stroke(); ctx.restore(); } });
    const wp = tf[S.test][w], nb = R.bank[R.s[w][1]], gd = glow('dist');
    ctx.save(); ctx.strokeStyle = C.red; ctx.lineWidth = 2 + 2.5 * gd; ctx.beginPath(); ctx.moveTo(X(nb), Y(nb)); ctx.lineTo(X(wp), Y(wp)); ctx.stroke(); ctx.restore();
    PE.dot(ctx, X(wp), Y(wp), 4.5 + 2 * glow('max'), C.yellow); PE.dot(ctx, X(nb), Y(nb), 3, C.blue);
    if (S.probe) { const [d, i] = nearest(R.bank, S.probe), m = R.bank[i];
      ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = d > S.tau ? C.red : C.blue; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(m), Y(m)); ctx.lineTo(X(S.probe), Y(S.probe)); ctx.stroke(); ctx.restore();
      ctx.save(); ctx.strokeStyle = C.yellow; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(S.probe), Y(S.probe), 7, 0, 7); ctx.stroke(); ctx.restore(); }
    ctx.restore();
    PE.label(ctx, 'feature space', L.fp.x + 6, L.fp.y + 16, C.muted, 12);
  }

  function say() {
    const R = scored(S.n, S.test), sc = R.s[R.worst][0], rej = sc > S.tau, nfar = R.s.filter(v => v[0] > S.tau).length;
    let t = `${S.n} good ${S.n === 1 ? 'cap' : 'caps'} in memory (${(S.n * TOY.G * TOY.G).toLocaleString('en-US')} patch features). The ${S.test === 'good' ? 'good' : 'scratched'} cap's worst patch is ${sc.toFixed(2)} from normal, ${rej ? 'over' : 'under'} the ${S.tau.toFixed(2)} threshold: ${rej ? 'reject' : 'pass'}`;
    t += rej ? ` (${nfar} of ${TOY.G * TOY.G} patches flagged${S.test === 'good' ? '; a false alarm' : ''}).` : '.';
    if (S.probe) { const d = nearest(R.bank, S.probe)[0]; t += ` Your probe patch is ${d.toFixed(2)} from the nearest normal feature: ${d > S.tau ? 'it would be flagged' : 'it looks normal'}.`; }
    out.textContent = t;
  }
  function setBank(n) {
    if (n === S.n) return; const from = S.n; S.n = n; S.shown = from; S.alpha = 1;
    PE.animate(500, u => { S.alpha = 1 - u; view.redraw(); }, () => { S.shown = n; S.alpha = 1; view.redraw(); });
    [1, 5, 20].forEach(k => document.getElementById('btn-cloud-bank-' + k).setAttribute('aria-pressed', k === n)); say();
  }
  function setTest(t) { S.test = t; document.getElementById('btn-cloud-good').setAttribute('aria-pressed', t === 'good'); document.getElementById('btn-cloud-scratch').setAttribute('aria-pressed', t === 'scratch'); view.redraw(); say(); }
  function setTau(v, animate) {
    const a = S.tau; slider.value = v; sOut.textContent = (+v).toFixed(2);
    if (!animate) { S.tau = +v; view.redraw(); say(); return; }
    PE.animate(500, u => { S.tau = a + (v - a) * u; view.redraw(); }, () => { S.tau = +v; view.redraw(); say(); });
  }
  [1, 5, 20].forEach(k => document.getElementById('btn-cloud-bank-' + k).addEventListener('click', () => setBank(k)));
  document.getElementById('btn-cloud-good').addEventListener('click', () => setTest('good'));
  document.getElementById('btn-cloud-scratch').addEventListener('click', () => setTest('scratch'));
  slider.addEventListener('input', () => setTau(+slider.value, false));
  document.getElementById('btn-cloud-reset').addEventListener('click', () => { S.probe = null; setBank(DEF.n); setTest(DEF.test); setTau(DEF.tau, true); });
  // Probe: drag in the feature space, or arrow keys when the canvas has focus.
  let drag = false;
  const at = e => { const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    return (L && x >= L.fp.x && x <= L.fp.x + L.fp.w && y >= L.fp.y && y <= L.fp.y + L.fp.h) ? [unX(x), unY(y)] : null; };
  el.addEventListener('pointerdown', e => { const p = at(e); if (!p) return; drag = true; el.setPointerCapture(e.pointerId); S.probe = p; view.redraw(); say(); });
  el.addEventListener('pointermove', e => { if (!drag) return; const p = at(e); if (p) { S.probe = p; view.redraw(); } });
  el.addEventListener('pointerup', () => { if (drag) { drag = false; say(); } });
  el.addEventListener('keydown', e => { const k = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key]; if (!k) return;
    e.preventDefault(); const p = S.probe || [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], st = .06;
    S.probe = [PE.clamp(p[0] + k[0] * st * (hi[0] - lo[0]) / 4, lo[0], hi[0]), PE.clamp(p[1] + k[1] * st * (hi[1] - lo[1]) / 4, lo[1], hi[1])]; view.redraw(); say(); });
  say();
})();
</script>
