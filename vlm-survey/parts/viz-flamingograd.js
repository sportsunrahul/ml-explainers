<script>
(() => {
  const fig = document.getElementById('viz-flamingograd'), el = document.getElementById('flamingograd-canvas');
  const out = document.getElementById('flamingograd-readout'), rng = document.getElementById('rng-flamingograd-alpha'), rv = document.getElementById('flamingograd-alpha-val');
  // Same toy block as build/sim/arch-flamingo-gategrad.cjs: 2-d vectors, three visual tokens, four words, loss = -log p("dog").
  const U = [[0.95, 0.31], [0.2, 0.98], [-0.9, 0.44], [-0.1, -1]], Y0 = [0.25, 0.55];
  const X0 = [[1.0, 0.1], [0.8, -0.35], [0.9, 0.4]], WV = [[-1.2, 0.3], [-0.4, -1.0]], F0 = [[0.9, 0.8], [-0.6, 0.9]];
  const mv = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
  const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1]];
  function loss(P) {
    const att = PE.softmax(P.X.map(v => 2 * (v[0] * P.y[0] + v[1] * P.y[1])));
    const A = P.X.map(v => mv(P.Wv, v)).reduce((s, v, i) => add(s, v, att[i]), [0, 0]);
    const y1 = add(P.y, A, Math.tanh(P.ax)), h = mv(P.F, y1).map(z => Math.max(0, z) ** 2);
    const y2 = add(y1, [h[0] - 0.3 * h[1], 0.4 * h[0] + h[1]], Math.tanh(P.ad));
    const y3 = add(y2, [0.25 * Math.tanh(y2[1]), 0.25 * Math.tanh(-y2[0] + y2[1])]);
    return -Math.log(PE.softmax(U.map(u => 3 * (u[0] * y3[0] + u[1] * y3[1])))[0]);
  }
  const cl = P => JSON.parse(JSON.stringify(P));
  // Central finite differences per parameter group; returns the gradient norm of each group (signed for the scalar gates).
  function grads(a, wz) {
    const P = { ax: a, ad: a, Wv: WV.map(r => r.map(v => v * (1 - wz))), F: F0, X: X0, y: Y0 }, e = 1e-6;
    const fd = path => { const p = cl(P), q = cl(P), o = k => path.slice(0, -1).reduce((s, j) => s[j], k);
      o(p)[path.at(-1)] += e; o(q)[path.at(-1)] -= e; return (loss(p) - loss(q)) / (2 * e); };
    const nrm = ps => Math.hypot(...ps.map(fd)), ij = (k, n) => [...Array(n * 2)].map((_, t) => [k, t >> 1, t & 1]);
    return { ax: fd(['ax']), ad: fd(['ad']), Wv: nrm(ij('Wv', 2)), F: nrm(ij('F', 2)), X: nrm(ij('X', 3)), y: nrm([['y', 0], ['y', 1]]), L: loss(P) };
  }
  const FROZEN_Y = grads(0, 0).y;   // at alpha = 0 the block is the identity, so this equals the frozen-LM-only gradient
  const S = { a: 0, wz: 0 };
  let stop = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());
  const ROWS = [
    { k: 'ax', name: 'alpha_xattn', note: 'xattn gate', col: 'G', link: 'alpha' },
    { k: 'ad', name: 'alpha_dense', note: 'FFW gate', col: 'G', link: 'alpha' },
    { k: 'Wv', name: 'xattn weights', note: 'W (values)', col: 'G', link: 'weights' },
    { k: 'F', name: 'FFW weights', note: 'new dense layer', col: 'G', link: 'weights' },
    { k: 'X', name: 'visual tokens x', note: 'on to the resampler', col: 'B', link: 'visual' },
    { k: 'y', name: 'text vector y', note: 'down to the LM below', col: 'Y', link: 'down' },
  ];
  const cv = PE.canvas(el, w => w < 560 ? .95 : .42, (ctx, W, H) => {
    const C = { G: PE.css('--green'), B: PE.css('--blue'), Y: PE.css('--yellow'), R: PE.css('--red'), Br: PE.css('--brown'), F: PE.css('--fg'), M: PE.css('--muted'), Ru: PE.css('--rule'), BG: PE.css('--bg') };
    ctx.fillStyle = C.BG; ctx.fillRect(0, 0, W, H);
    const g = grads(S.a, S.wz), small = W < 560, fs = small ? 10.5 : 12;
    const top = small ? 46 : 40, lx = small ? 8 : 16, bx = small ? W * .42 : W * .3, bw = W - bx - (small ? 44 : 70), rh = (H - top - 26) / ROWS.length, MAX = 6;
    const t = Math.tanh(S.a);
    PE.label(ctx, `alpha = ${S.a.toFixed(2)}   tanh(alpha) = ${t.toFixed(2)}`, lx, 18, C.G, fs + 1);
    PE.label(ctx, S.wz > .5 ? 'xattn weights W set to 0' : 'xattn weights W random (as at init)', lx, 34, S.wz > .5 ? C.Br : C.M, fs - 1);
    ROWS.forEach((r, i) => {
      const y = top + i * rh, v = g[r.k], gl = glow(r.link) + (r.link === 'alpha' || r.link === 'weights' || r.link === 'visual' ? glow('gate') : 0);
      if (gl > 0) { ctx.save(); ctx.globalAlpha = .18 * Math.min(1, gl); ctx.fillStyle = C[r.col]; ctx.fillRect(0, y + 2, W, rh - 4); ctx.restore(); }
      PE.label(ctx, r.name, lx, y + rh / 2 - 1, C[r.col], fs);
      PE.label(ctx, r.note, lx, y + rh / 2 + fs, C.M, fs - 1.5);
      ctx.fillStyle = PE.mix(C.BG.length === 7 ? C.BG : '#16181d', C.Ru.length === 7 ? C.Ru : '#888888', .6); ctx.fillRect(bx, y + rh / 2 - 7, bw, 12);
      const len = bw * Math.min(1, Math.abs(v) / MAX);
      ctx.fillStyle = C.R; ctx.fillRect(bx, y + rh / 2 - 7, len, 12);
      if (r.k === 'y') { const fx = bx + bw * FROZEN_Y / MAX; ctx.fillStyle = C.Br; ctx.fillRect(fx - 1.5, y + rh / 2 - 12, 3, 22);
        PE.label(ctx, 'frozen LM alone', fx, y + rh / 2 + 22, C.Br, fs - 2, small ? 'center' : 'center'); }
      PE.label(ctx, Math.abs(v) < 5e-4 ? '0' : (r.k === 'ax' || r.k === 'ad' ? v.toFixed(2) : Math.abs(v).toFixed(2)), bx + len + 6, y + rh / 2 + 4, Math.abs(v) < 5e-4 ? C.F : C.R, fs);
    });
    PE.label(ctx, 'gradient size (red); same scale for all rows', bx, H - 6, C.M, fs - 1.5);
  });
  function show() {
    const g = grads(S.a, S.wz), z = v => Math.abs(v) < 5e-4 ? '0' : v.toFixed(2);
    rng.value = S.a; rv.textContent = S.a.toFixed(2);
    out.textContent = S.wz > .5
      ? `alpha = 0 and W = 0: the cross-attention branch outputs nothing, so the gradient into alpha_xattn is ${z(g.ax)} as well, and the xattn weights get ${z(g.Wv)}. Neither can ever start. (The FFW gate still gets ${z(g.ad)}, because its own layer is random.)`
      : S.a < 1e-3
        ? `Step 0 (alpha = 0): xattn weights ${z(g.Wv)}, FFW weights ${z(g.F)}, visual tokens ${z(g.X)}. The gates are not stuck: alpha_xattn gets ${z(g.ax)}, alpha_dense ${z(g.ad)}. The gradient passed down, ${z(g.y)}, equals the frozen LM's own.`
        : `alpha = ${S.a.toFixed(2)}: the weights, visual tokens and down-pass now get a gradient; the weights' share is scaled by tanh(alpha) = ${Math.tanh(S.a).toFixed(2)}.${g.ax > 0 ? ' The alpha_xattn gradient is positive: opening this gate raised the toy loss, so training would push alpha_xattn negative.' : ''} The gradient passed down (${z(g.y)}) now differs from the frozen LM's (${z(FROZEN_Y)}).`;
  }
  function go(a, wz) { stop(); const a0 = S.a, w0 = S.wz; stop = PE.animate(700, u => { S.a = PE.lerp(a0, a, u); S.wz = PE.lerp(w0, wz, u); cv.redraw(); }, () => { S.a = a; S.wz = wz; show(); cv.redraw(); }); show(); }
  document.getElementById('btn-flamingograd-step0').addEventListener('click', () => go(0, 0));
  document.getElementById('btn-flamingograd-open').addEventListener('click', () => go(.5, 0));
  document.getElementById('btn-flamingograd-zerow').addEventListener('click', () => go(0, 1));
  document.getElementById('btn-flamingograd-reset').addEventListener('click', () => { stop(); S.a = 0; S.wz = 0; show(); cv.redraw(); });
  rng.addEventListener('input', () => { stop(); S.a = +rng.value; show(); cv.redraw(); });
  show();
})();
</script>
