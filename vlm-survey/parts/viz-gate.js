<script>
(() => {
  const fig = document.getElementById('viz-gate'), el = document.getElementById('gate-canvas');
  const slider = document.getElementById('rng-gate-progress'), pOut = document.getElementById('gate-progress-val');
  const out = document.getElementById('gate-readout');

  // Toy gated xattn-dense block in 2D (same code as build/sim/gate-model.cjs).
  const words = ['dog', 'cat', 'car', 'beach'];
  const U = [[0.95, 0.31], [0.2, 0.98], [-0.9, 0.44], [-0.1, -1]];
  const y0 = [0.25, 0.55];
  const X = [[1.0, 0.1], [0.8, -0.35], [0.9, 0.4]];
  const Wv0 = [[-1.2, 0.3], [-0.4, -1.0]], Wv1 = [[1.1, 0.2], [-0.5, 0.3]];
  const F0 = [[-0.8, 0.5], [0.6, -0.9]], F1 = [[0.7, -0.2], [0.2, -0.6]];
  const lerpM = (A, B, u) => A.map((r, i) => r.map((v, j) => v + (B[i][j] - v) * u));
  const mv = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];
  const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1]];
  function block(p, gx, gd) {
    const w = 1 - Math.exp(-5 * p), Wv = lerpM(Wv0, Wv1, w), Fw = lerpM(F0, F1, w);
    const att = PE.softmax(X.map(x => 2 * (x[0] * y0[0] + x[1] * y0[1])));
    const A = X.map(x => mv(Wv, x)).reduce((s, v, i) => add(s, v, att[i]), [0, 0]);
    const y1 = add(y0, A, gx);
    const h = mv(Fw, y1).map(z => Math.max(0, z) ** 2);
    const Fv = [h[0] - 0.3 * h[1], 0.4 * h[0] + h[1]];
    const y2 = add(y1, Fv, gd);
    const g = y => [0.25 * Math.tanh(y[1]), 0.25 * Math.tanh(-y[0] + y[1])];
    const y3 = add(y2, g(y2));
    const pr = PE.softmax(U.map(u => 3 * (u[0] * y3[0] + u[1] * y3[1])));
    return { att, y1, y2, y3, pr };
  }
  const ax = p => 1.4 * (1 - Math.exp(-6 * p)), ad = p => 0.9 * (1 - Math.exp(-4 * p));
  const ref = block(0, 0, 0);
  const kl = (a, b) => a.reduce((s, x, i) => s + x * Math.log(x / b[i]), 0);

  // p = toy training progress; ng = 1 replaces both tanh gates by 1 (the "no gate" ablation).
  const S = { p: 0, ng: 0 };
  const gates = () => [PE.lerp(Math.tanh(ax(S.p)), 1, S.ng), PE.lerp(Math.tanh(ad(S.p)), 1, S.ng)];
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};

  const layout = W => {
    if (W < 560) {
      const tw = W - 60;
      return { stacked: true, tanh: { x: 46, y: 30, w: tw, h: 90 }, diag: { x: 14, y: 176, w: W - 28, h: 200 }, bars: { x: 64, y: 450, w: W - 130, h: 140 } };
    }
    return { stacked: false, tanh: { x: 44, y: 44, w: W * .2, h: W * .19 }, diag: { x: W * .3, y: 34, w: W * .47, h: W * .38 }, bars: { x: W * .84, y: 74, w: W * .1, h: W * .28 } };
  };

  const cv = PE.canvas(el, w => w < 560 ? 640 / w : .5, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Br = PE.css('--brown'),
      F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const L = layout(W), [gx, gd] = gates(), r = block(S.p, gx, gd);
    const gG = glow('gate'), gX = glow('xattn'), gD = glow('dense'), gF = glow('frozen'), gT = glow('text');

    // Panel 1: the gate. tanh(alpha) against alpha, both alphas marked.
    const t = L.tanh, A = PE.axes(ctx, t, [0, 2.5], [0, 1], { xt: [0, 1, 2], yt: [0, .5, 1], xl: 'alpha', ylo: 30 });
    PE.label(ctx, 'gate = tanh(alpha)', t.x, t.y - 12, G, 12);
    const pts = []; for (let k = 0; k <= 60; k++) { const a = 2.5 * k / 60; pts.push([A.X(a), A.Y(Math.tanh(a))]); }
    PE.polyline(ctx, pts, G, 2 + 2 * gG);
    if (S.ng > 0) { ctx.save(); ctx.globalAlpha = S.ng; PE.polyline(ctx, [[A.X(0), A.Y(1)], [A.X(2.5), A.Y(1)]], F, 1.5, [5, 4]);
      PE.label(ctx, 'no gate: always 1', A.X(0) + 6, A.Y(1) + 14, F, 10.5); ctx.restore(); }
    const ax0 = ax(S.p), ad0 = ad(S.p), R = 5 + 3 * gG;
    PE.dot(ctx, A.X(ax0), A.Y(Math.tanh(ax0)), R, G);
    ctx.strokeStyle = G; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(A.X(ad0), A.Y(Math.tanh(ad0)), R, 0, 7); ctx.stroke();
    PE.label(ctx, '\u25cf xattn ' + Math.tanh(ax0).toFixed(2), t.x + t.w - 4, t.y + t.h - 26, G, 11, 'right');
    PE.label(ctx, '\u25cb dense ' + Math.tanh(ad0).toFixed(2), t.x + t.w - 4, t.y + t.h - 11, G, 11, 'right');

    // Panel 2: one text token's vector on the residual stream, built head to tail.
    const d = L.diag, xr = [-.85, 1.8], yr = [-.45, .8], s = Math.min(d.w / (xr[1] - xr[0]), d.h / (yr[1] - yr[0]));
    const ox = d.x + (d.w - s * (xr[1] - xr[0])) / 2 - s * xr[0], oy = d.y + d.h - (d.h - s * (yr[1] - yr[0])) / 2 + s * yr[0];
    const P = v => [ox + s * v[0], oy - s * v[1]];
    PE.label(ctx, 'one text token on the residual stream', d.x + d.w / 2, d.y - (L.stacked ? 12 : 10), F, 12, 'center');
    ctx.strokeStyle = Rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(...P([xr[0], 0])); ctx.lineTo(...P([xr[1], 0])); ctx.moveTo(...P([0, yr[0]])); ctx.lineTo(...P([0, yr[1]])); ctx.stroke();
    // Visual tokens (keys/values of the cross-attention), with attention weights from the text token.
    X.forEach((x, i) => { const [px, py] = P(x), [qx, qy] = P(y0);
      PE.polyline(ctx, [[qx, qy], [px, py]], B, 1 + 5 * r.att[i] * (1 + gX), [3, 3], .35 + .4 * gX);
      PE.dot(ctx, px, py, 5 + 2 * gX, B); });
    PE.label(ctx, 'visual tokens x', P(X[1])[0], P(X[1])[1] + 17, B, 11, 'center');
    // Frozen-only path (what the original LM computes), as a ghost.
    const [rx, ry] = P(ref.y3), [ax1, ay1] = P(y0), [o0, o1] = P([0, 0]);
    PE.arrow(ctx, ax1, ay1, rx, ry, Br, 2 + 2 * gF, 8, .45 + .4 * gF);
    PE.arrow(ctx, o0, o1, ax1, ay1, Y, 2.5 + 2 * gT, 9);
    PE.label(ctx, 'y (text)', ax1 - 6, ay1 - 6, Y, 11.5, 'right');
    const [p1x, p1y] = P(r.y1), [p2x, p2y] = P(r.y2), [p3x, p3y] = P(r.y3);
    if (Math.hypot(p1x - ax1, p1y - ay1) > 2) PE.arrow(ctx, ax1, ay1, p1x, p1y, G, 2.5 + 2 * gX, 9);
    if (Math.hypot(p2x - p1x, p2y - p1y) > 2) { ctx.save(); ctx.setLineDash([6, 3]); PE.arrow(ctx, p1x, p1y, p2x, p2y, G, 2.5 + 2 * gD, 9); ctx.restore(); }
    PE.arrow(ctx, p2x, p2y, p3x, p3y, Br, 2.5 + 2 * gF, 9);
    const moved = Math.hypot(r.y3[0] - ref.y3[0], r.y3[1] - ref.y3[1]);
    if (moved > .02) { PE.polyline(ctx, [[rx, ry], [p3x, p3y]], M, 1.5, [2, 3]); }
    PE.dot(ctx, rx, ry, 4, Br);
    PE.label(ctx, 'frozen LM alone', rx + 7, ry - 7, Br, 11);
    PE.dot(ctx, p3x, p3y, 5, F);
    { const txt = moved > .02 ? 'this model' : 'this model = frozen LM', right = p3x > d.x + d.w - 90; PE.label(ctx, txt, p3x + (right ? 0 : 8), p3y + 18, F, 11, right ? 'right' : 'left'); }
    const ky = d.y + d.h + (L.stacked ? 16 : 12);
    const key = [[Y, 'text y'], [G, '+ gated xattn'], [G, '+ gated FFW (dashed)'], [Br, '+ frozen LM layer']];
    ctx.font = '600 10.5px system-ui, sans-serif'; const rows = L.stacked ? [key.slice(0, 2), key.slice(2)] : [key];
    rows.forEach((row, ri) => { let kx = d.x + (L.stacked ? 0 : 8); row.forEach(([c, tx]) => { PE.label(ctx, tx, kx, ky + 15 * ri, c, 10.5); kx += ctx.measureText(tx).width + 12; }); });

    // Panel 3: next-word probabilities, frozen LM (brown outline) vs this model (yellow).
    const b = L.bars, rowH = b.h / words.length;
    PE.label(ctx, L.stacked ? 'next word after "a photo of a"' : 'next word after', L.stacked ? 14 : b.x - 48, b.y - (L.stacked ? 22 : 36), F, 12);
    if (!L.stacked) PE.label(ctx, '"a photo of a"', b.x - 48, b.y - 20, F, 12);
    words.forEach((wd, i) => { const yy = b.y + i * rowH, hh = rowH * .5;
      PE.label(ctx, wd, b.x - 8, yy + hh * .8, Y, 11.5, 'right');
      ctx.fillStyle = Rule; ctx.fillRect(b.x, yy, b.w, hh);
      ctx.fillStyle = Y; ctx.fillRect(b.x, yy, b.w * r.pr[i], hh);
      ctx.strokeStyle = Br; ctx.lineWidth = 2 + 2 * gF; ctx.strokeRect(b.x, yy - 2, b.w * ref.pr[i], hh + 4);
      PE.label(ctx, (100 * r.pr[i]).toFixed(0) + '%', b.x + b.w + 6, yy + hh * .8, M, 11); });
    PE.label(ctx, 'KL from frozen LM ' + kl(r.pr, ref.pr).toFixed(3), L.stacked ? 14 : b.x - 48, b.y + b.h + 4, F, 11.5);
    PE.label(ctx, 'brown box = frozen LM', L.stacked ? 14 : b.x - 48, b.y + b.h + 20, Br, 10.5);
  });

  function readout() {
    const [gx, gd] = gates(), r = block(S.p, gx, gd), k = kl(r.pr, ref.pr), top = r.pr.indexOf(Math.max(...r.pr));
    const pct = i => (100 * r.pr[i]).toFixed(0) + '%';
    if (S.ng > .5) { out.textContent = `No gates at step 0: the new layers are untrained random weights added at full strength. The next-word guess jumps to "${words[top]}" (${pct(top)}); KL from the frozen LM is ${k.toFixed(3)}. The language model was broken before it learned anything.`; return; }
    if (S.p === 0) { out.textContent = `Step 0: alpha_xattn = alpha_dense = 0, so both gates are tanh(0) = 0. The output is exactly the frozen language model's (KL ${k.toFixed(3)}): "${words[top]}" at ${pct(top)}.`; return; }
    out.textContent = `Training ${(100 * S.p).toFixed(0)}% (toy): gates ${gx.toFixed(2)} (xattn) and ${gd.toFixed(2)} (dense). Next word "${words[top]}" at ${pct(top)}${top ? `, "dog" at ${pct(0)}` : ''}; KL from the frozen LM ${k.toFixed(3)}.`;
  }
  function tweenTo(p, ng, ms = 800, done) {
    stop(); const p0 = S.p, n0 = S.ng;
    stop = PE.animate(ms, u => { S.p = PE.lerp(p0, p, u); S.ng = PE.lerp(n0, ng, u); if (u === 1) { S.p = p; S.ng = ng; } sync(); cv.redraw(); readout(); }, done);
  }
  const sync = () => { slider.value = Math.round(100 * S.p); pOut.textContent = Math.round(100 * S.p) + '%'; };
  document.getElementById('btn-gate-step0').addEventListener('click', () => tweenTo(0, 0));
  document.getElementById('btn-gate-nogate').addEventListener('click', () => tweenTo(0, 1));
  document.getElementById('btn-gate-trained').addEventListener('click', () => tweenTo(1, 0));
  document.getElementById('btn-gate-train').addEventListener('click', () => { stop(); S.p = 0; S.ng = 0; tweenTo(1, 0, 2600); });
  document.getElementById('btn-gate-reset').addEventListener('click', () => { stop(); S.p = 0; S.ng = 0; sync(); cv.redraw(); readout(); });
  slider.addEventListener('input', () => { stop(); S.p = +slider.value / 100; S.ng = 0; pOut.textContent = slider.value + '%'; cv.redraw(); readout(); });
  sync(); readout();
})();
</script>
