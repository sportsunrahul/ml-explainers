/* viz-ntxent (ch-simclr): NT-Xent (SimCLR Eq. 1) on 2N unit vectors in the plane, in the paper's order: views 2k-1 and
   2k come from image k. Left: the views on the circle (row 1's anchor blue, its partner yellow, its negatives red).
   Right: the 2N x 2N cosine matrix with the diagonal masked, one positive per row, each row's softmax as cell shading,
   and each row's loss. N = 2 uses cosines 0.8, 0.6, 0 and -0.6 so the worked example can be checked by hand. */
KIT.NTX_DEG = { 2: [0, 36.87, 90, 126.87], 4: [0, 36.87, 90, 126.87, 195, 225, 268, 306] };
// Pure math (node-safe): th = angles in radians, paper order. Returns cosines S, row softmax P (diagonal 0), row losses, mean.
KIT.ntxentRows = (th, tau) => {
  const M = th.length, z = th.map(t => [Math.cos(t), Math.sin(t)]), S = [], P = [], loss = [];
  for (let i = 0; i < M; i++) {
    S.push(z.map(v => z[i][0] * v[0] + z[i][1] * v[1]));
    const e = S[i].map((s, k) => k === i ? 0 : Math.exp(s / tau)), Z = e.reduce((a, b) => a + b, 0);
    P.push(e.map(x => x / Z)); loss.push(-Math.log(P[i][i ^ 1]));
  }
  return { S, P, loss, mean: loss.reduce((a, b) => a + b, 0) / M };
};
KIT.ntxent = (canvas, opts = {}) => {
  const state = { N: 2, tau: .5, reveal: 1 };
  const glow = opts.fig ? PE.linkPulses(opts.fig, () => cv.redraw()) : () => 0;
  let cancel = () => {};
  const rad = N => KIT.NTX_DEG[N].map(d => d * Math.PI / 180);
  const draw = (ctx, W, H) => {
    if (!W) return;
    const Bc = PE.css('--blue'), Yc = PE.css('--yellow'), Gc = PE.css('--green'), Rc = PE.css('--red'), M = PE.css('--muted'), F = PE.css('--fg'), RU = PE.css('--rule');
    ctx.clearRect(0, 0, W, H);
    const N = state.N, n = 2 * N, r = KIT.ntxentRows(rad(N), state.tau), nar = KIT.narrow(W);
    const gPos = glow('pos'), gRow = glow('row'), gMask = glow('mask'), gTau = glow('tau'), gSim = glow('sim'), gAll = glow('rows');
    // circle panel
    const cw = nar ? W : W * .34, ch = nar ? H * .42 : H, R = Math.min(cw, ch) * .34, cx = cw / 2, cy = nar ? ch / 2 + 6 : ch / 2;
    PE.ring(ctx, cx, cy, R, RU, 1.5);
    const th = rad(N), pt = k => [cx + R * Math.cos(th[k]), cy - R * Math.sin(th[k])];
    for (let k = 0; k < n; k += 2) { const [x1, y1] = pt(k), [x2, y2] = pt(k + 1); ctx.save(); ctx.strokeStyle = Gc; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore(); }
    if (gSim) for (let k = 1; k < n; k++) { const [x, y] = pt(k); PE.polyline(ctx, [[cx, cy], [x, y]], M, 1, [3, 3], gSim); }
    for (let k = 0; k < n; k++) {
      const [x, y] = pt(k), col = k === 0 ? Bc : k === 1 ? Yc : Rc, a = PE.clamp(state.reveal * n - k, 0, 1);
      PE.dot(ctx, x, y, k < 2 ? 6.5 : 5.5, col, a);
      const lx = cx + (R + 15) * Math.cos(th[k]), ly = cy - (R + 15) * Math.sin(th[k]) + 4;
      PE.label(ctx, String(k + 1), lx, ly, col, 12, 'center');
    }
    PE.label(ctx, 'views on the circle', nar ? 8 : 10, 16, M, 11, 'left', 400);
    PE.label(ctx, 'anchor 1', 10, nar ? ch - 26 : H - 40, Bc, 11);
    PE.label(ctx, 'its partner 2', 10, nar ? ch - 12 : H - 26, Yc, 11);
    PE.label(ctx, 'negatives for row 1', 10, nar ? ch + 2 : H - 12, Rc, 11);
    // matrix panel
    const mx0 = nar ? 26 : cw + 26, my0 = nar ? ch + 46 : 40, barW = nar ? 70 : 110;
    const cs = Math.min((W - mx0 - barW - 30) / n, ((nar ? H - my0 : H - my0 - 30) - 22) / n), fsz = Math.max(10, Math.min(13, cs * .3));
    PE.label(ctx, 'cosine sim(z_i, z_k), shaded by row softmax', mx0, my0 - 22, M, 11, 'left', 400);
    PE.label(ctx, '\u03c4 = ' + (+state.tau.toFixed(2)), W - 8, my0 - 22, gTau > .1 ? Rc : F, 12 + 3 * gTau, 'right');
    for (let k = 0; k < n; k++) { PE.label(ctx, String(k + 1), mx0 + (k + .5) * cs, my0 - 5, k === 0 ? Bc : M, 10.5, 'center'); PE.label(ctx, String(k + 1), mx0 - 6, my0 + (k + .6) * cs, k === 0 ? Bc : M, 10.5, 'right'); }
    for (let i = 0; i < n; i++) {
      const a = PE.clamp(state.reveal * n - i, 0, 1);
      ctx.save(); ctx.globalAlpha = a;
      for (let k = 0; k < n; k++) {
        const x = mx0 + k * cs, y = my0 + i * cs, p = r.P[i][k];
        if (k === i) {
          ctx.fillStyle = PE.css('--panel2'); ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
          ctx.save(); ctx.strokeStyle = M; ctx.globalAlpha = a * (.5 + .5 * gMask); ctx.lineWidth = 1 + 2 * gMask;
          ctx.beginPath(); ctx.moveTo(x + 5, y + 5); ctx.lineTo(x + cs - 5, y + cs - 5); ctx.moveTo(x + cs - 5, y + 5); ctx.lineTo(x + 5, y + cs - 5); ctx.stroke(); ctx.restore();
          continue;
        }
        const pos = k === (i ^ 1);
        ctx.fillStyle = pos ? Gc : Rc; ctx.globalAlpha = a * (.08 + .82 * p); ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2); ctx.globalAlpha = a;
        if (pos) { ctx.strokeStyle = Yc; ctx.lineWidth = 2 + 2.5 * gPos; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); }
        PE.label(ctx, (Math.abs(r.S[i][k]) < .05 ? 0 : r.S[i][k]).toFixed(n > 4 ? 1 : 2), x + cs / 2, y + cs / 2 + fsz * .35, F, fsz, 'center', 400 + 300 * gSim);
      }
      ctx.restore();
    }
    ctx.save(); ctx.strokeStyle = Bc; ctx.lineWidth = 2 + 2.5 * gRow; ctx.strokeRect(mx0 - 1, my0 - 1, n * cs + 2, cs + 2); ctx.restore();
    // row losses
    const bx = mx0 + n * cs + 14, bmax = Math.max(...r.loss, .5), bw = barW - 34;
    PE.label(ctx, 'row loss', bx, my0 - 5, Rc, 11);
    r.loss.forEach((L, i) => { const y = my0 + i * cs, a = PE.clamp(state.reveal * n - i, 0, 1), h = Math.min(cs - 6, 16);
      ctx.save(); ctx.globalAlpha = a * (.75 + .25 * gAll); ctx.fillStyle = Rc; ctx.fillRect(bx, y + (cs - h) / 2, Math.max(1, L / bmax * bw), h); ctx.restore();
      PE.label(ctx, L.toFixed(2), bx + Math.max(1, L / bmax * bw) + 4, y + cs / 2 + 4, F, 10.5, 'left', 400 + 300 * gAll); });
    PE.label(ctx, 'mean over all ' + n + ' rows: ' + r.mean.toFixed(3), mx0, my0 + n * cs + 18, Rc, 11.5);
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? 1.5 : .46, draw);
  const set = (N, tau) => {
    cancel(); const grow = N !== state.N; state.N = N;
    if (PE.instant()) { state.tau = tau; state.reveal = 1; cv.redraw(); return; }
    const t0 = Math.log(state.tau), t1 = Math.log(tau); state.reveal = grow ? 0 : 1;
    cancel = PE.animate(grow ? 1100 : 700, u => { state.tau = Math.exp(PE.lerp(t0, t1, u)); if (grow) state.reveal = u; cv.redraw(); }, () => { state.tau = tau; cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
KIT.ntxentText = (N, tau) => {
  const r = KIT.ntxentRows(KIT.NTX_DEG[N].map(d => d * Math.PI / 180), tau), n = 2 * N;
  return 'N = ' + N + ' images give ' + n + ' views and ' + (n === 8 ? 'an ' : 'a ') + n + ' x ' + n + ' matrix. Each row has 1 masked cell, 1 positive and ' + (n - 2) +
    ' negatives. Row 1 at \u03c4 = ' + tau + ': the partner gets probability ' + r.P[0][1].toFixed(3) + ', so its loss is ' + r.loss[0].toFixed(3) +
    '. Mean over all rows: ' + r.mean.toFixed(3) + '.';
};
