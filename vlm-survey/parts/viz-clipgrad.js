<script>
(() => {
  const fig = document.getElementById('viz-clipgrad'), el = document.getElementById('clipgrad-canvas');
  const out = document.getElementById('clipgrad-readout');
  const names = ['dog', 'cat', 'car', 'pizza', 'tree'], N = names.length, TAU = .07;
  const deg = d => d * Math.PI / 180;
  // Captions evenly spread; the other four photos already sit near their own captions (same as build/sim/clipgrad.cjs).
  const PH = [90, 162, 234, 306, 18].map(deg), OTH = [0, 168, 228, 300, 24].map(deg);
  const START = { far: deg(330), confused: deg(160), solved: deg(96) };
  const S = { a: START.far, rad: 0, sel: true };
  let stop = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());
  const sm = v => PE.softmax(v);
  // Exact gradient of CLIP's symmetric loss w.r.t. the dog photo's unit vector u, without the common 1/(N tau) factor:
  // coefficient c_j = (p_j + q_j)/2 - [j=0]; dL/du is proportional to sum_j c_j v_j.
  function grad(a) {
    const th = OTH.map((t, i) => i === 0 ? a : t), cos = (x, y) => Math.cos(x - y);
    const p = sm(PH.map(ph => cos(a, ph) / TAU)), q = PH.map(ph => sm(th.map(t => cos(t, ph) / TAU))[0]);
    const c = PH.map((_, j) => (p[j] + q[j]) / 2 - (j === 0 ? 1 : 0));
    let gx = 0, gy = 0; PH.forEach((ph, j) => { gx += c[j] * Math.cos(ph); gy += c[j] * Math.sin(ph); });
    const ux = Math.cos(a), uy = Math.sin(a), r = gx * ux + gy * uy;
    return { p, q, c, gx, gy, r, tx: gx - r * ux, ty: gy - r * uy };
  }
  const layout = (W, H) => { const R = Math.min(W, H) * .31; return { cx: W / 2, cy: H / 2 - 4, R }; };
  const cv = PE.canvas(el, w => w < 560 ? 1.15 : .62, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Br = PE.css('--brown'), F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const { cx, cy, R } = layout(W, H), P = a => [cx + R * Math.cos(a), cy - R * Math.sin(a)], small = W < 560;
    ctx.strokeStyle = Rule; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    const lab = (a, t, col, rr) => PE.label(ctx, t, cx + rr * Math.cos(a), cy - rr * Math.sin(a) + 4, col, small ? 10.5 : 11.5, Math.cos(a) > .3 ? 'left' : Math.cos(a) < -.3 ? 'right' : 'center');
    PH.forEach((ph, j) => { const [x, y] = P(ph), s = 7; ctx.fillStyle = Y; ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill();
      lab(ph, '"' + names[j] + '"', Y, R - 26); });
    OTH.forEach((t, i) => { if (i) { const [x, y] = P(t); PE.dot(ctx, x, y, 5, B, .45); } });
    const g = grad(S.a), [px, py] = P(S.a), k = R * .62, gp = glow('pos'), gn = glow('neg'), gt = glow('tau'), gr = glow('radial');
    PE.dot(ctx, px, py, 9, B); lab(S.a, 'dog photo', B, R + 16);
    // Descent direction is minus the gradient: each caption contributes -c_j v_j (drawn from the photo).
    PH.forEach((ph, j) => { const c = -g.c[j]; if (Math.abs(c) < .02) return;
      const dx = c * Math.cos(ph) * k, dy = -c * Math.sin(ph) * k, own = j === 0, w = own ? gp : gn;
      PE.arrow(ctx, px, py, px + dx, py + dy, own ? G : Br, 2.5 + 2 * w + gt, 9, .95);
      PE.label(ctx, (own ? 'pull ' : 'push ') + Math.abs(c).toFixed(2), px + dx * 1.08 + (dx >= 0 ? 4 : -4), py + dy * 1.08 + 4, own ? G : Br, 10.5, dx >= 0 ? 'left' : 'right'); });
    if (S.rad > 0) {
      const nx = -g.gx * k, ny = g.gy * k, ux = Math.cos(S.a), uy = -Math.sin(S.a), rr = -g.r * k, tx = -g.tx * k, ty = g.ty * k;
      ctx.save(); ctx.globalAlpha = S.rad;
      PE.polyline(ctx, [[px, py], [px + nx, py + ny]], F, 1.2, [3, 3], .6);
      PE.arrow(ctx, px, py, px + rr * ux, py + rr * uy, M, 2 + 2 * gr, 8, .9);
      PE.arrow(ctx, px, py, px + tx, py + ty, F, 3 + 2 * gr, 10, 1);
      const ly = H - 34; PE.label(ctx, 'dashed: raw sum of the arrows', small ? 8 : 12, ly, F, 10.5);
      PE.label(ctx, 'grey: radial part, deleted by the normalization', small ? 8 : 12, ly + 14, M, 10.5);
      PE.label(ctx, 'bold: what is left, it slides the photo along the circle', small ? 8 : 12, ly + 28, F, 10.5);
      ctx.restore();
    }
    if (S.sel && document.activeElement === el) { ctx.strokeStyle = F; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, 13, 0, 7); ctx.stroke(); }
    PE.label(ctx, 'arrows: minus the gradient on the dog photo (τ = 0.07)', small ? 8 : 12, small ? 44 : 18, M, 11);
  });
  function readout() {
    const g = grad(S.a), push = PH.map((_, j) => j).filter(j => j && g.c[j] > .02);
    const tan = Math.hypot(g.tx, g.ty);
    out.textContent = `Pull toward "dog": ${(-g.c[0]).toFixed(2)}. ` + (push.length ? 'Pushes: ' + push.map(j => `"${names[j]}" ${g.c[j].toFixed(2)}`).join(', ') + '. ' : 'No caption pushes. ') +
      `The other ${N - 1 - push.length} captions push with less than 0.02: they are already far away. ` + (tan < .02 ? 'The photo already wins its row and column, so the gradient is almost zero.' : '');
  }
  function tweenTo(a, ms = 700) { stop(); const a0 = S.a; let d = a - a0; d = Math.atan2(Math.sin(d), Math.cos(d));
    stop = PE.animate(ms, u => { S.a = a0 + d * u; cv.redraw(); readout(); }); }
  const set = key => { S.rad = 0; tweenTo(START[key]); };
  document.getElementById('btn-clipgrad-far').addEventListener('click', () => set('far'));
  document.getElementById('btn-clipgrad-confused').addEventListener('click', () => set('confused'));
  document.getElementById('btn-clipgrad-solved').addEventListener('click', () => set('solved'));
  document.getElementById('btn-clipgrad-radial').addEventListener('click', () => { stop(); const r0 = S.rad; stop = PE.animate(600, u => { S.rad = PE.lerp(r0, 1, u); cv.redraw(); }); });
  // Plain gradient descent on the photo's angle, starting from where it is now.
  document.getElementById('btn-clipgrad-descend').addEventListener('click', () => {
    stop(); S.rad = 0; let a = S.a; const F = [a];
    for (let s = 0; s < 240; s++) { const g = grad(a); a -= .08 * (g.gx * -Math.sin(a) + g.gy * Math.cos(a)); if (s % 3 === 2) F.push(a); }
    stop = PE.animate(2400, u => { const f = u * (F.length - 1), i = Math.floor(f), w = f - i; S.a = PE.lerp(F[i], F[Math.min(i + 1, F.length - 1)], w); cv.redraw(); readout(); });
  });
  document.getElementById('btn-clipgrad-reset').addEventListener('click', () => { stop(); S.a = START.far; S.rad = 0; cv.redraw(); readout(); });
  let drag = false;
  const ang = e => { const b = el.getBoundingClientRect(), L = layout(b.width, b.height); return Math.atan2(L.cy - (e.clientY - b.top), e.clientX - b.left - L.cx); };
  el.addEventListener('pointerdown', e => { const b = el.getBoundingClientRect(), L = layout(b.width, b.height), X = e.clientX - b.left, Yp = e.clientY - b.top;
    const px = L.cx + L.R * Math.cos(S.a), py = L.cy - L.R * Math.sin(S.a); if ((px - X) ** 2 + (py - Yp) ** 2 < 22 * 22) { stop(); drag = true; el.setPointerCapture(e.pointerId); } });
  el.addEventListener('pointermove', e => { if (!drag) return; S.a = ang(e); cv.redraw(); readout(); });
  el.addEventListener('pointerup', () => { drag = false; });
  el.addEventListener('keydown', e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); stop(); S.a += e.key === 'ArrowLeft' ? .06 : -.06; cv.redraw(); readout(); } });
  el.addEventListener('focus', () => cv.redraw()); el.addEventListener('blur', () => cv.redraw());
  readout();
})();
</script>
