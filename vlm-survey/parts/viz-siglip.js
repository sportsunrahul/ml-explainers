<script>
(() => {
  const fig = document.getElementById('viz-siglip'), el = document.getElementById('siglip-canvas');
  const out = document.getElementById('siglip-readout');
  const names = ['dog', 'cat', 'car', 'pizza', 'tree'], N = names.length, TAU = .1;  // t = 1/TAU = 10, SigLIP's init
  // Same scramble as chapter 1 (seed 65, angles drawn photos first, then captions).
  const r0 = PE.rng(65), init = { th: names.map(() => r0() * 2 * Math.PI), ph: names.map(() => r0() * 2 * Math.PI) };
  const S = { th: init.th.slice(), ph: init.ph.slice(), b: -10, mt: 0, cell: [0, 1], trained: null };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};
  const sig = u => 1 / (1 + Math.exp(-u));
  // A photo is matched only when its own caption is strictly closest. Collapsed points tie exactly, and argmax would hand the tie to the first caption.
  const wins = (row, i) => row.every((s, j) => j === i || row[i] - s > 1e-9);

  // Real gradient descent on the sigmoid loss: angles and the bias b are learned, t stays at 10.
  function trainFrames(b0) {
    let th = init.th.slice(), ph = init.ph.slice(), b = b0; const F = [[th, ph, b]], lr = .05;
    for (let k = 0; k < 400; k++) { const g = PE.contrast(th, ph, TAU, 'sigmoid', b);
      th = th.map((v, i) => v - lr * g.gth[i]); ph = ph.map((v, i) => v - lr * g.gph[i]); b -= lr * g.gb;
      if (k % 4 === 3) F.push([th, ph, b]); }
    return F;
  }

  const layout = (W, H) => {
    const stacked = W < 560;
    if (stacked) { const R = W * .27; return { stacked, cx: W / 2, cy: R + 34, R, mx: 64, my: 2 * R + 118, cell: Math.min(44, (W - 84) / N) }; }
    const R = Math.min(H * .36, W * .19), cx = R + 82, mx = cx + R + 118; return { stacked, cx, cy: H / 2 + 4, R, mx, my: 66, cell: Math.min(46, (W - mx - 16) / N) };
  };

  const cv = PE.canvas(el, w => w < 560 ? 1.95 : .6, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Rd = PE.css('--red'), Br = PE.css('--brown'),
      F = PE.css('--fg'), M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    const bg = BG.length === 7 ? BG : '#16181d';
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const Lh = layout(W, H), { cx, cy, R } = Lh;
    const P = a => [cx + R * Math.cos(a), cy - R * Math.sin(a)];
    const res = PE.contrast(S.th, S.ph, TAU, 'sigmoid', S.b), Sm = res.S;
    const gLab = glow('label'), gLog = glow('logit'), gBias = glow('bias'), gPairs = glow('pairs'), gRow = glow('row'), gCol = glow('col');

    ctx.strokeStyle = Rule; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    PE.label(ctx, 'shared embedding space', cx, cy + 4, M, 11, 'center');
    for (let i = 0; i < N; i++) {
      let best = 0; for (let j = 1; j < N; j++) if (Sm[i][j] > Sm[i][best]) best = j;
      const [x1, y1] = P(S.th[i]), [x2, y2] = P(S.ph[best]), ok = best === i && wins(Sm[i], i);
      PE.polyline(ctx, [[x1, y1], [x2, y2]], ok ? G : Rd, 2, ok ? null : [5, 4], .8);
    }
    const lab = (a, text, color, rr) => { const [x, y] = [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
      PE.label(ctx, text, x, y + 4, color, Lh.stacked ? 10.5 : 11.5, Math.cos(a) > .3 ? 'left' : Math.cos(a) < -.3 ? 'right' : 'center'); };
    // Points that have collapsed onto one spot share one label, so the collapse reads as a collapse, not as overprinting.
    const groups = A => { const G = []; A.forEach((a, k) => { const g = G.find(g => Math.abs(Math.atan2(Math.sin(a - A[g[0]]), Math.cos(a - A[g[0]]))) < .14);
      g ? g.push(k) : G.push([k]); }); return G; };
    const [si, sj] = S.cell;
    for (let j = 0; j < N; j++) { const [x, y] = P(S.ph[j]), s = 7 + (j === sj ? 2 : 0);
      ctx.fillStyle = Y; ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill();
    }
    groups(S.ph).forEach(g => lab(S.ph[g[0]], g.map(j => '"' + names[j] + '"').join(' '), Y, R - 26));
    for (let i = 0; i < N; i++) { const [x, y] = P(S.th[i]); PE.dot(ctx, x, y, 7 + (i === si ? 2 : 0), B);
    }
    groups(S.th).forEach(g => lab(S.th[g[0]], g.map(i => names[i]).join('/') + (g.length > 1 ? ' photos' : ' photo'), B, R + 14));

    // The table. Color = what the loss sees for that cell: the row softmax (mt = 0) or the cell's own sigmoid (mt = 1).
    const { mx, my, cell } = Lh, mt = S.mt;
    PE.label(ctx, mt < .5 ? 'softmax view: color = share of its row' : 'sigmoid view: color = P(match) for this cell alone', Lh.stacked ? 8 : mx - 50, my - 30, F, 12);
    for (let i = 0; i < N; i++) {
      const row = PE.softmax(Sm[i].map(s => s / TAU));
      PE.label(ctx, names[i], mx - 6, my + i * cell + cell * .6, B, 11, 'right');
      for (let j = 0; j < N; j++) {
        const x = mx + j * cell, y = my + i * cell, diag = i === j, ps = sig(Sm[i][j] / TAU + S.b), p = PE.lerp(row[j], ps, mt);
        ctx.fillStyle = PE.mix(bg, diag ? G : Br, .1 + .9 * p); ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        if (gLab > 0) { ctx.strokeStyle = diag ? G : Br; ctx.lineWidth = 3 * gLab; ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3); }
        if (gPairs > 0) { ctx.strokeStyle = F; ctx.globalAlpha = gPairs; ctx.lineWidth = 1.5; ctx.strokeRect(x + 3, y + 3, cell - 6, cell - 6); ctx.globalAlpha = 1; }
        PE.label(ctx, Math.round(100 * (mt < .5 ? row[j] : ps)) + '%', x + cell / 2, y + cell * .6, F, cell < 40 ? 9.5 : 10.5, 'center');
      }
    }
    for (let j = 0; j < N; j++) PE.label(ctx, names[j], mx + j * cell + cell / 2, my - 6, Y, 11, 'center');
    // What one loss term touches: the softmax couples a whole row and a whole column; the sigmoid touches one cell.
    ctx.save(); ctx.strokeStyle = Rd;
    const coup = 1 - mt;
    ctx.globalAlpha = Math.max(coup, gRow); ctx.lineWidth = 2.5 + 2 * gRow; ctx.strokeRect(mx - 1, my + si * cell - 1, N * cell + 2, cell + 2);
    ctx.globalAlpha = Math.max(coup, gCol); ctx.lineWidth = 2.5 + 2 * gCol; ctx.setLineDash([6, 4]); ctx.strokeRect(mx + sj * cell - 1, my - 1, cell + 2, N * cell + 2);
    ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.lineWidth = 3 + 2.5 * gLog; ctx.strokeRect(mx + sj * cell + 1, my + si * cell + 1, cell - 2, cell - 2);
    ctx.restore();

    const ly = my + N * cell + 24, bw = cell * N;
    PE.label(ctx, 'sigmoid loss ' + res.L.toFixed(2), mx, ly - 5, Rd, 12);
    ctx.fillStyle = Rule; ctx.fillRect(mx, ly, bw, 10); ctx.fillStyle = Rd; ctx.fillRect(mx, ly, bw * Math.min(1, res.L / 14), 10);
    // Bias: where the "no" prior sits. Axis from -12 to 2; tick at 0.
    const by = ly + 36, bx = v => mx + (v + 12) / 14 * bw;
    PE.label(ctx, 'bias b = ' + S.b.toFixed(2), mx, by - 7, gBias > 0 ? F : M, 11.5 + 1.5 * gBias);
    ctx.fillStyle = Rule; ctx.fillRect(mx, by, bw, 3);
    ctx.fillStyle = M; ctx.fillRect(bx(0) - 1, by - 4, 2, 11); PE.label(ctx, '0', bx(0), by + 18, M, 10, 'center');
    PE.label(ctx, '-10', bx(-10), by + 18, M, 10, 'center');
    PE.dot(ctx, bx(PE.clamp(S.b, -12, 2)), by + 1.5, 6 + 3 * gBias, F);
  });

  function stats() {
    const res = PE.contrast(S.th, S.ph, TAU, 'sigmoid', S.b);
    let right = 0, yes = 0, wrongYes = 0;
    res.S.forEach((row, i) => { if (wins(row, i)) right++;
      row.forEach((s, j) => { if (s / TAU + S.b > 0) { yes++; if (i !== j) wrongYes++; } }); });
    return { res, right, yes, wrongYes };
  }
  function readout() {
    const { res, right, yes, wrongYes } = stats(), [i, j] = S.cell;
    const term = S.mt < .5
      ? `Softmax view: the cell (${names[i]} photo, "${names[j]}") sits inside the ${names[i]} photo's row softmax and the "${names[j]}" caption's column softmax, so its loss terms couple ${2 * N - 1} cells.`
      : `Sigmoid view: the cell (${names[i]} photo, "${names[j]}") is its own yes/no question; its loss term touches only that cell.`;
    const state = S.trained === null ? '' : ` Trained from bias ${S.trained}: ${right} of ${N} photos are closer to their own caption than to any other; bias now ${S.b.toFixed(2)}, loss ${res.L.toFixed(2)}.`;
    out.textContent = `${term} ${yes} of ${N * N} cells currently say "match" (${wrongYes} of them wrongly).${state}`;
  }
  function tweenMode(to, ms = 600) { const m0 = S.mt; stop(); stop = PE.animate(ms, u => { S.mt = PE.lerp(m0, to, u); cv.redraw(); readout(); }); }
  function train(b0) {
    stop(); S.th = init.th.slice(); S.ph = init.ph.slice(); S.b = b0; S.trained = b0; const m0 = S.mt;
    const Fr = trainFrames(b0);
    stop = PE.animate(2800, u => { S.mt = PE.lerp(m0, 1, Math.min(1, u * 6));
      const f = u * (Fr.length - 1), k = Math.floor(f), w = f - k, A = Fr[k], Bn = Fr[Math.min(k + 1, Fr.length - 1)];
      S.th = A[0].map((v, i) => PE.lerp(v, Bn[0][i], w)); S.ph = A[1].map((v, i) => PE.lerp(v, Bn[1][i], w)); S.b = PE.lerp(A[2], Bn[2], w);
      cv.redraw(); readout(); });
  }
  document.getElementById('btn-siglip-softmax').addEventListener('click', () => tweenMode(0));
  document.getElementById('btn-siglip-sigmoid').addEventListener('click', () => tweenMode(1));
  document.getElementById('btn-siglip-b10').addEventListener('click', () => train(-10));
  document.getElementById('btn-siglip-b0').addEventListener('click', () => train(0));
  document.getElementById('btn-siglip-reset').addEventListener('click', () => {
    stop(); S.th = init.th.slice(); S.ph = init.ph.slice(); S.b = -10; S.mt = 0; S.cell = [0, 1]; S.trained = null; cv.redraw(); readout(); });

  // Pick a cell: click it, or Tab to the canvas and use the arrow keys.
  el.addEventListener('pointerdown', e => { const b = el.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top, L = layout(b.width, b.height);
    const j = Math.floor((x - L.mx) / L.cell), i = Math.floor((y - L.my) / L.cell);
    if (i >= 0 && i < N && j >= 0 && j < N) { S.cell = [i, j]; cv.redraw(); readout(); } });
  el.addEventListener('keydown', e => { const d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key];
    if (!d) return; e.preventDefault(); S.cell = [(S.cell[0] + d[0] + N) % N, (S.cell[1] + d[1] + N) % N]; cv.redraw(); readout(); });
  readout();
})();
</script>
