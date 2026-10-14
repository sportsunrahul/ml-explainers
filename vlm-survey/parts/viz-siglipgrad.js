<script>
(() => {
  const fig = document.getElementById('viz-siglipgrad'), el = document.getElementById('siglipgrad-canvas'), out = document.getElementById('siglipgrad-readout');
  const names = ['dog', 'cat', 'car', 'pizza', 'tree'], N = 5, T = 10, BIAS = -10, LOW = .10, HIGH = .85;
  // Cosines of a table that is already mostly solved (same as build/sim/arch-siglip-grad.cjs); cell (0, 1) is the one that rises.
  const BASE = [[.80, .10, -.20, .05, -.10], [.15, .80, .00, -.25, .10], [-.10, .05, .80, .20, -.15], [.00, -.20, .10, .80, .05], [-.05, .15, -.10, .00, .80]];
  const S = { c01: LOW };
  let stop = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());
  const table = c => BASE.map((r, i) => r.map((v, j) => (i === 0 && j === 1 ? c : v)));
  const sig = u => 1 / (1 + Math.exp(-u));
  // Per-logit gradients times N (the shared 1/N is left out). Softmax: CLIP's symmetric loss, (p_ij + q_ij)/2 - delta_ij. Sigmoid: sigma(t c + b) - delta_ij.
  function gSoft(c) {
    const L = c.map(r => r.map(v => T * v)), sm = a => { const m = Math.max(...a), e = a.map(v => Math.exp(v - m)), s = e.reduce((x, y) => x + y); return e.map(v => v / s); };
    const p = L.map(sm), q = L.map(r => r.slice()); for (let j = 0; j < N; j++) { const col = sm(L.map(r => r[j])); for (let i = 0; i < N; i++) q[i][j] = col[i]; }
    return L.map((r, i) => r.map((_, j) => (p[i][j] + q[i][j]) / 2 - (i === j ? 1 : 0)));
  }
  const gSig = c => c.map((r, i) => r.map((v, j) => sig(T * v + BIAS) - (i === j ? 1 : 0)));
  const G0 = { soft: gSoft(table(LOW)), sig: gSig(table(LOW)) };
  const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2);
  const changed = (A, B, eps) => { let n = 0; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (Math.abs(A[i][j] - B[i][j]) > eps) n++; return n; };

  const cv = PE.canvas(el, w => w < 560 ? 1.62 : .54, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Br = PE.css('--brown'), Rd = PE.css('--red'), F = PE.css('--fg'), M = PE.css('--muted'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const small = W < 560, c = table(S.c01), grids = [['soft', 'softmax (CLIP)', gSoft(c)], ['sig', 'sigmoid (SigLIP)', gSig(c)]];
    const cs = small ? Math.min((W - 70) / N, (H / 2 - 72) / N) : Math.min((W / 2 - 80) / N, (H - 90) / N);
    grids.forEach(([key, title, g], k) => {
      const x0 = small ? (W - N * cs) / 2 + 22 : W / 2 * k + (W / 2 - N * cs) / 2 + 26, y0 = small ? 60 + k * (H / 2 - 8) : 66;
      PE.label(ctx, title, x0 + N * cs / 2, y0 - 26, F, 12.5, 'center');
      names.forEach((n, j) => PE.label(ctx, '"' + n + '"', x0 + (j + .5) * cs, y0 - 8, Y, small ? 9 : 10, 'center'));
      names.forEach((n, i) => PE.label(ctx, n, x0 - 6, y0 + (i + .5) * cs + 4, B, small ? 9 : 10, 'right'));
      const rc = key === 'soft' ? glow('rowcol') : 0, own = glow('own'), cell = glow('cell');
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const v = g[i][j], a = PE.clamp(Math.abs(v) / .6, 0, 1), x = x0 + j * cs, y = y0 + i * cs;
        ctx.fillStyle = v < 0 ? PE.mix(BG, G, .12 + .8 * a) : PE.mix(BG, Br, .12 + .8 * a); ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
        if (rc > 0 && (i === 0 || j === 1)) { ctx.fillStyle = Rd; ctx.globalAlpha = .25 * rc; ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2); ctx.globalAlpha = 1; }
        if (Math.abs(v - G0[key][i][j]) > .005 && !(i === 0 && j === 1)) { ctx.strokeStyle = F; ctx.lineWidth = 2; ctx.strokeRect(x + 2.5, y + 2.5, cs - 5, cs - 5); }
        if (i === j && own > 0) { ctx.strokeStyle = G; ctx.lineWidth = 2 + 2 * own; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); }
        PE.label(ctx, (v >= .005 ? '+' : '') + f2(v), x + cs / 2, y + cs / 2 + 4, a > .45 ? BG : F, small ? 9.5 : 10.5, 'center');
      }
      ctx.strokeStyle = Rd; ctx.lineWidth = 2.5 + 2 * cell; ctx.strokeRect(x0 + cs + 1, y0 + 1, cs - 2, cs - 2);
    });
    PE.label(ctx, 'cells: gradient on each logit, times N', 8, H - 6, M, 10.5);
  });
  function readout() {
    const c = table(S.c01), s = gSoft(c), g = gSig(c), up = S.c01 > (LOW + HIGH) / 2;
    out.textContent = `The dog photo and the caption "cat" now have cosine ${S.c01.toFixed(2)}. ` +
      (up ? `Softmax: ${changed(G0.soft, s, 1e-6)} cells' gradients changed (all of row "dog" and column "cat"), ${changed(G0.soft, s, .005)} of them visibly; the dog photo's own cell now pulls with ${f2(s[0][0])} instead of ${f2(G0.soft[0][0])}. ` +
        `Sigmoid: ${changed(G0.sig, g, 1e-6)} cell changed, the one that rose (+${f2(g[0][1])}); the dog photo's own cell pulls with ${f2(g[0][0])} either way.`
        : 'Every wrong cell is low, so the softmax gradients are all near zero: each row and column already picks its own caption. The sigmoid still pulls every true pair, because a cosine of 0.80 gives a logit of 10 x 0.80 - 10 = -2, still on the "no" side.');
  }
  function tween(to) { stop(); const a = S.c01; stop = PE.animate(800, u => { S.c01 = PE.lerp(a, to, u); cv.redraw(); readout(); }); }
  document.getElementById('btn-siglipgrad-before').addEventListener('click', () => tween(LOW));
  document.getElementById('btn-siglipgrad-raise').addEventListener('click', () => tween(HIGH));
  document.getElementById('btn-siglipgrad-reset').addEventListener('click', () => { stop(); S.c01 = LOW; cv.redraw(); readout(); });
  readout();
})();
</script>
