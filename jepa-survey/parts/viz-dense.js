<script>
(() => {
  const fig = document.getElementById('viz-dense'), el = document.getElementById('dense-canvas'), out = document.getElementById('dense-readout');
  const G = 16, M = { top: 5, left: 6, h: 5, w: 6 }, S = { mode: 'v2', u: 1 };
  const inM = (i, j) => i >= M.top && i < M.top + M.h && j >= M.left && j < M.left + M.w;
  const dmin = (i, j) => { let d = 1e9; for (let a = M.top; a < M.top + M.h; a++) for (let b = M.left; b < M.left + M.w; b++) d = Math.min(d, Math.hypot(a - i, b - j)); return d; };
  const weight = (i, j) => inM(i, j) ? 1 : S.mode === 'v2' ? 0 : S.mode === 'flat' ? 1 : .5 / Math.sqrt(dmin(i, j));
  const cv = PE.canvas(el, w => w < 560 ? 1 : .5, (ctx, W, H) => {
    const C = { bg: PE.css('--bg'), yel: PE.css('--yellow'), blue: PE.css('--blue'), red: PE.css('--red'), mu: PE.css('--muted'), fg: PE.css('--fg'), rule: PE.css('--rule') };
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const narrow = W < 560, side = narrow ? W - 28 : Math.min(H - 36, W * .5), cs = side / G, ox = 14, oy = 24;
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const w = weight(i, j) * S.u + (S.prev ? S.prev[i][j] * (1 - S.u) : 0), m = inM(i, j);
      ctx.fillStyle = m ? C.yel : C.blue; ctx.globalAlpha = .18; ctx.fillRect(ox + j * cs + 1, oy + i * cs + 1, cs - 2, cs - 2);
      if (w > .001) { ctx.fillStyle = C.red; ctx.globalAlpha = .15 + .75 * Math.min(1, w); ctx.fillRect(ox + j * cs + 1, oy + i * cs + 1, cs - 2, cs - 2); }
      ctx.globalAlpha = 1; }
    PE.label(ctx, 'red = loss weight on this token', ox, oy - 8, C.red, 12);
    // Encoder depth stack on the right (or below on narrow)
    const sx = narrow ? ox : ox + side + 30, sy = narrow ? oy + side + 30 : oy + 10, sw = narrow ? W - 28 : W - sx - 14;
    const levels = S.mode === 'v2' ? [48] : [12, 24, 36, 48];
    [12, 24, 36, 48].forEach((b, k) => { const yy = sy + k * 30, on = levels.includes(b); ctx.fillStyle = on ? C.red : C.rule; ctx.globalAlpha = on ? .8 : .5; ctx.fillRect(sx, yy, Math.min(120, sw * .45), 20); ctx.globalAlpha = 1; PE.label(ctx, `block ${b}${on ? ': graded' : ''}`, sx + Math.min(120, sw * .45) + 8, yy + 14, on ? C.fg : C.mu, 12); });
    PE.label(ctx, 'encoder depths that receive the loss (ViT-G)', sx, sy - 8, C.mu, 11);
  });
  function set(m) { S.prev = []; for (let i = 0; i < G; i++) { S.prev.push([]); for (let j = 0; j < G; j++) S.prev[i].push(weight(i, j)); } S.mode = m;
    fig.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === m))); S.u = 0; PE.animate(500, u => { S.u = u; cv.redraw(); });
    out.textContent = { v2: 'V-JEPA 2: only the 30 masked tokens carry loss, at the last block. Visible tokens may drift into global summaries.',
      v21: 'V-JEPA 2.1: every token carries loss; visible tokens next to the hole weigh 0.5, three patches away 0.29, far away under 0.1. Applied at four depths.',
      flat: 'Flat weight 1.0 on every visible token: copying the context becomes the easy answer; the paper measures Something-Something falling from 72.8 to 51.1.' }[m]; }
  fig.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => set(b.dataset.mode)));
  document.getElementById('btn-dense-reset').addEventListener('click', () => set('v2'));
  PE.linkPulses(fig, () => cv.redraw()); set('v2');
})();
</script>
