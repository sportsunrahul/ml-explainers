<script>
(() => {
  // I-JEPA multi-block mask sampler, following src/masks/multiblock.py: block scale and aspect ratio are
  // drawn uniformly, h = round(sqrt(n * ar)), w = round(sqrt(n / ar)); the context loses every target patch.
  const fig = document.getElementById('viz-ijmask'), el = document.getElementById('ijmask-canvas'), out = document.getElementById('ijmask-readout');
  const S = { G: 14, strat: 'multi', seed: 1, grid: null, u: 1 };
  const CFG = { multi: { nt: 4, ts: [.15, .2], ta: [.75, 1.5], cs: [.85, 1] }, tiny: { nt: 4, ts: [.0075, .01], ta: [1, 1], cs: [.85, 1] } };
  function block(r, G, scale, ar) {
    const n = Math.floor(G * G * (scale[0] + r() * (scale[1] - scale[0]))), a = ar[0] + r() * (ar[1] - ar[0]);
    let h = Math.max(1, Math.round(Math.sqrt(n * a))), w = Math.max(1, Math.round(Math.sqrt(n / a)));
    h = Math.min(h, G); w = Math.min(w, G);
    const top = Math.floor(r() * (G - h + 1)), left = Math.floor(r() * (G - w + 1));
    return { top, left, h, w };
  }
  function sample() {
    const G = S.G, r = PE.rng(S.seed * 7919 + G), g = []; for (let i = 0; i < G; i++) { g.push(new Array(G).fill(0)); } // 0 = unused, 1 = context, 2 = target, 3 = removed from context
    const paint = (b, v) => { for (let i = b.top; i < b.top + b.h; i++) for (let j = b.left; j < b.left + b.w; j++) g[i][j] = v; };
    const st = S.strat;
    if (st === 'multi' || st === 'tiny') { const c = CFG[st];
      const ctx = block(r, G, c.cs, [1, 1]); paint(ctx, 1);
      for (let k = 0; k < c.nt; k++) { const b = block(r, G, c.ts, c.ta); for (let i = b.top; i < b.top + b.h; i++) for (let j = b.left; j < b.left + b.w; j++) g[i][j] = g[i][j] === 1 || g[i][j] === 3 ? 3 : 2; }
      // a removed patch is one the context block covered but a target claimed; targets themselves show yellow
      for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) if (g[i][j] === 3) g[i][j] = 2;
    } else if (st === 'raster') { const q = Math.floor(r() * 4), h = Math.ceil(G / 2);
      for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const qq = (i < h ? 0 : 2) + (j < h ? 0 : 1); g[i][j] = qq === q ? 1 : 2; }
    } else if (st === 'block') { const b = block(r, G, [.6, .6], [1, 1]); for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) g[i][j] = 1; paint(b, 2);
    } else { for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) g[i][j] = r() < .6 ? 2 : 1; }
    S.grid = g;
  }
  const cv = PE.canvas(el, w => w < 560 ? 1.35 : .55, draw);
  function draw(ctx, W, H) {
    const C = { bg: PE.css('--bg'), blue: PE.css('--blue'), yel: PE.css('--yellow'), mu: PE.css('--muted'), rule: PE.css('--rule'), fg: PE.css('--fg') };
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const G = S.G, side = W < 560 ? Math.min(W - 28, H - 150) : Math.min(H - 40, W * .5), cs = side / G, ox = 14, oy = 26;
    let nc = 0, nt = 0;
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const v = S.grid[i][j];
      ctx.globalAlpha = v === 1 ? .85 * S.u : v === 2 ? .8 : .18; ctx.fillStyle = v === 1 ? C.blue : v === 2 ? C.yel : C.mu;
      ctx.fillRect(ox + j * cs + 1, oy + i * cs + 1, cs - 2, cs - 2); ctx.globalAlpha = 1; if (v === 1) nc++; if (v === 2) nt++; }
    PE.label(ctx, `${G} x ${G} patches`, ox, oy - 8, C.mu, 12);
    const narrow = W < 560, lx = narrow ? ox : ox + side + 22, ly = narrow ? oy + side + 22 : oy + 10;
    const key = [[C.blue, `context: ${nc} patches seen by the context encoder`], [C.yel, `targets: ${nt} patches to predict`], [C.mu, 'unused: neither seen nor predicted']];
    key.forEach(([c, t], k) => { ctx.fillStyle = c; ctx.globalAlpha = k === 2 ? .3 : .85; ctx.fillRect(lx, ly + k * 24 - 10, 14, 14); ctx.globalAlpha = 1; PE.label(ctx, t, lx + 22, ly + k * 24 + 2, C.fg, 12); });
    PE.label(ctx, `context fraction ${(nc / (G * G)).toFixed(2)}, target fraction ${(nt / (G * G)).toFixed(2)}`, lx, ly + 86, C.mu, 12);
    S.last = { nc, nt };
  }
  function say() { const { nc, nt } = S.last, G = S.G;
    out.textContent = { multi: `Multi-block: ${nt} target patches in four blocks, ${nc} context patches left after removing the overlap (fraction ${(nc / (G * G)).toFixed(2)}; the paper measures 0.25 on average). Each target needs a guess about a region, not a copy of a neighbour.`,
      tiny: `Tiny targets: each block is about one patch. The predictor can fill it by interpolating its neighbours; the paper's Table 8 shows accuracy collapsing as targets shrink.`,
      raster: 'Rasterised: one quadrant predicts the other three. Lots to predict, little to predict from; 15.5% in Table 6.',
      block: 'Single block: one large target, the rest is context. 20.2% in Table 6.',
      random: 'Random patches: 60% dropped at random. Every hole has visible neighbours, so copying works; 17.6% in Table 6.' }[S.strat]; }
  function go(anim) { sample(); if (anim) { S.u = 0; PE.animate(450, u => { S.u = u; cv.redraw(); }, say); } else { S.u = 1; cv.redraw(); say(); } }
  document.getElementById('btn-ijmask-sample').addEventListener('click', () => { S.seed++; go(true); });
  fig.querySelectorAll('[data-strat]').forEach(b => b.addEventListener('click', () => { S.strat = b.dataset.strat; fig.querySelectorAll('[data-strat]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); go(true); }));
  fig.querySelectorAll('[data-grid]').forEach(b => b.addEventListener('click', () => { S.G = +b.dataset.grid; fig.querySelectorAll('[data-grid]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); go(true); }));
  document.getElementById('btn-ijmask-reset').addEventListener('click', () => { S.seed = 1; S.G = 14; S.strat = 'multi';
    fig.querySelectorAll('[data-strat]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.strat === 'multi'))); fig.querySelectorAll('[data-grid]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.grid === '14'))); go(false); });
  PE.linkPulses(fig, () => cv.redraw());
  go(false);
})();
</script>
