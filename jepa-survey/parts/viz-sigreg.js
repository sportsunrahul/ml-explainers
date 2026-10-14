<script>
(() => {
  const fig = document.getElementById('viz-sigreg'), el = document.getElementById('sigreg-canvas'), out = document.getElementById('sigreg-readout');
  const N = 600, S = { cloud: 'stretched', dir: [1, 0, 0], dirName: 'x', seed: 5 };
  const base = PE3.gaussian(N, { scale: [1, 1, 1], seed: 11 });
  const shape = { stretched: p => [p[0] * 2.2, p[1] * .9, p[2] * .25], iso: p => p.slice(), clumpy: (p, k) => { const c = [[1.1, .3, .2], [-1, -.4, .5], [.1, .9, -1]][k % 3]; return [c[0] + p[0] * .35, c[1] + p[1] * .35, c[2] + p[2] * .35]; } };
  let pts = base.map((p, k) => shape.stretched(p, k));
  // Epps-Pulley for one direction (Algorithm 1 of LeJEPA), divided by N for display.
  function ep(dir) { const n = Math.hypot(...dir), a = dir.map(v => v / n), u = pts.map(p => a[0] * p[0] + a[1] * p[1] + a[2] * p[2]);
    const t = Array.from({ length: 17 }, (_, k) => -5 + 10 * k / 16), dt = 10 / 16; const err = t.map(tk => { let re = 0, im = 0; for (const x of u) { re += Math.cos(tk * x); im += Math.sin(tk * x); } re /= u.length; im /= u.length; const phi = Math.exp(-tk * tk / 2); return ((re - phi) ** 2 + im * im) * phi; });
    let I = 0; for (let k = 0; k < 16; k++) I += (err[k] + err[k + 1]) / 2 * dt; return { T: I * u.length, Tn: I, u }; }
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? 1.1 : .55, fov: 36, dist: 7.4, yaw: .5, pitch: .3 });
  function scene() {
    const r = ep(S.dir), a = S.dir.map(v => v / Math.hypot(...S.dir));
    const items = [{ type: 'grid', y: -2.2, size: 5, step: 1, color: '--rule' }, { type: 'points', pts, color: '--blue', size: 4, alpha: .8, link: 'cloud' },
      { type: 'line', pts: [a.map(v => -2.6 * v), a.map(v => 2.6 * v)], color: '--yellow', width: 3, arrow: true, link: 'dir' }, { type: 'label', at: a.map(v => 2.7 * v), text: 'test direction', color: '--yellow', dy: -8 }];
    s3.items = items; s3.redraw(); S.last = r;
    out.textContent = `Direction ${S.dirName}: Epps-Pulley statistic ${r.Tn.toFixed(3)} per sample (${r.T.toFixed(0)} scaled by N). ${r.Tn < .004 ? 'Passes: the 1D projection looks standard Gaussian.' : 'Fails: the projection is not standard Gaussian.'} ${S.cloud === 'iso' ? 'An isotropic Gaussian cloud passes in every direction.' : S.cloud === 'stretched' ? 'The stretched cloud passes only along its unit-variance axis.' : 'A clumpy cloud fails on most directions even though its overall variance is modest.'}`;
    drawHist(r.u);
  }
  // Histogram overlay drawn by the page on the same canvas after pe3 renders: use a second canvas stacked via CSS would be cleaner; here we draw into a corner box on each redraw.
  const hist = document.createElement('canvas'); hist.className = 'pe3-hist'; hist.style.cssText = 'position:absolute;right:1.2rem;bottom:3.6rem;width:200px;height:120px;border-radius:8px'; hist.setAttribute('aria-hidden', 'true'); fig.style.position = 'relative'; fig.appendChild(hist);
  function drawHist(u) { const r = devicePixelRatio || 1; hist.width = 200 * r; hist.height = 120 * r; const ctx = hist.getContext('2d'); ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.fillStyle = PE.css('--panel2'); ctx.fillRect(0, 0, 200, 120); const bins = new Array(30).fill(0); for (const x of u) { const b = Math.floor((x + 4) / 8 * 30); if (b >= 0 && b < 30) bins[b]++; }
    const mx = Math.max(...bins, 1); ctx.fillStyle = PE.css('--yellow'); bins.forEach((c, i) => { const h = 90 * c / mx; ctx.globalAlpha = .7; ctx.fillRect(10 + i * 6, 105 - h, 5, h); }); ctx.globalAlpha = 1;
    ctx.strokeStyle = PE.css('--fg'); ctx.lineWidth = 1.5; ctx.beginPath(); for (let i = 0; i <= 60; i++) { const x = -4 + 8 * i / 60, y = Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI) * u.length * (8 / 30); const h = 90 * y / mx; i ? ctx.lineTo(10 + i * 3, 105 - h) : ctx.moveTo(10 + i * 3, 105 - h); } ctx.stroke();
    PE.label(ctx, 'projection vs N(0,1)', 10, 12, PE.css('--muted'), 10); }
  function setCloud(k) { S.cloud = k; fig.querySelectorAll('[data-cloud]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.cloud === k))); const from = pts.map(p => p.slice()), to = base.map((p, i) => shape[k](p, i));
    s3.tween(700, u => { pts = from.map((p, i) => p.map((v, j) => v + (to[i][j] - v) * u)); scene(); }); }
  function setDir(k) { S.dirName = k; fig.querySelectorAll('[data-dir]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.dir === k))); if (k === 'rand') { const r = PE.rng(S.seed++), g = PE.gauss(r); S.dir = [g(), g(), g()]; } else S.dir = k === 'x' ? [1, 0, 0] : [0, 0, 1]; scene(); }
  fig.querySelectorAll('[data-cloud]').forEach(b => b.addEventListener('click', () => setCloud(b.dataset.cloud)));
  fig.querySelectorAll('[data-dir]').forEach(b => b.addEventListener('click', () => setDir(b.dataset.dir)));
  document.getElementById('btn-sigreg-reset').addEventListener('click', () => { S.seed = 5; setDir('x'); setCloud('stretched'); });
  scene();
})();
</script>
