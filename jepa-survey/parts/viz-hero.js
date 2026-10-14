<script>
(() => {
  // Hero: the whole page in one motion. Patch embeddings rise as blue points; a block turns yellow and is hidden;
  // the predictor's guess (green) lands near where the yellow points were; the red miss shrinks; the scene settles.
  const el = document.getElementById('hero-canvas'), btn = document.getElementById('btn-hero-replay');
  const G = 10, pts0 = [], tgt = [];
  const r = PE.rng(17), gs = PE.gauss(r);
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const hid = i >= 3 && i < 6 && j >= 5 && j < 8; const p = [(j - 4.5) * .3 + gs() * .05, -1.6, (i - 4.5) * .3 + gs() * .05]; const e = [p[0] * 1.6 + gs() * .25, .4 + gs() * .5 + (hid ? .3 : 0), p[2] * 1.6 + gs() * .25]; pts0.push({ p, e, hid }); }
  const hidden = pts0.filter(q => q.hid), cen = hidden.reduce((s, q) => [s[0] + q.e[0] / hidden.length, s[1] + q.e[1] / hidden.length, s[2] + q.e[2] / hidden.length], [0, 0, 0]);
  const guess0 = [cen[0] - 1.1, cen[1] - .5, cen[2] + .9];
  const s3 = PE3.scene(el, { aspect: w => w < 560 ? .9 : .46, fov: 34, dist: 7.6, yaw: .5, pitch: .32, autoRotate: 0 });
  let t0 = 0, cancel = () => {};
  const sm = (a, b, u) => PE.clamp((u - a) / (b - a), 0, 1);
  function frame(u) {
    // phases: 0-.3 rise, .3-.5 hide, .5-.75 guess lands, .75-1 miss shrinks
    const rise = sm(0, .3, u), hide = sm(.3, .5, u), land = sm(.5, .75, u), fix = sm(.75, 1, u);
    const items = [{ type: 'grid', y: -1.6, size: 3.2, step: .3, color: '--rule' }];
    const vis = [], yel = [];
    pts0.forEach(q => { const p = [q.p[0] + (q.e[0] - q.p[0]) * rise, q.p[1] + (q.e[1] - q.p[1]) * rise, q.p[2] + (q.e[2] - q.p[2]) * rise]; (q.hid ? yel : vis).push(p); });
    items.push({ type: 'points', pts: vis, color: '--blue', size: 5, alpha: .9, link: 'context' });
    items.push({ type: 'points', pts: yel, color: hide > 0 ? '--yellow' : '--blue', size: 5, alpha: hide > 0 ? .35 + .55 * hide : .9, link: 'target' });
    if (hide > 0) items.push({ type: 'label', at: cen, text: 'hidden targets', color: '--yellow', dy: -14, alpha: hide });
    if (rise >= 1) { const cv = vis.reduce((s, p) => [s[0] + p[0] / vis.length, s[1] + p[1] / vis.length, s[2] + p[2] / vis.length], [0, 0, 0]); items.push({ type: 'label', at: cv, text: 'context embeddings', color: '--blue', dy: 16, alpha: rise }); }
    if (land > 0) { const g = [guess0[0] + (cen[0] - guess0[0]) * fix, guess0[1] + (cen[1] - guess0[1]) * fix, guess0[2] + (cen[2] - guess0[2]) * fix];
      const start = [-1.2, .3, -.6]; const tip = [start[0] + (g[0] - start[0]) * land, start[1] + (g[1] - start[1]) * land, start[2] + (g[2] - start[2]) * land];
      items.push({ type: 'line', pts: [start, tip], color: '--green', width: 3, arrow: true, link: 'predictor' }); items.push({ type: 'points', pts: [tip], color: '--green', size: 8 });
      if (land >= 1) { items.push({ type: 'line', pts: [g, cen], color: '--red', width: 2.5, dash: [4, 3], link: 'loss' }); items.push({ type: 'label', at: [(g[0] + cen[0]) / 2, (g[1] + cen[1]) / 2, (g[2] + cen[2]) / 2], text: fix < 1 ? 'the miss' : 'the miss, learned away', color: '--red', dy: -10, dx: 8 }); items.push({ type: 'label', at: tip, text: 'predicted', color: '--green', dy: 16 }); }
    }
    s3.items = items; s3.redraw();
  }
  function play() { cancel(); if (PE.instant()) { frame(1); return; } cancel = s3.tween(6000, u => frame(u)); }
  btn.addEventListener('click', play);
  frame(PE.instant() ? 1 : 0); play();
})();
</script>
