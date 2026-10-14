/* ch-zeroshot: SAM minus RITM, IoU at one center point, on the 23 datasets (SAM v1, Fig. 9a, numbers printed on the bars).
   The oracle overlay (best of SAM's 3 masks) is not printed in the paper: values were read from the figure's vector file
   and are drawn only with that label. */
SCENES.zeroshot = ctx => {
  const el = mkCanvas(ctx.stage);
  const D = [['PPDLS', 'plants', 46.9, 53.2], ['BBBC038v1', 'microscopy', 44.7, 52.6], ['DOORS', 'boulders', 41.1, 50.3], ['TimberSeg', 'logs', 28.9, 37.6],
    ['NDD20', 'underwater', 21.1, 26.4], ['LVIS', 'scenes', 18.5, 25.9], ['STREETS', 'traffic camera', 17.3, 30.3], ['ZeroWaste-f', 'recycling', 9.1, 22.3],
    ['iShape', 'irregular shapes', 8.8, 22.8], ['ADE20K', 'scenes', 7.8, 19.6], ['OVIS', 'occlusions', 7.0, 22.5], ['Hypersim', 'simulation', 6.1, 16.6],
    ['NDISPark', 'parking lots', 2.7, 16.0], ['VISOR', 'egocentric', 1.8, 20.0], ['Plittersdorf', 'stereo', 1.5, 6.6], ['EgoHOS', 'egocentric', 0.8, 21.1],
    ['IBD', 'drones', -0.3, 13.4], ['WoodScape', 'fisheye driving', -0.6, 9.6], ['Cityscapes', 'driving', -2.0, 8.5], ['PIDRay', 'X-ray', -5.8, 13.9],
    ['DRAM', 'paintings', -6.5, 9.7], ['TrashCan', 'underwater', -15.0, 5.7], ['GTEA', 'egocentric', -21.4, 6.3]];
  const S = { oracle: 0, u: 1, ou: 0 };
  let cancel = () => {};
  const btns = ctx.qa('[data-zs]');
  const offs = btns.map(b => listen(b, 'click', () => set(b.dataset.zs === 'oracle' ? 1 : 0)));
  const glow = {};
  const onLink = e => { const k = e.detail; PE.animate(1100, u => { glow[k] = Math.sin(Math.PI * u); cv.redraw(); }, () => { glow[k] = 0; cv.redraw(); }); };
  ctx.article.addEventListener('pe-link', onLink);
  const cv = PE.canvas(el, S3ASPECT(ctx.stage), draw);
  function draw(c, W, H) {
    c.clearRect(0, 0, W, H);
    const narrow = W < 620, bottom = ctx.reserve || 0, top = 44, lab = narrow ? 92 : 190;
    const b = { x: lab + 8, y: top, w: W - lab - 8 - (narrow ? 14 : 40), h: H - bottom - top - 34 };
    const xr = narrow ? [-34, 58] : [-25, 55], X = v => b.x + (v - xr[0]) / (xr[1] - xr[0]) * b.w, rowH = b.h / D.length;
    const G = PE.css('--green'), R = PE.css('--red'), M = PE.css('--muted'), F = PE.css('--fg'), Br = PE.css('--brown');
    c.save(); c.strokeStyle = PE.css('--rule'); c.lineWidth = 1; c.font = '11px system-ui, sans-serif'; c.fillStyle = M; c.textAlign = 'center';
    for (let v = -20; v <= 50; v += 10) { c.beginPath(); c.moveTo(X(v), b.y); c.lineTo(X(v), b.y + b.h); c.stroke(); c.fillText((v > 0 ? '+' : '') + v, X(v), b.y + b.h + 14); }
    c.fillText('IoU of SAM minus IoU of RITM, one center point', b.x + b.w / 2, b.y + b.h + 29); c.restore();
    c.save(); c.strokeStyle = F; c.lineWidth = 1.5; c.beginPath(); c.moveTo(X(0), b.y - 4); c.lineTo(X(0), b.y + b.h); c.stroke(); c.restore();
    D.forEach(([name, kind, d, o], i) => {
      const y = b.y + i * rowH, bh = Math.max(3, rowH * .62), v = d * S.u, col = d >= 0 ? G : R;
      c.save(); c.globalAlpha = .85 + .15 * (glow.sam || 0); c.fillStyle = col; c.fillRect(Math.min(X(0), X(v)), y + (rowH - bh) / 2, Math.abs(X(v) - X(0)), bh); c.restore();
      PE.label(c, narrow ? name : `${name} (${kind})`, b.x - 6, y + rowH / 2 + 4, F, narrow ? 10 : 11, 'right', 500);
      const tx = d >= 0 ? X(v) + 4 : X(v) - 4; if (!(narrow && S.ou > .5)) PE.label(c, (d > 0 ? '+' : '') + d.toFixed(1), tx, y + rowH / 2 + 4, col, narrow ? 9 : 10, d >= 0 ? 'left' : 'right', 600);
      if (S.ou > 0) { const ox = X(PE.lerp(d, o, S.ou)); PE.ring(c, ox, y + rowH / 2, Math.min(5, rowH * .32), PE.css('--yellow'), 2, S.ou); }
    });
    if (S.ou > 0) PE.label(c, narrow ? 'rings: best of 3 masks (read from the figure, not printed)' : 'rings: best of SAM\'s 3 masks (read from the figure\'s vector file, not printed)', narrow ? 8 : b.x, top - 10, PE.css('--yellow'), narrow ? 9 : 11, 'left', 600);
    else PE.label(c, narrow ? 'bars: most confident mask (printed in Fig. 9a)' : 'bars: SAM\'s most confident mask (printed on the bars of Fig. 9a)', narrow ? 8 : b.x, top - 10, M, narrow ? 9 : 11, 'left', 500);
  }
  function readout() { ctx.hud(badge('sourced', S.oracle ? 'With the best of its three masks, SAM is above RITM on every one of the <span data-claim="sam-zs-n-datasets">23</span> datasets. Ring positions are read from the figure\'s vector file; the paper prints only the bars.' : 'With its most confident mask, SAM beats RITM on <span data-claim="sam-zs-wins">16</span> of the <span data-claim="sam-zs-n-datasets">23</span> datasets, by up to <span data-claim="sam-zs-delta-ppdls">+46.9</span> IoU (PPDLS), and loses most on GTEA (<span data-claim="sam-zs-delta-gtea">-21.4</span>) and TrashCan (<span data-claim="sam-zs-delta-trashcan">-15.0</span>).')); }
  function set(o) { S.oracle = o; press(btns, b => (b.dataset.zs === 'oracle') === !!o); readout(); cancel(); const from = S.ou; cancel = PE.animate(700, t => { S.ou = PE.lerp(from, o, t); cv.redraw(); }); }
  function grow() { cancel(); cancel = PE.animate(900, t => { S.u = t; cv.redraw(); }); }
  press(btns, b => b.dataset.zs === 'single'); readout(); grow();
  return { flat: true, replay: grow, resize: () => cv.resize(), destroy: () => { cancel(); offs.forEach(f => f()); ctx.article.removeEventListener('pe-link', onLink); cv.destroy(); } };
};
