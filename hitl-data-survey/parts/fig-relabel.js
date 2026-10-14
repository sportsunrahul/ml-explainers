/* FIG.relabel: the order of the review queue (ch-cleaning). On our run C (20,000 reviews of 50,000 images, one reviewer who is
   always right), the share of flipped labels fixed after each number of reviews, for three orders: random, confident-learning
   margin (lowest normalized margin first) and active label cleaning's Phi = CE - H (highest first). Computed. */
FIG.relabel = (el, o = {}) => {
  const R = DATA.runC.review, S = { budget: 5000, show: { random: true, margin: true, phi: true } };
  const cv = PE.canvas(el, o.aspect || (w => w < 620 ? .9 : .42), draw);
  function draw(c, W, H) {
    const C = { fg: PE.css('--fg'), mu: PE.css('--muted'), blue: PE.css('--blue'), yel: PE.css('--yellow'), brown: PE.css('--brown'), green: PE.css('--green') };
    c.clearRect(0, 0, W, H); H = Math.max(140, H - (o.padBottom || 0));
    const b = { x: 48, y: 18, w: W - 64, h: H - 56 }, g = R.grid;
    const ax = PE.axes(c, b, [0, g.at(-1)], [0, 1], { xt: [0, 5000, 10000, 15000, 20000], xf: v => v.toLocaleString('en-US'), yt: [0, .25, .5, .75, 1], yf: v => Math.round(v * 100) + '%', xl: 'reviews done (one image each)' });
    const col = { random: C.brown, margin: C.blue, phi: C.green }, lab = { random: 'random order', margin: 'confident-learning margin', phi: 'Phi = CE - H' };
    Object.keys(col).forEach(k => { if (!S.show[k]) return; PE.polyline(c, g.map((x, i) => [ax.X(x), ax.Y(R.curves[k][i])]), col[k], 2.4);
      const i = g.length - 1; PE.label(c, lab[k], ax.X(g[i]) - 4, ax.Y(R.curves[k][i]) - 8, col[k], 11, 'right'); });
    const bi = g.indexOf(S.budget);
    c.strokeStyle = C.yel; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(ax.X(S.budget), b.y); c.lineTo(ax.X(S.budget), b.y + b.h); c.stroke(); c.setLineDash([]);
    PE.label(c, `budget: ${S.budget.toLocaleString('en-US')} reviews`, ax.X(S.budget) + 6, b.y + 12, C.yel, 11);
    o.say && o.say(`After ${S.budget.toLocaleString('en-US')} reviews: ${['random', 'margin', 'phi'].map(k => `${lab[k]} fixes ${(100 * R.curves[k][bi]).toFixed(1)}%`).join(', ')} of the ${DATA.runC.nErr.toLocaleString('en-US')} flipped labels.`);
  }
  PE.onTheme(() => cv.redraw());
  return { budget(v) { S.budget = Math.round(v / 500) * 500; cv.redraw(); }, reset() { S.budget = 5000; cv.redraw(); }, redraw: () => cv.redraw(), destroy() { cv.destroy && cv.destroy(); } };
};
