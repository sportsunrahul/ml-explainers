/* ch-moco sourced chart: MoCo Fig. 3, ImageNet linear top-1 (ResNet-50, IN-1M) vs number of negatives K for three
   mechanisms. Values are the figure's data labels (claims moco-fig3-*). End-to-end stops at K = 1024. */
KIT.MECHS = {
  K: [256, 512, 1024, 4096, 16384, 65536],
  e2e: { name: 'end-to-end', v: [54.9, 56.3, 57.3] },
  bank: { name: 'memory bank', v: [50.0, 52.0, 54.1, 56.5, 57.8, 58.0] },
  moco: { name: 'MoCo', v: [54.7, 56.4, 57.5, 59.0, 60.4, 60.6] },
};
KIT.mechs = (canvas, opts = {}) => {
  const D = KIT.MECHS, keys = ['e2e', 'bank', 'moco'];
  const state = { focus: null, a: { e2e: 1, bank: 1, moco: 1 }, reveal: 1 };
  let cancel = () => {};
  const col = k => k === 'bank' ? PE.css('--brown') : k === 'moco' ? PE.css('--fg') : PE.css('--muted');
  const dash = k => k === 'e2e' ? [6, 4] : null;
  const draw = (ctx, W, H) => {
    if (!W) return;
    ctx.clearRect(0, 0, W, H);
    const nar = KIT.narrow(W), M = PE.css('--muted');
    const b = { x: nar ? 44 : 58, y: 18, w: W - (nar ? 44 : 58) - (nar ? 16 : 120), h: H - 18 - 50 };
    const ax = PE.axes(ctx, b, [256, 65536], [49, 61.5], { logx: true, xt: D.K, xf: v => v >= 1024 ? (v / 1024) + 'k' : v,
      yt: [50, 52, 54, 56, 58, 60], xl: 'number of negatives K (log scale)', yl: 'linear top-1 (%)', ylo: nar ? 32 : 40 });
    keys.forEach(k => {
      const s = D[k], al = state.a[k], n = s.v.length, f = state.focus === k ? state.reveal : 1;
      const pts = s.v.map((v, i) => [ax.X(D.K[i]), ax.Y(v)]);
      // partial reveal along the line for the focused series
      const upto = f * (n - 1), last = Math.floor(upto), frac = upto - last;
      const shown = pts.slice(0, last + 1);
      if (last < n - 1) shown.push([PE.lerp(pts[last][0], pts[last + 1][0], frac), PE.lerp(pts[last][1], pts[last + 1][1], frac)]);
      PE.polyline(ctx, shown, col(k), state.focus === k ? 3 : 2, dash(k), al);
      pts.forEach((p, i) => { if (i <= upto + 1e-6) {
        PE.dot(ctx, p[0], p[1], state.focus === k ? 4.5 : 3.5, col(k), al);
        if (state.focus === k) PE.label(ctx, s.v[i].toFixed(1), p[0], p[1] + (k === 'bank' ? 17 : -9), col(k), nar ? 10 : 11, 'center');
      } });
      const e = pts[n - 1];
      if (nar) PE.label(ctx, s.name, e[0] - 4, e[1] + (k === 'moco' ? -10 : k === 'e2e' ? -10 : 16), col(k), 11, k === 'e2e' ? 'center' : 'right', 600);
      else PE.label(ctx, s.name, e[0] + 8, e[1] + (k === 'e2e' ? 16 : 4), col(k), 12, 'left', 600);
      ctx.globalAlpha = 1;
    });
    if (state.focus === 'e2e' || state.focus === null) {
      const x = ax.X(1024);
      PE.label(ctx, 'GPU memory limit', x + 6, b.y + 12, M, nar ? 10 : 11, 'left', 400);
      PE.polyline(ctx, [[x, b.y], [x, b.y + b.h]], M, 1, [2, 3], .8);
    }
  };
  const cv = PE.canvas(canvas, w => KIT.narrow(w) ? .95 : .48, draw);
  const set = focus => {
    cancel(); state.focus = focus;
    const to = {}; keys.forEach(k => { to[k] = focus === null || focus === k ? 1 : .18; });
    const from = { ...state.a };
    state.reveal = 0;
    cancel = PE.animate(1200, u => { keys.forEach(k => { state.a[k] = PE.lerp(from[k], to[k], Math.min(1, u * 2)); }); state.reveal = focus ? u : 1; cv.redraw(); });
  };
  return { set, state, redraw: cv.redraw, resize: cv.resize, destroy: () => { cancel(); cv.destroy(); } };
};
// Returns HTML: sourced numbers carry their data-claim spans.
KIT.mechsText = focus => ({
  null: 'All three mechanisms. Every curve rises with K: more negatives, better features. Only end-to-end stops early.',
  e2e: 'End-to-end: the negatives are the other images in the current batch, all encoded by the current network. Consistent, but K is tied to the batch, and K = <span data-claim="moco-e2e-maxbatch">1024</span> was the largest batch that fit.',
  bank: 'Memory bank: one stored vector per image, so K can be large. But every curve point sits below MoCo, by <span data-claim="moco-bank-gap">2.6</span> points at K = <span data-claim="moco-K">65536</span>: the stored keys are stale.',
  moco: 'MoCo: a queue of recent keys from a momentum encoder. It matches end-to-end where both exist, and keeps climbing to K = <span data-claim="moco-K">65536</span>.',
})[String(focus)];
