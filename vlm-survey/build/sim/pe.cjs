
/* Shared helpers for all interactives. */
global.PE = (() => {
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const instant = () => matchMedia('(prefers-reduced-motion: reduce)').matches
    || document.documentElement.dataset.cueReplay === '1';
  const redrawers = [];
  const onTheme = fn => { redrawers.push(fn); };
  
  // Canvas sized to its box, DPR-aware; draw(ctx, w, h) gets CSS-pixel dimensions.
  function canvas(el, aspect, draw) {
    const ctx = el.getContext('2d');
    const state = { w: 0, h: 0 };
    function resize() {
      const w = el.clientWidth, h = Math.round(w * (typeof aspect === 'function' ? aspect(w) : aspect)), r = devicePixelRatio || 1;
      el.style.height = h + 'px'; el.width = Math.round(w * r); el.height = Math.round(h * r);
      ctx.setTransform(r, 0, 0, r, 0, 0); state.w = w; state.h = h; draw(ctx, w, h);
    }
    new ResizeObserver(resize).observe(el);
    onTheme(() => draw(ctx, state.w, state.h));
    return { redraw: () => draw(ctx, state.w, state.h), state, ctx };
  }
  function animate(ms, step, done) {
    if (instant()) { step(1); done && done(); return () => {}; }
    let t0 = null, id = 0, stop = false;
    const f = t => { if (stop) return; if (t0 === null) t0 = t; const u = Math.min(1, (t - t0) / ms);
      step(u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2); if (u < 1) id = requestAnimationFrame(f); else done && done(); };
    id = requestAnimationFrame(f);
    return () => { stop = true; cancelAnimationFrame(id); };
  }
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gauss(r) { return () => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r()); }
  function label(ctx, text, x, y, color, size = 12, align = 'left') {
    ctx.save(); ctx.font = `600 ${size}px system-ui, sans-serif`; ctx.fillStyle = color; ctx.textAlign = align; ctx.fillText(text, x, y); ctx.restore(); }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, u) => a + (b - a) * u;
  // Mix two "#rrggbb" colors.
  function mix(c1, c2, u) {
    const p = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
    const a = p(c1), b = p(c2); return `rgb(${a.map((v, i) => Math.round(lerp(v, b[i], u))).join(',')})`;
  }

  // World-to-canvas mapping centered on the origin; `span` world units fit the canvas height.
  function view(W, H, span) { const s = H / span; return { s, X: x => W / 2 + x * s, Y: y => H / 2 - y * s, ix: px => (px - W / 2) / s, iy: py => (H / 2 - py) / s }; }
  function dot(ctx, x, y, r, color, alpha = 1) { ctx.globalAlpha = alpha; ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.globalAlpha = 1; }
  // Pulse a named layer when an equation term with that link word is shown.
  function linkPulses(fig, redraw) {
    const glow = {};
    fig.addEventListener('pe-link', e => { const k = e.detail;
      animate(1100, u => { glow[k] = Math.sin(Math.PI * u); redraw(); }, () => { glow[k] = 0; redraw(); }); });
    return k => glow[k] || 0;
  }
  function arrow(ctx, x1, y1, x2, y2, color, lw = 2, head = 8, alpha = 1) {
    const a = Math.atan2(y2 - y1, x2 - x1), L = Math.hypot(x2 - x1, y2 - y1), hd = Math.min(head, L * .45);
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - hd * .6 * Math.cos(a), y2 - hd * .6 * Math.sin(a)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - hd * Math.cos(a - .42), y2 - hd * Math.sin(a - .42));
    ctx.lineTo(x2 - hd * Math.cos(a + .42), y2 - hd * Math.sin(a + .42)); ctx.closePath(); ctx.fill(); ctx.restore();
  }
  // Minimal chart frame: returns scale functions for box b = {x, y, w, h} and ranges; log axes when asked.
  function axes(ctx, b, xr, yr, o = {}) {
    const lx = o.logx ? Math.log10 : v => v, ly = o.logy ? Math.log10 : v => v;
    const X = v => b.x + (lx(v) - lx(xr[0])) / (lx(xr[1]) - lx(xr[0])) * b.w, Y = v => b.y + b.h - (ly(v) - ly(yr[0])) / (ly(yr[1]) - ly(yr[0])) * b.h;
    const M = css('--muted'), R = css('--rule');
    ctx.save(); ctx.strokeStyle = R; ctx.lineWidth = 1; ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = M;
    (o.xt || []).forEach(v => { ctx.beginPath(); ctx.moveTo(X(v), b.y); ctx.lineTo(X(v), b.y + b.h); ctx.stroke(); ctx.textAlign = 'center'; ctx.fillText(o.xf ? o.xf(v) : v, X(v), b.y + b.h + 14); });
    (o.yt || []).forEach(v => { ctx.beginPath(); ctx.moveTo(b.x, Y(v)); ctx.lineTo(b.x + b.w, Y(v)); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(o.yf ? o.yf(v) : v, b.x - 5, Y(v) + 4); });
    ctx.textAlign = 'center'; if (o.xl) ctx.fillText(o.xl, b.x + b.w / 2, b.y + b.h + 29);
    if (o.yl) { ctx.translate(b.x - (o.ylo || 34), b.y + b.h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(o.yl, 0, 0); }
    ctx.restore(); return { X, Y };
  }
  function polyline(ctx, pts, color, lw = 2, dash = null, alpha = 1) {
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.globalAlpha = alpha; if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); ctx.restore(); }
    // ---- Contrastive toy kit: N image and N caption embeddings as angles on the unit circle. ----
  // kind 'softmax' = CLIP's symmetric cross-entropy over cosine/tau; 'sigmoid' = SigLIP's per-pair loss with bias b.
  // Returns loss, gradients w.r.t. the angles (and b), and the similarity matrix S[i][j] = cos(th_i - ph_j).
  function contrast(th, ph, tau, kind = 'softmax', b = 0) {
    const N = th.length, gth = new Array(N).fill(0), gph = new Array(N).fill(0);
    const S = th.map(t => ph.map(p => Math.cos(t - p))), D = th.map(t => ph.map(p => -Math.sin(t - p)));
    let L = 0, gb = 0;
    const add = (i, j, g) => { gth[i] += g * D[i][j]; gph[j] -= g * D[i][j]; };
    if (kind === 'softmax') {
      for (const dir of [0, 1]) for (let i = 0; i < N; i++) {
        const lg = Array.from({ length: N }, (_, j) => (dir ? S[j][i] : S[i][j]) / tau);
        const m = Math.max(...lg), e = lg.map(x => Math.exp(x - m)), Z = e.reduce((s, x) => s + x, 0);
        L -= (lg[i] - m - Math.log(Z)) / (2 * N);
        for (let j = 0; j < N; j++) { const g = (e[j] / Z - (i === j ? 1 : 0)) / tau / (2 * N); dir ? add(j, i, g) : add(i, j, g); }
      }
    } else {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const z = i === j ? 1 : -1, x = z * (S[i][j] / tau + b);
        L += Math.log1p(Math.exp(-x)) / N; const g = -z / (1 + Math.exp(x)) / N; add(i, j, g / tau); gb += g;
      }
    }
    return { L, gth, gph, gb, S };
  }
  const softmax = v => { const m = Math.max(...v), e = v.map(x => Math.exp(x - m)), Z = e.reduce((s, x) => s + x, 0); return e.map(x => x / Z); };
  return { css, instant, onTheme, canvas, animate, rng, gauss, label, clamp, lerp, mix, view, dot, arrow, axes, polyline, linkPulses, contrast, softmax };
})();

