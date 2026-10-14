<script>
/* Toy for the sub-sentence mask: one attention layer (temperature 1, no residual) over fixed random 8-d word
   vectors. Word level lets every word attend to every word of the prompt; sub-sentence level keeps attention
   inside each category name. The figure tracks the output feature of "pallet" in two prompts. */
(() => {
  const el = document.getElementById('subsentence-canvas'), out = document.getElementById('subsentence-readout');
  const B = id => document.getElementById('btn-subsentence-' + id);
  const PROMPTS = { a: [['forklift'], ['pallet'], ['traffic', 'cone']], b: [['traffic', 'cone'], ['helmet'], ['pallet']] };
  const g = PE.gauss(PE.rng(7)), E = {}, D = 8, FOCUS = 'pallet';
  ['forklift', 'pallet', 'traffic', 'cone', 'helmet', 'ladder'].forEach(w => { const v = Array.from({ length: D }, () => g()), n = Math.hypot(...v); E[w] = v.map(x => x / n); });
  E.cone = E.cone.map((x, i) => .7 * x + .3 * E.traffic[i]);
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0), cos = (a, b) => dot(a, b) / Math.hypot(...a) / Math.hypot(...b);
  // rows attend over allowed columns; blocked cells get weight 0
  function attn(P, sub) {
    const toks = P.flat(), grp = toks.map(t => P.findIndex(gp => gp.includes(t)));
    const A = toks.map((q, i) => { const ok = toks.map((_, j) => !sub || grp[i] === grp[j]);
      const lg = toks.map(t => dot(E[q], E[t])), m = Math.max(...lg.filter((_, j) => ok[j])), e = lg.map((v, j) => ok[j] ? Math.exp(v - m) : 0), Z = e.reduce((s, v) => s + v, 0);
      return e.map(v => v / Z); });
    return { toks, grp, A };
  }
  function feat(P, sub) { const { toks, A } = attn(P, sub), i = toks.indexOf(FOCUS), o = new Array(D).fill(0);
    toks.forEach((t, j) => E[t].forEach((x, d) => { o[d] += A[i][j] * x; })); return o; }
  const S0 = () => ({ mode: 'word', prompt: 'a', u: 0, fade: 1 });
  let S = S0(), stopU = () => {}, stopF = () => {};
  const txt = (ctx, s, x, y, col, size = 11.5, weight = 500, align = 'left') => {
    ctx.save(); ctx.font = `${weight} ${size}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); ctx.restore(); };
  function hatch(ctx, x, y, w, h, col, a) {
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); ctx.globalAlpha = a * .5; ctx.strokeStyle = col; ctx.lineWidth = 1;
    for (let d = -h; d < w; d += 6) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke(); } ctx.restore();
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const narrow = w < 620, P = PROMPTS[S.prompt], W0 = attn(P, false), W1 = attn(P, true), n = W0.toks.length;
    // prompt line
    let x = 10; const y0 = narrow ? 46 : 20;
    txt(ctx, `prompt ${S.prompt.toUpperCase()}:`, x, y0, k.mu, 11.5, 600); x += 72;
    P.forEach(gp => { const s = gp.join(' '); ctx.save(); ctx.font = '600 12px system-ui, sans-serif'; const tw = ctx.measureText(s).width + 16; ctx.restore();
      DK.rr(ctx, x, y0 - 10, tw, 20, 10); ctx.strokeStyle = k.Y; ctx.lineWidth = 1.2; ctx.stroke(); txt(ctx, s, x + tw / 2, y0 + .5, k.Y, 12, 600, 'center'); x += tw + 4;
      txt(ctx, '.', x + 1, y0, k.mu, 13, 700); x += 10; });
    // matrix
    const lab = 62, top = narrow ? 96 : 72, cs = narrow ? Math.min((w - lab - 24) / n, 62) : Math.min((h - top - 26) / n, 64), mx = 10 + lab, my = top;
    txt(ctx, 'each row attends to the columns', mx, my - 27, k.mu, 10.5);
    ctx.save(); ctx.globalAlpha = S.fade;
    W0.toks.forEach((t, j) => txt(ctx, t, mx + (j + .5) * cs, my - 10, k.Y, narrow ? 10 : 11, 600, 'center'));
    W0.toks.forEach((q, i) => {
      txt(ctx, q, mx - 6, my + (i + .5) * cs, q === FOCUS ? k.fg : k.Y, narrow ? 10 : 11, 600, 'right');
      W0.toks.forEach((_, j) => {
        const a = PE.lerp(W0.A[i][j], W1.A[i][j], S.u), blocked = W0.grp[i] !== W0.grp[j], cx = mx + j * cs, cy = my + i * cs;
        ctx.fillStyle = k.p2; ctx.fillRect(cx + 1, cy + 1, cs - 2, cs - 2);
        ctx.fillStyle = k.Y; ctx.globalAlpha = S.fade * (.08 + .82 * a); ctx.fillRect(cx + 1, cy + 1, cs - 2, cs - 2); ctx.globalAlpha = S.fade;
        if (blocked && S.u > 0) hatch(ctx, cx + 1, cy + 1, cs - 2, cs - 2, k.Br, S.u);
        if (cs > 40) txt(ctx, a < .005 ? '0' : a.toFixed(2), cx + cs / 2, cy + cs / 2, a > .45 ? k.bg : k.fg, 10.5, 500, 'center');
      });
    });
    // name blocks at sub-sentence level
    if (S.u > 0) { let s0 = 0; P.forEach(gp => { ctx.save(); ctx.globalAlpha = S.fade * S.u; ctx.strokeStyle = k.Y; ctx.lineWidth = 2;
      ctx.strokeRect(mx + s0 * cs + .5, my + s0 * cs + .5, gp.length * cs - 1, gp.length * cs - 1); ctx.restore(); s0 += gp.length; }); }
    const fi = W0.toks.indexOf(FOCUS); ctx.strokeStyle = k.fg; ctx.lineWidth = 1.6; ctx.strokeRect(mx - lab + 2, my + fi * cs + .5, lab - 2 + n * cs, cs - 1);
    ctx.restore();
    if (S.u > .5) txt(ctx, 'hatched: blocked by the mask', mx, my + n * cs + 14, k.Br, 10.5, 600);
    // feature bars for "pallet" in both prompts
    const fA = PROMPTS.a, fB = PROMPTS.b, va = feat(fA, false).map((v, d) => PE.lerp(v, feat(fA, true)[d], S.u)), vb = feat(fB, false).map((v, d) => PE.lerp(v, feat(fB, true)[d], S.u));
    const bx = narrow ? 14 : mx + n * cs + 48, by = narrow ? my + n * cs + 46 : top - 10, bw = (narrow ? w - 28 : w - bx - 12), bh = narrow ? h - by - 12 : h - by - 20;
    txt(ctx, 'feature of "pallet" after attention', bx, by, k.fg, 12, 600);
    const zy = by + 22 + (bh - 52) / 2, half = (bh - 52) / 2, gw = bw / D;
    ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(bx, zy); ctx.lineTo(bx + bw, zy); ctx.stroke();
    for (let d = 0; d < D; d++) {
      const x0 = bx + d * gw + gw * .14, ww = gw * .32;
      ctx.save(); ctx.globalAlpha = S.prompt === 'a' ? 1 : .55; ctx.fillStyle = k.Y; const ha = va[d] / .8 * half; ctx.fillRect(x0, Math.min(zy, zy - ha), ww, Math.abs(ha)); ctx.restore();
      ctx.save(); ctx.globalAlpha = S.prompt === 'b' ? 1 : .55; ctx.strokeStyle = k.Y; ctx.lineWidth = 1.5; const hb = vb[d] / .8 * half; ctx.strokeRect(x0 + ww + 3, Math.min(zy, zy - hb), ww, Math.max(1, Math.abs(hb))); ctx.restore();
    }
    const c = cos(va, vb), ly = by + bh - 16;
    ctx.fillStyle = k.Y; ctx.fillRect(bx, ly - 5, 11, 10); txt(ctx, 'prompt A', bx + 16, ly, k.fg, 11);
    ctx.strokeStyle = k.Y; ctx.lineWidth = 1.5; ctx.strokeRect(bx + 88, ly - 5, 11, 10); txt(ctx, 'prompt B', bx + 104, ly, k.fg, 11);
    txt(ctx, `cosine(A, B) = ${c.toFixed(2)}`, bx + bw, ly, c > .995 ? k.G : k.fg, 12, 700, 'right');
  }
  const cv = PE.canvas(el, w => (w < 620 ? Math.max(640, w * 1.7) / w : .44), draw);
  function describe() {
    const P = PROMPTS[S.prompt], { toks, A } = attn(P, S.mode === 'sub'), i = toks.indexOf(FOCUS), self = A[i][i];
    const c = cos(feat(PROMPTS.a, S.mode === 'sub'), feat(PROMPTS.b, S.mode === 'sub'));
    return S.mode === 'word'
      ? `Word level, prompt ${S.prompt.toUpperCase()}: "pallet" puts ${Math.round(self * 100)}% of its attention on itself and ${Math.round((1 - self) * 100)}% on the other names. Its feature has cosine ${c.toFixed(2)} with its feature in the other prompt.`
      : `Sub-sentence level, prompt ${S.prompt.toUpperCase()}: "pallet" attends only to its own name. Its feature is the same in both prompts (cosine ${c.toFixed(2)}).`;
  }
  const say = () => { out.textContent = describe(); };
  const press = () => { B('word').setAttribute('aria-pressed', String(S.mode === 'word')); B('sub').setAttribute('aria-pressed', String(S.mode === 'sub'));
    B('a').setAttribute('aria-pressed', String(S.prompt === 'a')); B('b').setAttribute('aria-pressed', String(S.prompt === 'b')); };
  function setMode(m) { if (S.mode === m) { say(); return; } S.mode = m; const from = S.u, to = m === 'sub' ? 1 : 0; stopU();
    stopU = PE.animate(700, u => { S.u = PE.lerp(from, to, u); cv.redraw(); }); press(); say(); }
  function setPrompt(p) { if (S.prompt === p) { say(); return; } S.prompt = p; stopF(); stopF = PE.animate(500, u => { S.fade = u; cv.redraw(); }); press(); say(); }
  B('word').addEventListener('click', () => setMode('word'));
  B('sub').addEventListener('click', () => setMode('sub'));
  B('a').addEventListener('click', () => setPrompt('a'));
  B('b').addEventListener('click', () => setPrompt('b'));
  B('reset').addEventListener('click', () => { stopU(); stopF(); S = S0(); cv.redraw(); press(); say(); });
  press(); say();
})();
</script>
