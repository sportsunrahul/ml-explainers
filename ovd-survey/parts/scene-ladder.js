<script>
/* SceneLadder(el, o): RF100-VL label-budget ladder for one domain (WHEN_DATA.rf, Tables 2 and 7 of 2505.20612v4).
   Columns: prompt only, 10 examples per class (trained or in-context), 10% of the labels, all labels.
   Filled yellow = text prompt, no training; hollow yellow = examples in the prompt, no training; blue = trained
   on the target labels. Diamonds = multimodal LLMs. The Grounding DINO pair (zero-shot -> 10-shot fine-tune)
   is joined by a line. o: { aspect, bottom, glow, onSay }. Returns { set(cat), reset, redraw, destroy }. */
const SceneLadder = (el, o = {}) => {
  const D = WHEN_DATA.rf, glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const COLS = ['Prompt only', '10 boxes per class', '10% of labels', 'All labels'];
  const kind = r => {
    const mllm = /Qwen|Gemini/.test(r.m);
    if (r.s === 'zero-shot') return { col: 0, fill: 'text', mllm };
    if (r.s === 'few-shot-10') return { col: 1, fill: /MQ-GLIP|Qwen|Gemini/.test(r.m) ? 'context' : 'trained', mllm };
    return { col: r.s.startsWith('semi') ? 2 : 3, fill: 'trained', mllm };
  };
  let S = { cat: D.cats.indexOf('Industrial'), u: 1, from: null }, stop = () => {};
  const val = (r, c) => r.v[c];
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80) return;
    const hh = h - bottom, narrow = w < 620;
    const b = { x: narrow ? 40 : 58, y: 30, w: w - (narrow ? 52 : 80), h: hh - (narrow ? 90 : 74) };
    const { X, Y } = PE.axes(ctx, b, [0, 4], [0, 76], { yt: [0, 10, 20, 30, 40, 50, 60, 70], yl: 'AP (RF100-VL)', ylo: narrow ? 30 : 40 });
    const cx = i => b.x + b.w * (i + .5) / 4;
    ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 12}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.fillStyle = k.fg;
    COLS.forEach((c, i) => { const words = c.split(' '), mid = Math.ceil(words.length / 2);
      if (narrow) { ctx.fillText(words.slice(0, mid).join(' '), cx(i), b.y + b.h + 16); ctx.fillText(words.slice(mid).join(' '), cx(i), b.y + b.h + 29); }
      else ctx.fillText(c, cx(i), b.y + b.h + 18); });
    ctx.restore();
    // values blend from the previous domain while switching
    const v = r => S.from == null ? val(r, S.cat) : PE.lerp(val(r, S.from), val(r, S.cat), S.u);
    const groups = [[], [], [], []];
    D.rows.forEach(r => { const q = kind(r); groups[q.col].push({ r, q, y: v(r) }); });
    const gd = {};
    groups.forEach((g, i) => {
      g.sort((a, b2) => a.y - b2.y);
      g.forEach((p, j) => {
        // spread dots sideways inside the column so equal values stay visible
        const x = cx(i) + (j - (g.length - 1) / 2) * Math.min(narrow ? 6 : 9, b.w / 4 / (g.length + 1)), y = Y(p.y);
        p.x = x; p.py = y;
        if (/GroundingDINO/.test(p.r.m)) gd[i] = p;
        const col = p.q.fill === 'trained' ? k.B : k.Y, R = p.q.mllm ? 5.5 : 5;
        ctx.save(); ctx.beginPath();
        if (p.q.mllm) { ctx.moveTo(x, y - R); ctx.lineTo(x + R, y); ctx.lineTo(x, y + R); ctx.lineTo(x - R, y); ctx.closePath(); } else ctx.arc(x, y, R, 0, 7);
        if (p.q.fill === 'context') { ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.fillStyle = k.bg; ctx.fill(); ctx.stroke(); }
        else { ctx.fillStyle = col; ctx.fill(); }
        ctx.restore();
      });
      // best in the column
      const best = g[g.length - 1];
      if (best) { ctx.save(); ctx.font = `600 ${narrow ? 10.5 : 12}px system-ui, sans-serif`; ctx.fillStyle = k.fg; ctx.textAlign = 'center';
        ctx.fillText(best.y.toFixed(1), cx(i), best.py - 11); ctx.restore(); }
    });
    // Grounding DINO: the same model before and after a 10-shot fine-tune
    if (gd[0] && gd[1]) {
      const gl = glow('gdino');
      ctx.save(); ctx.strokeStyle = k.G; ctx.lineWidth = 2 + 2 * gl; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(gd[0].x + 7, gd[0].py); ctx.lineTo(gd[1].x - 7, gd[1].py); ctx.stroke(); ctx.restore();
      ctx.save(); ctx.font = `600 ${narrow ? 10 : 11.5}px system-ui, sans-serif`; ctx.fillStyle = k.G; ctx.textAlign = 'center';
      ctx.fillText('Grounding DINO, fine-tuned', (gd[0].x + gd[1].x) / 2, Math.min(gd[0].py, gd[1].py) - (narrow ? 28 : 26)); ctx.restore();
    }
    // title: the domain
    ctx.save(); ctx.font = `600 ${narrow ? 12 : 13.5}px system-ui, sans-serif`; ctx.fillStyle = k.fg;
    ctx.fillText(D.cats[S.cat] === 'All' ? 'All 100 datasets' : `${D.cats[S.cat]} datasets`, b.x, 16); ctx.restore();
    // legend, under the column labels
    { const ly = hh - 10, items = [['text prompt', k.Y, 'f', 0], ['examples in prompt', k.Y, 'h', 0], ['trained on your labels', k.B, 'f', 0], ['multimodal LLM', k.mu, 'f', 1]];
      ctx.save(); ctx.font = `${narrow ? 10.5 : 11.5}px system-ui, sans-serif`; ctx.textBaseline = 'middle'; let x = b.x;
      items.forEach(([t, c, f, dia]) => { ctx.beginPath(); if (dia) { ctx.moveTo(x, ly - 5); ctx.lineTo(x + 5, ly); ctx.lineTo(x, ly + 5); ctx.lineTo(x - 5, ly); ctx.closePath(); } else ctx.arc(x, ly, 4.5, 0, 7);
        if (f === 'h') { ctx.strokeStyle = c; ctx.lineWidth = 1.5; ctx.stroke(); } else { ctx.fillStyle = c; ctx.fill(); }
        ctx.fillStyle = k.mu; ctx.fillText(t, x + 8, ly); x += ctx.measureText(t).width + (narrow ? 20 : 28); });
      ctx.restore(); }
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.12 : .5)), draw);
  function describe() {
    const c = S.cat, best = col => Math.max(...D.rows.filter(r => kind(r).col === col).map(r => r.v[c]));
    const name = D.cats[c] === 'All' ? 'all 100 datasets' : `the ${D.cats[c]} datasets`;
    return `RF100-VL, ${name}: best prompt-only AP ${best(0).toFixed(1)}; best with 10 boxes per class ${best(1).toFixed(1)}; best with 10% of the labels ${best(2).toFixed(1)}; best with all labels ${best(3).toFixed(1)}.`;
  }
  const say = () => o.onSay && o.onSay(describe());
  function set(cat) {
    const c = typeof cat === 'number' ? cat : D.cats.indexOf(cat); if (c < 0) return;
    stop(); S.from = S.cat === c ? null : S.cat; S.cat = c; S.u = 0;
    stop = PE.animate(700, u => { S.u = u; cv.redraw(); }, () => { S.from = null; cv.redraw(); });
    say();
  }
  function reset() { stop(); S = { cat: D.cats.indexOf('Industrial'), u: 1, from: null }; cv.redraw(); say(); }
  say();
  return { set, reset, redraw: cv.redraw, cat: () => D.cats[S.cat], destroy: () => { stop(); cv.destroy && cv.destroy(); } };
};
</script>
