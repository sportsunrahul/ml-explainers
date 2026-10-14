<script>
/* GLIP toy, shared by the long page and the atlas: detection as phrase grounding. Four boxes from the
   warehouse scene are scored against the prompt "person. traffic cone. toothbrush". Phrase view: one
   column per phrase (the rows of W in chapter 1). Token view: one column per token, including the tokens
   GLIP adds ([CLS], ".", [SEP], [NoObj]); the subword split of "toothbrush" is the paper's own example.
   A selected phrase marks its tokens as positive targets T' for the matched box. Logits are hand-set
   toy values; a phrase score is the mean of its token probabilities (GLIP Sec. 3.1).
   o: { aspect(w) -> h/w, glow(word) -> 0..1, onSay(text), onPick(state), bottom (px kept free) } */
const SceneGlipGround = (el, o = {}) => {
  const glow = o.glow || (() => 0), bottom = o.bottom || 0;
  const REG = [1, 2, 3, 4].map(i => DK.WAREHOUSE[i]);
  const RN = ['person', 'pallet', 'forklift', 'cone'];
  const PH = [{ key: 'person', name: 'person', reg: 0 }, { key: 'cone', name: 'traffic cone', reg: 3 }, { key: 'tooth', name: 'toothbrush', reg: -1 }];
  const TOK = [['[CLS]', -1], ['person', 0], ['.', -1], ['traffic', 1], ['cone', 1], ['.', -1], ['tooth#', 2], ['#brush', 2], ['[SEP]', -1], ['[NoObj]', -1]];
  const LOGIT = [
    [-4.0, 2.6, -4.5, -3.0, -3.2, -4.5, -3.5, -3.8, -4.2, -3.6],
    [-4.2, -2.8, -4.5, -2.4, -1.9, -4.5, -3.6, -3.2, -4.4, -3.0],
    [-4.0, -1.8, -4.5, -2.2, -2.6, -4.5, -3.9, -3.4, -4.3, -2.9],
    [-4.1, -3.0, -4.5, 2.1, 2.9, -4.5, -3.2, -2.9, -4.4, -3.5]];
  const PROB = LOGIT.map(r => r.map(x => 1 / (1 + Math.exp(-x))));
  const toksOf = k => TOK.map((t, j) => j).filter(j => TOK[j][1] === k);
  const phraseScore = (r, k) => { const js = toksOf(k); return js.reduce((s, j) => s + PROB[r][j], 0) / js.length; };
  const isPos = (r, j) => TOK[j][1] >= 0 && PH[TOK[j][1]].reg === r;
  const bestPhrase = r => { let b = -1, v = 0; PH.forEach((p, k) => { const s = phraseScore(r, k); if (s > v) { v = s; b = k; } }); return [b, v]; };
  const selK = s => PH.findIndex(p => p.key === s);
  const inSel = (j, s) => s === 'all' || (selK(s) >= 0 && TOK[j][1] === selK(s));
  const S = { view: 'phrase', uView: 0, sel: null, uSel: 0 };
  let stop = () => {};

  function geo(w, h) {
    const hh = h - bottom, narrow = w < 620;
    let sx, sy, sw, sh, mx, my, mw, rh, labW;
    if (narrow) {
      sx = 8; sw = w - 16; sh = Math.min(sw * .42, hh * .36); sy = 6;
      labW = 64; mx = 8 + labW; mw = w - mx - 18; my = sy + sh + 92;
      rh = Math.min(40, (hh - my - 30) / 4);
    } else {
      sw = Math.min(w * .36, (hh - 40) * 1.5); sh = sw * .72; sx = 12;
      labW = 92; mx = sx + sw + 22 + labW; mw = w - mx - 12;
      rh = Math.min(54, (hh - 150) / 4); my = Math.max(108, (hh - 4 * rh) / 2 + 30);
      sy = my + 2 * rh - sh / 2;
    }
    return { hh, narrow, sx, sy, sw, sh, mx, my, mw, rh, labW };
  }
  // Column rectangles for every token, morphing from phrase columns (u = 0) to token columns (u = 1).
  function cols(mx, mw, u) {
    const tw = mw / TOK.length, pw = mw / PH.length;
    return TOK.map(([, k], j) => {
      let xp, wp;
      if (k >= 0) { const js = toksOf(k), q = js.indexOf(j); wp = pw / js.length; xp = mx + k * pw + q * wp; }
      else { const nx = TOK.findIndex((t, i) => i > j && t[1] >= 0); xp = nx < 0 ? mx + mw : mx + TOK[nx][1] * pw; wp = 0; }
      return { x: PE.lerp(xp, mx + j * tw, u), w: PE.lerp(wp, tw, u) };
    });
  }
  function draw(ctx, w, h) {
    const k = DK.C(); ctx.clearRect(0, 0, w, h); if (w < 80 || h < 80) return;
    const g = geo(w, h), { narrow, sx, sy, sw, sh, mx, my, mw, rh } = g, u = S.uView;
    const gR = glow('regions'), gT = glow('tokens'), gS = glow('scores'), gG = glow('targets'), gL = glow('loss'), gP = glow('phrase');
    const fs = narrow ? 10.5 : 12;
    // scene and detections
    DK.drawScene(ctx, DK.WAREHOUSE, sx, sy, sw, sh, k);
    const sk = selK(S.sel);
    REG.forEach((ob, r) => {
      const [x, y, bw, bh] = DK.boxOf(ob, sx, sy, sw, sh), [b, v] = bestPhrase(r), hit = v >= .5;
      const focus = sk >= 0 ? (PH[sk].reg === r ? 1 : .35) : 1, a = PE.lerp(1, focus, S.uSel);
      DK.brackets(ctx, x, y, bw, bh, hit ? k.G : k.Br, { lw: 2 + 1.5 * gR + (sk >= 0 && PH[sk].reg === r ? S.uSel : 0), alpha: a, dash: hit ? null : [4, 3] });
      if (hit) DK.tag(ctx, r === 3 ? x + bw : x, y, PH[b].name, v.toFixed(2), k.G, k, { size: narrow ? 10 : 11.5, alpha: a, align: r === 3 ? 'right' : 'left', maxX: sx + sw + (narrow ? 0 : 14) });
    });
    // matrix header: title, phrase chips, token labels
    const C = cols(mx, mw, u);
    ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textBaseline = 'alphabetic';
    ctx.fillText(narrow ? 'S_ground: every box x every token' : 'S_ground: every box scored against every token', narrow ? 8 : mx - g.labW, my - (narrow ? 78 : 86));
    ctx.restore();
    const chipY = my - (narrow ? 66 : 70);
    PH.forEach((p, kk) => {
      const js = toksOf(kk), x0 = C[js[0]].x, x1 = C[js[js.length - 1]].x + C[js[js.length - 1]].w, on = S.sel === p.key || S.sel === 'all';
      ctx.save(); ctx.font = `600 ${fs}px system-ui, sans-serif`;
      const tw = ctx.measureText(p.name).width, cw = Math.max(tw + 14, Math.min(x1 - x0 - 4, tw + 30)), cx = (x0 + x1) / 2;
      DK.rr(ctx, cx - cw / 2, chipY, cw, 20, 10);
      ctx.fillStyle = on ? DK.mixh(k.bg, k.Y, .22 * (S.sel === 'all' ? 1 : S.uSel)) : k.bg; ctx.fill();
      ctx.strokeStyle = k.Y; ctx.lineWidth = 1.2 + 1.4 * gP; ctx.stroke();
      ctx.fillStyle = k.Y; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(p.name, cx, chipY + 10.5);
      p.hit = [cx - cw / 2, chipY, cw, 20];
      if (u > .02 && js.length > 1) { ctx.strokeStyle = DK.mixh(k.bg, k.Y, .6); ctx.lineWidth = 1; ctx.globalAlpha = u;
        ctx.beginPath(); ctx.moveTo(x0 + 3, chipY + 24); ctx.lineTo(x1 - 3, chipY + 24); ctx.stroke(); }
      ctx.restore();
    });
    const tcw = mw / TOK.length, rot = tcw < 50;
    TOK.forEach(([t, kk], j) => {
      if (u < .02) return;
      const c = C[j], cx = c.x + c.w / 2, y = my - 8, sel = S.sel && inSel(j, S.sel);
      ctx.save(); ctx.globalAlpha = u; ctx.font = `${sel || gT > .05 ? 700 : 500} ${narrow ? 10 : 11}px system-ui, sans-serif`;
      ctx.fillStyle = kk >= 0 ? k.Y : k.Br; ctx.textBaseline = 'middle';
      if (rot) { ctx.translate(cx + 3, y); ctx.rotate(narrow ? -1.2 : -.9); ctx.textAlign = 'left'; ctx.fillText(t, 0, 0); }
      else { ctx.textAlign = 'center'; ctx.fillText(t, cx, y - 2); }
      ctx.restore();
      if (gT > .02) { ctx.save(); ctx.globalAlpha = gT * u; ctx.strokeStyle = k.Y; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(c.x + 3, my - 2); ctx.lineTo(c.x + c.w - 3, my - 2); ctx.stroke(); ctx.restore(); }
    });
    // rows: labels and cells; text goes in a second pass so a neighbouring cell never covers it
    const late = [];
    RN.forEach((n, r) => {
      const y = my + r * rh, hi = sk >= 0 && PH[sk].reg === r;
      ctx.save(); ctx.font = `${hi || gR > .05 ? 700 : 600} ${fs}px system-ui, sans-serif`; ctx.fillStyle = k.B; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(narrow ? n : n + ' box', mx - 8, y + rh / 2); ctx.restore();
      TOK.forEach(([, kk], j) => {
        const c = C[j]; if (c.w < .6) return;
        const js = kk >= 0 ? toksOf(kk) : [j], first = js[0] === j, last = js.at(-1) === j;
        const x0 = c.x + (first ? 1.5 : 1.5 * u), x1 = c.x + c.w - (last ? 1.5 : 1.5 * u - .8 * (1 - u));
        const v = kk >= 0 ? PE.lerp(phraseScore(r, kk), PROB[r][j], u) : PROB[r][j];
        ctx.fillStyle = DK.mixh(k.bg, k.fg, .07 + .62 * v); ctx.fillRect(x0, y + 1.5, x1 - x0, rh - 3);
        if (gS > .02) { ctx.save(); ctx.globalAlpha = gS; ctx.strokeStyle = k.fg; ctx.lineWidth = 1.5; ctx.strokeRect(x0, y + 1.5, x1 - x0, rh - 3); ctx.restore(); }
        // value: one per phrase in phrase view (centred on the phrase), one per token in token view
        const spanX0 = C[js[0]].x, spanX1 = C[js.at(-1)].x + C[js.at(-1)].w, merged = u <= .5;
        if (merged ? first && spanX1 - spanX0 >= 34 : x1 - x0 >= 34)
          late.push(() => { ctx.save(); ctx.font = `500 ${narrow ? 10 : 11}px system-ui, sans-serif`; ctx.fillStyle = v > .5 ? k.bg : k.mu; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(v.toFixed(2), merged ? (spanX0 + spanX1) / 2 : (x0 + x1) / 2, y + rh / 2 + .5); ctx.restore(); });
        // targets: the selected phrase's columns (or all of T'), plus a pulse of the whole T' on the equation link
        const aSel = S.sel && inSel(j, S.sel) ? S.uSel : 0, a = Math.max(aSel, gG * u);
        if (a > .01) late.push(() => {
          ctx.save(); ctx.globalAlpha = a; const pos = isPos(r, j), bw = narrow ? 11 : 13, bh = narrow ? 12 : 13;
          if (pos) { ctx.strokeStyle = k.G; ctx.lineWidth = 2.6; ctx.strokeRect(x0 + 1.3, y + 2.8, x1 - x0 - 2.6, rh - 5.6);
            DK.rr(ctx, x0 + 3, y + 4, bw, bh, 3); ctx.fillStyle = k.G; ctx.fill(); }
          ctx.font = `700 ${narrow ? 10 : 10.5}px system-ui, sans-serif`; ctx.fillStyle = pos ? k.bg : k.Br; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(pos ? '1' : '0', x0 + 3 + bw / 2, y + 4 + bh / 2 + .5); ctx.restore();
        });
      });
    });
    late.forEach(f => f());
    if (gL > .02) { ctx.save(); ctx.globalAlpha = gL; ctx.strokeStyle = k.R; ctx.lineWidth = 2.5; DK.rr(ctx, mx - 4, my - 4, mw + 8, 4 * rh + 8, 6); ctx.stroke(); ctx.restore(); }
    // legend
    ctx.save(); ctx.font = `500 ${narrow ? 10 : 11}px system-ui, sans-serif`; ctx.fillStyle = k.mu; ctx.textBaseline = 'top';
    const ly = my + 4 * rh + 8;
    const leg = narrow ? ['darker cell = higher score', 'green 1 = positive target, brown 0 = negative'] : ['darker cell = higher score (sigmoid of O P^T)', 'green 1 = positive target in T\', brown 0 = negative'];
    if (narrow) leg.forEach((s, i) => ctx.fillText(s, 8, ly + i * 13)); else ctx.fillText(leg.join(';  '), mx - g.labW, ly);
    ctx.restore();
  }
  const cv = PE.canvas(el, o.aspect || (w => (w < 620 ? 1.18 : .46)), draw);
  function describe() {
    const p = r => phraseScore(r, 0), c = r => phraseScore(r, 1), t1 = PROB[3][3], t2 = PROB[3][4];
    const npos = RN.reduce((s, n, r) => s + TOK.filter((t, j) => isPos(r, j)).length, 0);
    if (S.sel === 'person') return `"person" is one token. It is a positive target for the person box (score ${PROB[0][1].toFixed(2)}) and a negative target for the other three boxes.`;
    if (S.sel === 'cone') return `"traffic cone" is two tokens, and both are positive targets for the cone box. At inference the phrase score is their mean: (${t1.toFixed(2)} + ${t2.toFixed(2)}) / 2 = ${c(3).toFixed(2)}.`;
    if (S.sel === 'tooth') return 'No box shows a toothbrush, so both of its subword tokens are negative targets for every box. A class name in the prompt with nothing in the image is a free negative example.';
    if (S.sel === 'all') return `The full target T': ${npos} positive cells out of ${RN.length * TOK.length}. The added tokens are never positive, and the pallet and forklift rows are all zero: for this prompt, those boxes are background.`;
    if (S.view === 'token') return `Token view: the prompt is ${TOK.length} tokens, not ${PH.length} phrases. "traffic cone" has two words, "toothbrush" two subwords, and [CLS], the separators, [SEP] and [NoObj] are added tokens. Every box gets a score for every token.`;
    return `Phrase view: one column per phrase, like the rows of W in chapter 1. The person box scores ${p(0).toFixed(2)} on "person" and the cone box ${c(3).toFixed(2)} on "traffic cone". The pallet and the forklift match no phrase.`;
  }
  const say = () => o.onSay && o.onSay(describe());
  function tw(to, ms, done) {
    stop(); const from = { uView: S.uView, uSel: S.uSel };
    stop = PE.animate(ms, e => { for (const key in to) S[key] = PE.lerp(from[key], to[key], e); cv.redraw(); }, done);
  }
  // Each state is set, never toggled: setting it twice equals setting it once.
  function set(state) {
    if (state === 'phrases') { S.view = 'phrase'; S.sel = null; tw({ uView: 0, uSel: 0 }, 750); }
    else if (state === 'tokens') { S.view = 'token'; S.sel = null; tw({ uView: 1, uSel: 0 }, 750); }
    else if (S.sel !== state) { const need = S.uView < 1; S.view = 'token'; S.sel = state; S.uSel = 0; tw({ uView: 1, uSel: 1 }, need ? 900 : 650); }
    say();
  }
  const state = () => S.sel || (S.view === 'token' ? 'tokens' : 'phrases');
  function reset() { stop(); Object.assign(S, { view: 'phrase', uView: 0, sel: null, uSel: 0 }); cv.redraw(); say(); }
  const onClick = e => {
    const r = el.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    const p = PH.find(q => q.hit && x >= q.hit[0] && x <= q.hit[0] + q.hit[2] && y >= q.hit[1] - 4 && y <= q.hit[1] + q.hit[3] + 4);
    if (p) { set(p.key); o.onPick && o.onPick(p.key); }
  };
  el.addEventListener('click', onClick);
  say();
  return { set, reset, state, redraw: cv.redraw, resize: () => cv.resize && cv.resize(),
    destroy: () => { stop(); el.removeEventListener('click', onClick); cv.destroy && cv.destroy(); } };
};
</script>
