/* CoreLti: selective copying as a toy. Four letters sit among filler tokens; a state of four slots must hold them in order.
   Both lanes run the same update with a gate g in [0, 1]: each slot keeps (1 - g) of itself and takes g of the slot before
   it, and slot 1 takes g of the token. Filler is embedded as zero in both lanes, so the only difference is the gate:
   the time-invariant lane uses one fixed g for every token (slot 1 then forgets at the fixed rate 1 - g); the selective
   lane sets g from the token itself (1 for a letter, 0 for filler). The read-out takes the strongest letter in each slot. */
const CoreLti = (() => {
  const T = 24, K = 4, NS = 4, NL = 8, LET = 'ABCDEFGH';
  function sequence(mode, seed) {
    const r = PE.rng(seed * 2654435761 >>> 0), pool = [...Array(NL).keys()], letters = [];
    for (let i = 0; i < K; i++) letters.push(pool.splice(Math.floor(r() * pool.length), 1)[0]);
    let pos;
    if (mode === 'even') pos = [5, 11, 17, 23];
    else { const p = [...Array(T).keys()]; pos = []; for (let i = 0; i < K; i++) pos.push(p.splice(Math.floor(r() * p.length), 1)[0]); pos.sort((a, b) => a - b); }
    const toks = Array(T).fill(-1); pos.forEach((q, i) => { toks[q] = letters[i]; });
    return { pos, letters, toks };
  }
  function run(toks, gate) {
    let h = Array.from({ length: NS }, () => Array(NL).fill(0)); const hs = [h];
    toks.forEach(tk => { const g = gate(tk), prev = h;
      h = prev.map((row, k) => row.map((v, c) => (1 - g) * v + g * (k === 0 ? (tk === c ? 1 : 0) : prev[k - 1][c]))); hs.push(h); });
    return hs;
  }
  function decode(h) {
    return h.map(row => { let m = 0, j = -1, s = 0; row.forEach((v, c) => { s += v; if (v > m) { m = v; j = c; } });
      return { letter: m > 1e-6 ? j : -1, share: s > 1e-9 ? m / s : 0, total: s }; });
  }
  // Weight of a token that arrived k steps before the end in slot n (0 = newest), fixed step p: p C(k, n) p^n (1 - p)^(k - n).
  function kernel(p, k, n) { if (k < n) return 0; let c = 1; for (let i = 0; i < n; i++) c = c * (k - i) / (i + 1); return p * c * Math.pow(p, n) * Math.pow(1 - p, k - n); }
  const ltiGate = p => () => p, selGate = tk => (tk >= 0 ? 1 : 0);
  // Slots are read oldest first: slot 4 holds the first letter if all went well.
  const readOut = h => decode(h).slice().reverse();
  const score = (dec, letters) => dec.filter((d, i) => d.letter === letters[i]).length;
  const tallies = {};
  function tally(p, n = 200) {
    const key = p.toFixed(3); if (tallies[key]) return tallies[key];
    let all = 0, sum = 0;
    for (let s = 1; s <= n; s++) { const q = sequence('random', s), c = score(readOut(run(q.toks, ltiGate(p)).at(-1)), q.letters); sum += c; if (c === K) all++; }
    return (tallies[key] = { all, mean: sum / n, n });
  }
  function result(S) {
    const q = sequence(S.mode, S.seed), lh = run(q.toks, ltiGate(S.p)), sh = run(q.toks, selGate);
    const ld = readOut(lh.at(-1)), sd = readOut(sh.at(-1));
    return { q, lh, sh, ld, sd, lc: score(ld, q.letters), sc: score(sd, q.letters) };
  }
  // State at fractional time tau in [0, T] (for the sweep).
  const at = (hs, tau) => { const i = Math.floor(tau), f = tau - i; if (i >= T) return hs[T]; return hs[i].map((row, k) => row.map((v, c) => PE.lerp(v, hs[i + 1][k][c], f))); };

  function rr(ctx, x, y, w, h, r, fill, stroke, lw = 1.3, alpha = 1) {
    ctx.save(); ctx.globalAlpha = alpha; ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } ctx.restore();
  }
  function layout(W, H, bottom) {
    const narrow = W < 520;
    if (!narrow) {
      const xL = 16, xR = W * .63, slot = (xR - xL) / T, rx = W * .69, cw = Math.min(46, (W - 16 - rx) / 4 - 8);
      return { narrow, xL, xR, slot, tok: Math.min(slot - 4, 24), ty: 46, lanes: [{ title: 86, y0: 100, y1: 172, ry: 100, oy: 152 }, { title: 214, y0: 232, y1: 300, ry: 228, oy: 280 }], rx, cw, ch: 40, gap: 8 };
    }
    const xL = 8, xR = W - 8, slot = (xR - xL) / T, cw = Math.min(34, (W - 24) / 8 - 6);
    return { narrow, xL, xR, slot, tok: Math.min(slot - 2, 16), ty: 36, lanes: [{ title: 74, y0: 92, y1: 140, ry: 150, oy: 150 }, { title: 214, y0: 226, y1: 258, ry: 268, oy: 268 }], rx: xL, cw, ch: 30, gap: 6 };
  }
  // S = { mode, seed, p, revealed, u }
  function draw(ctx, W, H, S, glow = () => 0) {
    const C = { bg: PE.css('--bg'), fg: PE.css('--fg'), mu: PE.css('--muted'), rule: PE.css('--rule'), B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), R: PE.css('--red') };
    ctx.clearRect(0, 0, W, H);
    const G = layout(W, H, S.bottom || 0), R = result(S), q = R.q, narrow = G.narrow;
    const u = S.revealed ? (S.u == null ? 1 : S.u) : 0, tau = u * T, fin = S.revealed && u >= 1;
    const gK = glow('kernel'), gG = glow('gate'), gT = glow('tokens'), gS = glow('state'), gW = glow('weight');
    const tx = t => G.xL + G.slot * (t + .5);
    // Token row.
    PE.label(ctx, narrow ? 'input: 4 letters among filler' : `input x: ${K} letters among ${T - K} filler tokens`, G.xL, G.ty - 16, C.fg, narrow ? 11 : 12);
    for (let t = 0; t < T; t++) {
      const x = tx(t), s = G.tok, y = G.ty, d = q.toks[t] >= 0, passed = !S.revealed || t < tau;
      const al = S.revealed && !passed ? .45 : 1;
      if (d) { rr(ctx, x - s / 2, y, s, s, 4, C.B, gT > 0 ? C.fg : null, 1 + 2 * gT, al); ctx.save(); ctx.globalAlpha = al; PE.label(ctx, LET[q.toks[t]], x, y + s / 2 + (narrow ? 3.5 : 4.5), C.bg, narrow ? 10 : 13, 'center'); ctx.restore(); }
      else { rr(ctx, x - s / 2 + 2, y + 2, s - 4, s - 4, 3, null, C.mu, 1 + gT, al * .8); PE.dot(ctx, x, y + s / 2, 1.6, C.mu, al * .8); }
    }
    if (S.revealed && u < 1) { const x = G.xL + G.slot * tau; PE.polyline(ctx, [[x, G.ty - 4], [x, G.lanes[1].y1]], C.fg, 1.2, [3, 3], .6); }
    // Faint guides from each letter down through both lanes.
    q.pos.forEach(t => PE.polyline(ctx, [[tx(t), G.ty + G.tok + 2], [tx(t), G.lanes[1].y1]], C.B, 1, [2, 4], .35));
    const lanes = [
      { name: narrow ? `fixed decay (time-invariant), step ${S.p.toFixed(2)}` : `fixed decay (time-invariant): every token moves the state by the same step p = ${S.p.toFixed(2)}`, hs: R.lh, dec: R.ld, ok: R.lc },
      { name: narrow ? 'gate set by each token (selective)' : 'gate set by each token (selective): a letter moves the state one slot, filler leaves it alone', hs: R.sh, dec: R.sd, ok: R.sc },
    ];
    lanes.forEach((ln, li) => {
      const g = G.lanes[li];
      PE.label(ctx, ln.name, G.xL, g.title, C.fg, narrow ? 10.5 : 11.5);
      if (li === 0) {
        // The fixed kernel of each slot over position: it depends on the lag only, never on what sits there.
        const pts = [], mx = { v: 1e-9 };
        for (let n = 0; n < NS; n++) { const row = []; for (let t = 0; t < T; t++) { const v = kernel(S.p, T - 1 - t, n); row.push(v); mx.v = Math.max(mx.v, v); } pts.push(row); }
        for (let n = 0; n < NS; n++) {
          const ys = pts[n].map(v => g.y1 - (g.y1 - g.y0 - 6) * v / mx.v), al = [1, .78, .6, .45][n];
          PE.polyline(ctx, ys.map((y, t) => [tx(t), y]), C.Y, 1.6 + 2.2 * Math.max(gK, gW), null, al);
          let im = 0; pts[n].forEach((v, t) => { if (v > pts[n][im]) im = t; });
          PE.label(ctx, `slot ${n + 1}`, PE.clamp(tx(im), G.xL + 16, G.xR - 16), ys[im] - 5, C.Y, narrow ? 8.5 : 9.5, 'center');
        }
        PE.polyline(ctx, [[G.xL, g.y1], [G.xR, g.y1]], C.rule, 1);
        if (!narrow) PE.label(ctx, 'weight each slot gives each position', G.xR, g.y1 + 13, C.mu, 9.5, 'right');
      } else {
        // The gate, read from each token.
        for (let t = 0; t < T; t++) {
          const v = q.toks[t] >= 0 ? 1 : 0, x = tx(t), hh = (g.y1 - g.y0 - 14) * v;
          rr(ctx, x - Math.min(5, G.slot * .25), g.y1 - Math.max(2, hh), Math.min(10, G.slot * .5), Math.max(2, hh), 2, PE.mix(C.bg, C.fg, .35 + .4 * gG), null);
        }
        PE.polyline(ctx, [[G.xL, g.y1], [G.xR, g.y1]], C.rule, 1);
        if (!narrow) PE.label(ctx, S.revealed ? 'gate g = 1 on a letter, 0 on filler; boxed: the slot each letter ends in' : 'gate g = 1 on a letter, 0 on filler', G.xR, g.y1 + 13, C.mu, 9.5, 'right');
      }
      // The register (yellow, oldest slot first) and the read-out (green, or red where wrong).
      const cx = i => narrow ? G.xL + i * (G.cw + G.gap) : G.rx + i * (G.cw + G.gap), oy = narrow ? g.ry : g.oy, ox = i => narrow ? W / 2 + 8 + i * (G.cw + G.gap) : cx(i);
      if (!narrow) PE.label(ctx, 'state: 4 slots, oldest first', G.rx, g.ry - 6, C.Y, 10);
      const h = S.revealed ? decode(at(ln.hs, tau)).slice().reverse() : null, tot = S.revealed ? Math.max(1e-9, ...decode(ln.hs[T]).map(d => d.total)) : 1;
      for (let i = 0; i < NS; i++) {
        const x = cx(i), d = h && h[i];
        rr(ctx, x, g.ry, G.cw, G.ch, 5, d ? PE.mix(C.bg, C.Y, .12 + .3 * Math.min(1, d.total / tot)) : PE.mix(C.bg, C.Y, .06), C.Y, 1.3 + 2 * gS);
        if (d && d.letter >= 0) {
          ctx.save(); ctx.globalAlpha = .25 + .75 * d.share; PE.label(ctx, LET[d.letter], x + G.cw / 2, g.ry + G.ch / 2 + (narrow ? 4 : 5), C.fg, narrow ? 13 : 17, 'center'); ctx.restore();
          PE.label(ctx, `${Math.round(100 * d.share)}%`, x + G.cw / 2, g.ry + G.ch - 3, C.mu, narrow ? 7.5 : 8.5, 'center');
        } else PE.label(ctx, '?', x + G.cw / 2, g.ry + G.ch / 2 + 4, C.mu, narrow ? 11 : 13, 'center');
        // Read-out under (or beside) the register.
        const ow = narrow ? G.cw : G.cw, oh = narrow ? G.ch : 24, oyy = narrow ? oy : oy;
        const o = ln.dec[i], ok = o.letter === q.letters[i], show = fin;
        rr(ctx, ox(i), oyy, ow, oh, 4, show ? PE.mix(C.bg, ok ? C.G : C.R, .18) : null, show ? (ok ? C.G : C.R) : C.rule, 1.3);
        PE.label(ctx, show ? (o.letter >= 0 ? LET[o.letter] : '?') : '?', ox(i) + ow / 2, oyy + oh / 2 + (narrow ? 4 : 4.5), show ? (ok ? C.G : C.R) : C.mu, narrow ? 12 : 13, 'center');
      }
      if (narrow) PE.arrow(ctx, cx(3) + G.cw + 3, g.ry + G.ch / 2, ox(0) - 3, g.ry + G.ch / 2, C.mu, 1.2, 6, .8);
      else PE.label(ctx, fin ? `read-out: ${ln.ok} of ${K} right` : 'read-out', G.rx, oy + 38, fin ? (ln.ok === K ? C.G : C.R) : C.mu, 10);
      if (narrow && fin) PE.label(ctx, `${ln.ok} of ${K}`, W - 8, g.title, ln.ok === K ? C.G : C.R, 10.5, 'right');
      // Selective lane, once revealed: above each letter's gate, the slot that letter ends in.
      if (li === 1 && S.revealed) q.pos.forEach((t, j) => {
        if (tau <= t) return;
        const a = Math.min(1, (tau - t) * 1.5), x = tx(t), y = g.y0 - 2, s = narrow ? 11 : 15;
        rr(ctx, x - s / 2, y - s + 4, s, s, 3, PE.mix(C.bg, C.Y, .25), C.Y, 1 + 1.5 * gW, a);
        ctx.save(); ctx.globalAlpha = a; PE.label(ctx, String(K - j), x, y + (narrow ? 1.5 : 2.5), C.Y, narrow ? 8.5 : 10, 'center'); ctx.restore();
      });
    });
  }
  function readout(S) {
    const R = result(S), q = R.q, w = q.letters.map(i => LET[i]).join(' '), where = q.pos.map(p => p + 1).join(', ');
    const lay = S.mode === 'even' ? 'Even gaps' : 'Random gaps';
    if (!S.revealed) return `${lay}: ${w} at positions ${where}. Which lane will read ${w} back from its four slots? Predict, then press Reveal.`;
    const sh = R.ld.map(d => Math.round(100 * d.share)), t = tally(S.p), L = d => d.map(x => (x.letter >= 0 ? LET[x.letter] : '?')).join(' ');
    return `${lay}. Fixed step ${S.p.toFixed(2)} reads ${L(R.ld)}: ${R.lc} of ${K} right, its slots ${Math.min(...sh)}% to ${Math.max(...sh)}% one letter. The gate reads ${L(R.sd)}: ${R.sc} of ${K}. Over ${t.n} random spacings this fixed step gets all four right in ${t.all}; the gate in all ${t.n}.`;
  }
  return { T, K, NS, LET, sequence, run, decode, kernel, tally, result, draw, readout, layout };
})();
