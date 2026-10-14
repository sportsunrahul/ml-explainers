<script>
(() => {
  const fig = document.getElementById('viz-capfilt'), el = document.getElementById('capfilt-canvas');
  const out = document.getElementById('capfilt-readout'), verdict = document.getElementById('capfilt-verdict');
  const bBeam = document.getElementById('btn-capfilt-beam'), bNuc = document.getElementById('btn-capfilt-nucleus');
  // Hand-set ITM match probabilities (toy). Beam captions are safe and generic; nucleus captions are specific, and one is wrong.
  const D = [
    { web: ['IMG_4032.JPG', .04], beam: ['a dog on the beach', .81], nuc: ['a brown dog leaps for a frisbee over the sand', .88] },
    { web: ['Best sofa deals, free shipping!', .31], beam: ['a cat sitting on a couch', .86], nuc: ['a sleepy cat curled up on a striped sofa', .83] },
    { web: ['homemade margherita pizza with fresh basil', .91], beam: ['a pizza on a table', .79], nuc: ['a pizza topped with basil leaves on a wooden board', .87] },
    { web: ['vintage car parked by the road', .89], beam: ['a car parked on the street', .84], nuc: ['an old car towing a camper through the snow', .22] },
    { web: ['Our trip, summer 2019, day 3 :)', .12], beam: ['a lake with mountains in the background', .80], nuc: ['two snowy peaks reflected in a still lake under the sun', .85] },
  ];
  const TH = .5;
  const S = { mode: 'nuc', cap: 0, filt: 0, clean: 0, ver: 0 };
  let stop = () => {};
  const glow = PE.linkPulses(fig, () => cv.redraw());
  const mctx = document.createElement('canvas').getContext('2d');

  const dims = W => { const sm = W < 560; return { sm, fs: sm ? 11 : 12.5, ts: sm ? 9.5 : 10.5, imgW: sm ? 62 : 104, barW: sm ? 60 : 110, head: 60, foot: 46 }; };
  function wrap(text, maxW, size) {
    mctx.font = `600 ${size}px system-ui, sans-serif`;
    const words = text.split(' '), lines = []; let cur = '';
    words.forEach(w => { const t = cur ? cur + ' ' + w : w; if (mctx.measureText(t).width > maxW && cur) { lines.push(cur); cur = w; } else cur = t; });
    if (cur) lines.push(cur); return lines;
  }
  function layout(W) {
    const d = dims(W), tx = d.imgW + 14, tw = W - tx - d.barW - 10, lh = d.fs * 1.32, tl = d.ts * 1.5;
    let y = d.head; const rows = D.map(r => {
      const nl = k => Math.max(...(k === 'syn' ? [r.beam[0], r.nuc[0]] : [r.web[0]]).map(t => wrap(t, tw, d.fs).length));
      const hWeb = tl + nl('web') * lh, hSyn = tl + nl('syn') * lh, h = Math.max(d.imgW * .72, hWeb + hSyn + 8) + 14;
      const row = { y, h, yWeb: y + 6, ySyn: y + 6 + hWeb + 8 }; y += h; return row; });
    return { ...d, tx, tw, lh, tl, rows, H: y + d.foot };
  }

  // Line-art "photos" in the vision color: dog + frisbee, cat on sofa, pizza, car, mountains + lake.
  function drawImage(ctx, k, x, y, w, h, B, Rule) {
    ctx.save(); ctx.strokeStyle = B; ctx.fillStyle = B; ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.strokeRect(x + .5, y + .5, w - 1, h - 1);
    ctx.beginPath(); ctx.rect(x + 1, y + 1, w - 2, h - 2); ctx.clip();
    const X = u => x + u * w, Y = v => y + v * h, s = Math.min(w, h);
    const line = pts => { ctx.beginPath(); pts.forEach(([u, v], i) => i ? ctx.lineTo(X(u), Y(v)) : ctx.moveTo(X(u), Y(v))); ctx.stroke(); };
    const circ = (u, v, r, fill) => { ctx.beginPath(); ctx.arc(X(u), Y(v), r * s, 0, 7); fill ? ctx.fill() : ctx.stroke(); };
    if (k === 0) { // dog leaping for a frisbee on the sand
      line([[0, .82], [.3, .78], [.6, .84], [1, .8]]);
      ctx.beginPath(); ctx.ellipse(X(.45), Y(.52), .2 * w, .09 * h, -.25, 0, 7); ctx.stroke();
      circ(.66, .36, .08); line([[.3, .58], [.2, .74]]); line([[.38, .6], [.34, .76]]); line([[.52, .5], [.62, .64]]); line([[.26, .54], [.14, .46]]);
      ctx.beginPath(); ctx.ellipse(X(.82), Y(.2), .1 * w, .035 * h, .2, 0, 7); ctx.fill();
    } else if (k === 1) { // cat curled on a striped sofa
      ctx.strokeRect(X(.1), Y(.52), .8 * w, .3 * h); ctx.strokeRect(X(.1), Y(.34), .8 * w, .18 * h);
      [.3, .5, .7].forEach(u => line([[u, .52], [u, .82]])); line([[.16, .82], [.16, .92]]); line([[.84, .82], [.84, .92]]);
      ctx.beginPath(); ctx.ellipse(X(.52), Y(.45), .17 * w, .08 * h, 0, 0, 7); ctx.stroke();
      circ(.36, .43, .07); line([[.32, .36], [.33, .3], [.36, .35]]); line([[.39, .35], [.41, .3], [.42, .36]]);
    } else if (k === 2) { // pizza with leaves on a board
      ctx.strokeRect(X(.12), Y(.12), .76 * w, .8 * h); circ(.5, .52, .32);
      for (let a = 0; a < 6; a++) line([[.5, .52], [.5 + .32 * s / w * Math.cos(a * 1.047), .52 + .32 * s / h * Math.sin(a * 1.047)]]);
      [[.42, .42], [.6, .46], [.5, .66], [.36, .6]].forEach(([u, v]) => circ(u, v, .035, true));
    } else if (k === 3) { // car by the road
      line([[0, .82], [1, .82]]);
      line([[.12, .7], [.12, .55], [.3, .52], [.4, .36], [.66, .36], [.76, .52], [.9, .55], [.9, .7], [.12, .7]]);
      line([[.43, .4], [.43, .52]]); line([[.63, .4], [.7, .52]]); circ(.3, .72, .075); circ(.72, .72, .075);
    } else { // two snowy peaks, a lake, the sun
      circ(.82, .18, .07);
      line([[0, .62], [.3, .18], [.5, .46], [.66, .26], [1, .62]]); line([[.22, .3], [.3, .18], [.38, .3]]); line([[.6, .34], [.66, .26], [.72, .34]]);
      line([[0, .62], [1, .62]]); ctx.globalAlpha = .55; line([[0, .62], [.3, .98], [.5, .74], [.66, .9], [1, .62]]); ctx.globalAlpha = 1;
      line([[.15, .78], [.35, .78]]); line([[.6, .84], [.8, .84]]);
    }
    ctx.restore();
  }

  const cv = PE.canvas(el, w => layout(w).H / w, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), G = PE.css('--green'), Rd = PE.css('--red'), F = PE.css('--fg'),
      M = PE.css('--muted'), Rule = PE.css('--rule'), BG = PE.css('--bg');
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const Lh = layout(W), { tx, tw, lh, tl, fs, ts, imgW, barW } = Lh;
    const gi = glow('itm'), gl = glow('lm');
    PE.label(ctx, Lh.sm ? 'image' : 'web image', 6, 50, B, 12);
    PE.label(ctx, 'its texts', tx, 50, Y, 12);
    PE.label(ctx, Lh.sm ? 'filter (ITM)' : 'filter: ITM p(match)', W - 6, 50, S.filt > .01 ? F : M, 12, 'right');
    const drawText = (text, y, kind, p) => {
      const syn = kind !== 'web', shown = syn ? text.slice(0, Math.round(text.length * S.cap)) : text;
      const drop = p < TH, fade = drop ? 1 - .8 * S.clean : 1, keep = !drop ? S.clean : 0;
      PE.label(ctx, syn ? `captioner (${S.mode === 'nuc' ? 'nucleus sampling' : 'beam search'})` : 'web alt-text', tx, y + ts, syn && gl > 0 ? F : M, ts + (syn ? 1.5 * gl : 0));
      const lines = wrap(text, tw, fs), vis = wrap(shown, tw, fs), y0 = y + tl;
      if (syn && S.cap < .02) {
        ctx.save(); ctx.strokeStyle = Rule; ctx.setLineDash([4, 4]); ctx.strokeRect(tx, y0 - 1, tw, lh * lines.length); ctx.restore();
        PE.label(ctx, 'no synthetic caption yet', tx + 6, y0 + fs, M, fs - 1); return;
      }
      ctx.save(); ctx.globalAlpha = fade;
      if (keep > 0) { ctx.fillStyle = G; ctx.globalAlpha = keep; ctx.fillRect(tx - 7, y0 - 1, 3, lh * lines.length); ctx.globalAlpha = fade; }
      vis.forEach((ln, i) => PE.label(ctx, ln, tx, y0 + fs + i * lh, Y, fs));
      if (drop && S.filt > 0) { ctx.globalAlpha = S.filt * fade; ctx.strokeStyle = Rd; ctx.lineWidth = 1.6;
        mctx.font = `600 ${fs}px system-ui, sans-serif`;
        lines.forEach((ln, i) => { const yy = y0 + fs * .68 + i * lh; ctx.beginPath(); ctx.moveTo(tx - 2, yy); ctx.lineTo(tx + 2 + mctx.measureText(ln).width * Math.min(1, S.filt * 1.4), yy); ctx.stroke(); }); }
      ctx.restore();
      // ITM score bar: grows with the filter step; threshold tick at 0.5; mark = kept (green) or dropped (red).
      if (S.filt > 0 && (!syn || S.cap > .98)) {
        const bx = W - barW, bw = barW - 22, by = y0 + 2, fill = p * bw * S.filt;
        ctx.save(); ctx.globalAlpha = fade;
        ctx.fillStyle = Rule; ctx.fillRect(bx, by, bw, 9); ctx.fillStyle = drop ? Rd : G; ctx.fillRect(bx, by, fill, 9);
        ctx.strokeStyle = F; ctx.lineWidth = 1 + 1.5 * gi; ctx.beginPath(); ctx.moveTo(bx + bw * TH, by - 3); ctx.lineTo(bx + bw * TH, by + 12); ctx.stroke();
        if (gi > 0) { ctx.strokeStyle = F; ctx.globalAlpha = gi; ctx.strokeRect(bx - 2, by - 2, bw + 4, 13); ctx.globalAlpha = fade; }
        ctx.globalAlpha = S.filt * fade; PE.label(ctx, (p * S.filt).toFixed(2), bx, by + 23, M, 10);
        const mx = W - 9, my = by + 4; ctx.strokeStyle = drop ? Rd : G; ctx.lineWidth = 2.2; ctx.beginPath();
        if (drop) { ctx.moveTo(mx - 5, my - 5); ctx.lineTo(mx + 5, my + 5); ctx.moveTo(mx + 5, my - 5); ctx.lineTo(mx - 5, my + 5); }
        else { ctx.moveTo(mx - 6, my); ctx.lineTo(mx - 2, my + 5); ctx.lineTo(mx + 6, my - 5); }
        ctx.stroke(); ctx.restore();
      }
    };
    D.forEach((r, k) => {
      const R = Lh.rows[k];
      ctx.strokeStyle = Rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, R.y); ctx.lineTo(W, R.y); ctx.stroke();
      drawImage(ctx, k, 4, R.y + 8, imgW, imgW * .72, B, Rule);
      drawText(r.web[0], R.yWeb, 'web', r.web[1]);
      const syn = r[S.mode]; drawText(syn[0], R.ySyn, 'syn', syn[1]);
    });
    const fy = Lh.rows[4].y + Lh.rows[4].h;
    ctx.strokeStyle = Rule; ctx.beginPath(); ctx.moveTo(0, fy); ctx.lineTo(W, fy); ctx.stroke();
    ctx.save(); ctx.globalAlpha = S.clean;
    PE.label(ctx, '+ the human-written pairs (COCO, Visual Genome)', 6, fy + 18, Y, Lh.sm ? 10.5 : 12);
    PE.label(ctx, 'green-marked texts + human pairs = the new set; pre-train a NEW model on it', 6, fy + 36, G, Lh.sm ? 10 : 12);
    ctx.restore();
  });

  const stats = () => {
    const kept = D.filter(r => r[S.mode][1] >= TH), web = D.filter(r => r.web[1] >= TH).length;
    const stopw = new Set('a an the on in of with for by over under up through and to'.split(' '));
    const vocab = new Set(kept.flatMap(r => r[S.mode][0].toLowerCase().split(' ').filter(w => !stopw.has(w))));
    return { syn: kept.length, web, vocab: vocab.size };
  };
  function readout() {
    const st = stats(), how = S.mode === 'nuc' ? 'nucleus sampling' : 'beam search';
    if (S.clean > .5) out.textContent = `Cleaned set: ${st.web} of 5 web texts and ${st.syn} of 5 synthetic captions (${how}) survive, ${st.web + st.syn} of 10 texts in all, and every image keeps at least one. A new model is pre-trained on these plus the human-written pairs.`;
    else if (S.filt > .5) out.textContent = `Filter: the ITM head scores each text against its image; below 0.5 means "unmatched" and the text is dropped. Web texts kept: ${st.web} of 5. Synthetic captions kept: ${st.syn} of 5. The kept ${how} captions use ${st.vocab} different content words.`;
    else if (S.cap > .5) out.textContent = `Captioner: one synthetic caption per web image, decoded by ${how}. Nothing has been judged yet.`;
    else out.textContent = 'Raw web pairs: five images with their alt-text as scraped. Three of the five texts say little or nothing about the picture.';
  }
  function tweenTo(tg, ms, done) {
    stop(); const a = { cap: S.cap, filt: S.filt, clean: S.clean };
    stop = PE.animate(ms, u => { for (const k in tg) S[k] = PE.lerp(a[k], tg[k], u); cv.redraw(); readout(); }, done);
  }
  function setMode(m) {
    S.mode = m; bBeam.setAttribute('aria-pressed', m === 'beam'); bNuc.setAttribute('aria-pressed', m === 'nuc');
    stop(); S.cap = 0; S.filt = 0; S.clean = 0;
    stop = PE.animate(1400, u => { S.cap = Math.min(1, u * 1.7); S.filt = Math.max(0, (u - .62) / .38); cv.redraw(); readout(); });
  }
  function showVerdict(v) {
    if (v) verdict.hidden = false;
    const a = S.ver; PE.animate(600, u => { S.ver = PE.lerp(a, v, u); verdict.style.opacity = S.ver; }, () => { if (!v) verdict.hidden = true; });
  }
  document.getElementById('btn-capfilt-raw').addEventListener('click', () => tweenTo({ cap: 0, filt: 0, clean: 0 }, 700));
  document.getElementById('btn-capfilt-caption').addEventListener('click', () => tweenTo({ cap: 1, filt: 0, clean: 0 }, 1300));
  document.getElementById('btn-capfilt-filter').addEventListener('click', () => tweenTo({ cap: 1, filt: 1, clean: 0 }, 1100));
  document.getElementById('btn-capfilt-clean').addEventListener('click', () => tweenTo({ cap: 1, filt: 1, clean: 1 }, 900));
  bBeam.addEventListener('click', () => setMode('beam'));
  bNuc.addEventListener('click', () => setMode('nuc'));
  document.getElementById('btn-capfilt-verdict').addEventListener('click', () => showVerdict(1));
  document.getElementById('btn-capfilt-reset').addEventListener('click', () => {
    stop(); S.mode = 'nuc'; S.cap = S.filt = S.clean = 0; S.ver = 0; verdict.style.opacity = 0; verdict.hidden = true;
    bBeam.setAttribute('aria-pressed', false); bNuc.setAttribute('aria-pressed', true); cv.redraw(); readout(); });
  verdict.style.opacity = 0; verdict.hidden = true; bNuc.setAttribute('aria-pressed', true); bBeam.setAttribute('aria-pressed', false); readout();
})();
</script>
