<script>
(() => {
  const fig = document.getElementById('viz-siglip-batch'), el = document.getElementById('siglip-batch-canvas');
  const out = document.getElementById('siglip-batch-readout');
  const rD = document.getElementById('rng-siglip-batch-dev'), rP = document.getElementById('rng-siglip-batch-per');
  const oD = document.getElementById('siglip-batch-dev-val'), oP = document.getElementById('siglip-batch-per-val');
  // D devices, per-device batch pb; global batch |B| = D * pb. mode: 'full' (vanilla softmax) or 'chunk' (sigmoid, s = steps done).
  const S = { D: 3, pb: 4, mode: 'chunk', s: 0, fa: 1 };
  const glow = PE.linkPulses(fig, () => cv.redraw());
  let stop = () => {};
  const fmt = n => n >= 1e12 ? (n / 1e12).toPrecision(3) + ' trillion' : n >= 1e9 ? (n / 1e9).toPrecision(3) + ' billion'
    : n >= 1e6 ? (n / 1e6).toPrecision(3) + ' million' : n.toLocaleString('en-US');

  const cv = PE.canvas(el, w => w < 560 ? 1.35 : .5, (ctx, W, H) => {
    const B = PE.css('--blue'), Y = PE.css('--yellow'), Rd = PE.css('--red'), F = PE.css('--fg'), M = PE.css('--muted'),
      Rule = PE.css('--rule'), BG = PE.css('--bg');
    const bg = BG.length === 7 ? BG : '#16181d';
    ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
    const stacked = W < 560, D = S.D, NB = D * S.pb;
    const Q = stacked ? Math.min(W - 70, H * .52) : H - 70, ox = 46, oy = stacked ? 54 : 40;
    const gDev = glow('devices'), gSwap = glow('swap'), gBlock = glow('block');
    const q = Q / D;
    // Image embeddings stay on their device (blue bands, left); text embeddings travel (yellow bands, top).
    for (let d = 0; d < D; d++) {
      ctx.fillStyle = PE.mix(bg, B, .35 + .5 * gDev + .15 * (d % 2)); ctx.fillRect(ox - 12, oy + d * q, 8, q - (D <= 32 ? 1 : 0));
      const k = S.mode === 'chunk' ? Math.min(D - 1, Math.floor(S.s)) : 0;
      ctx.fillStyle = PE.mix(bg, Y, .35 + .5 * gSwap + .15 * (d % 2)); ctx.fillRect(ox + d * q, oy - 12, q - (D <= 32 ? 1 : 0), 8);
      if (D <= 6) { PE.label(ctx, 'dev ' + (d + 1), ox - 16, oy + d * q + q / 2 + 4, B, 10.5, 'right');
        if (S.mode === 'chunk') PE.label(ctx, 'on dev ' + (((d - k) % D + D) % D + 1), ox + d * q + q / 2, oy - 17, Y, 10, 'center'); }
    }
    PE.label(ctx, 'images (rows) x texts (columns), ' + NB.toLocaleString('en-US') + ' each', ox, oy + Q + 18, M, 11);
    ctx.strokeStyle = Rule; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, Q, Q);
    if (NB <= 24) { ctx.globalAlpha = .5; for (let c = 1; c < NB; c++) { const t = Q * c / NB;
      ctx.beginPath(); ctx.moveTo(ox + t, oy); ctx.lineTo(ox + t, oy + Q); ctx.moveTo(ox, oy + t); ctx.lineTo(ox + Q, oy + t); ctx.stroke(); } ctx.globalAlpha = 1; }
    if (D > 1 && D <= 32) { ctx.strokeStyle = M; ctx.lineWidth = 1.2; for (let d = 1; d < D; d++) { const t = d * q;
      ctx.beginPath(); ctx.moveTo(ox + t, oy); ctx.lineTo(ox + t, oy + Q); ctx.moveTo(ox, oy + t); ctx.lineTo(ox + Q, oy + t); ctx.stroke(); } }
    if (S.mode === 'full') {
      ctx.fillStyle = Rd; ctx.globalAlpha = (.55 + .3 * gBlock) * S.fa; ctx.fillRect(ox, oy, Q, Q); ctx.globalAlpha = 1;
      PE.label(ctx, 'whole table in memory', ox + Q / 2, oy + Q / 2 + 4, F, 12, 'center');
    } else {
      // Step k: device d holds image block d and the text block that started on device (d + k) mod D.
      const k = Math.floor(S.s), frac = S.s - k;
      for (let d = 0; d < D; d++) {
        const y = oy + d * q;
        for (let kk = 0; kk < Math.min(k, D); kk++) { const c = (d + kk) % D; ctx.fillStyle = PE.mix(bg, Rd, .22); ctx.fillRect(ox + c * q, y, q, q); }
        if (k < D) { const c = (d + k) % D; ctx.fillStyle = Rd; ctx.globalAlpha = Math.min(1, .75 + .25 * gBlock) * (k === 0 ? 1 : Math.min(1, .6 + frac * 2)); ctx.fillRect(ox + c * q, y, q, q); ctx.globalAlpha = 1;
          if (gBlock > 0) { ctx.strokeStyle = F; ctx.lineWidth = 2 * gBlock; ctx.strokeRect(ox + c * q + 1, y + 1, q - 2, q - 2); } }
      }
    }
    // Memory: cells held at once, log scale from 1 to 10^13.
    const bx = stacked ? 12 : ox + Q + 40, by = stacked ? oy + Q + 44 : oy + 30, bw = stacked ? W - 24 : W - bx - 16, lg = v => Math.min(1, Math.log10(Math.max(1, v)) / 15);
    const full = NB * NB, blk = S.pb * S.pb;
    PE.label(ctx, 'similarities held in memory at once (log scale)', bx, by, F, 12);
    const bar = (y, v, txt, c) => { ctx.fillStyle = Rule; ctx.fillRect(bx, y, bw, 14); ctx.fillStyle = c; ctx.fillRect(bx, y, bw * lg(v), 14); PE.label(ctx, txt, bx, y - 5, c === Rd ? F : M, 11); };
    bar(by + 26, full, 'softmax, full table: |B|^2 = ' + fmt(full), S.mode === 'full' ? Rd : M);
    bar(by + 66, blk, 'sigmoid, chunked: (per-device batch)^2 = ' + fmt(blk), S.mode === 'chunk' ? Rd : M);
    PE.label(ctx, 'ratio ' + fmt(D * D) + ' = D^2', bx, by + 104, M, 11);
    const steps = S.mode === 'chunk' ? `permute step ${Math.min(D, Math.floor(S.s) + 1)} of ${D}` : 'after all-gathers';
    PE.label(ctx, steps, bx, by + 124, S.mode === 'chunk' ? Y : M, 11);
  });

  function readout() {
    const NB = S.D * S.pb;
    out.textContent = `Global batch ${NB.toLocaleString('en-US')} on ${S.D} device${S.D > 1 ? 's' : ''} (per-device batch ${S.pb.toLocaleString('en-US')}). ` +
      (S.mode === 'full' ? `Softmax: all ${fmt(NB * NB)} similarities are materialised together.`
        : `Chunked sigmoid: each device holds one ${S.pb} by ${S.pb} block (${fmt(S.pb * S.pb)} similarities), then passes its texts on; ${S.D} steps cover the whole table, ${S.D * S.D} times less memory at any moment.`);
  }
  function syncSliders() { rD.value = Math.round(Math.log2(S.D)); rP.value = Math.round(Math.log2(S.pb)); oD.textContent = S.D; oP.textContent = S.pb.toLocaleString('en-US'); }
  function sweep() { stop(); S.mode = 'chunk'; S.s = 0; const D = S.D;
    stop = PE.animate(Math.min(6000, 900 * D), u => { S.s = u * D; cv.redraw(); readout(); }, () => { S.s = D; cv.redraw(); readout(); }); }
  function full() { stop(); S.mode = 'full'; S.fa = 0; readout(); stop = PE.animate(700, u => { S.fa = u; cv.redraw(); }); }
  document.getElementById('btn-siglip-batch-full').addEventListener('click', full);
  document.getElementById('btn-siglip-batch-chunk').addEventListener('click', sweep);
  document.getElementById('btn-siglip-batch-mock').addEventListener('click', () => { S.D = 3; S.pb = 4; syncSliders(); sweep(); });
  document.getElementById('btn-siglip-batch-reset').addEventListener('click', () => { stop(); S.D = 3; S.pb = 4; S.mode = 'chunk'; S.s = 0; syncSliders(); cv.redraw(); readout(); });
  rD.addEventListener('input', () => { stop(); S.D = 2 ** +rD.value; S.s = S.mode === 'chunk' ? 0 : S.s; syncSliders(); cv.redraw(); readout(); });
  rP.addEventListener('input', () => { stop(); S.pb = 2 ** +rP.value; syncSliders(); cv.redraw(); readout(); });
  syncSliders(); readout();
})();
</script>
