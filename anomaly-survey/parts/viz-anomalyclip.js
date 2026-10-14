<style>
#viz-anomalyclip svg { display:block; width:100%; height:auto; font-family:system-ui, sans-serif; touch-action:auto; }
#viz-anomalyclip svg .pe-lit { stroke:var(--yellow); stroke-width:2.5px; }
#viz-anomalyclip svg text.pe-lit { fill:var(--yellow); stroke:none; }
</style>
<script>
(() => {
  // ch-anomalyclip, sourced. Values and claim ids from AnomalyCLIP v12 Table 1, Fig. 5 (gains) and Fig. 8 (product name at test).
  const fig = document.getElementById('viz-anomalyclip'), wrap = document.getElementById('svgwrap-anomalyclip'), out = fig.querySelector('.readout');
  const T = [
    { set: 'MVTec AD', src: 'visa', img: ['91.5', 'anomalyclip-mvtec-img-auroc'], pix: ['91.1', 'anomalyclip-mvtec-pix-auroc'],
      wimg: ['91.8', 'anomalyclip-t1-winclip-mvtec-img-auroc', 'copied'], wpix: ['85.1', 'anomalyclip-t1-winclip-mvtec-pix-auroc', 'copied'] },
    { set: 'VisA', src: 'mvtec', img: ['82.1', 'anomalyclip-visa-img-auroc'], pix: ['95.5', 'anomalyclip-visa-pix-auroc'],
      wimg: ['78.1', 'anomalyclip-t1-winclip-visa-img-auroc', 'copied'], wpix: ['79.6', 'anomalyclip-t1-winclip-visa-pix-auroc', 'copied'] },
    { set: 'MPDD', src: 'mvtec', img: ['77.0', 'anomalyclip-mpdd-img-auroc'], pix: ['96.5', 'anomalyclip-mpdd-pix-auroc'], wimg: ['63.6', 'anomalyclip-t1-winclip-mpdd-img-auroc', 're-run'], wpix: null },
    { set: 'BTAD', src: 'mvtec', img: ['88.3', 'anomalyclip-btad-img-auroc'], pix: ['94.2', 'anomalyclip-btad-pix-auroc'], wimg: null, wpix: ['72.7', 'anomalyclip-t1-winclip-btad-pix-auroc', 're-run'] },
    { set: 'SDD', src: 'mvtec', img: ['84.7', 'anomalyclip-sdd-img-auroc'], pix: ['90.6', 'anomalyclip-sdd-pix-auroc'], wimg: null, wpix: null },
    { set: 'DAGM', src: 'mvtec', img: ['97.5', 'anomalyclip-dagm-img-auroc'], pix: ['95.6', 'anomalyclip-dagm-pix-auroc'], wimg: null, wpix: null },
    { set: 'DTD-Synthetic', src: 'mvtec', img: ['93.5', 'anomalyclip-dtd-img-auroc'], pix: ['97.9', 'anomalyclip-dtd-pix-auroc'], wimg: null, wpix: null },
  ];
  const MET = ['image AUROC', 'image AP', 'pixel AUROC', 'AUPRO'], g = (d, m) => [d, `anomalyclip-gain-agnostic-${d}-${m}`];
  const GAINS = [
    { set: 'MVTec AD', v: [['0.8', 'mvtec', 'img-auroc'], ['0.0', 'mvtec', 'img-ap'], ['0.6', 'mvtec', 'pix-auroc'], ['1.0', 'mvtec', 'aupro']] },
    { set: 'VisA', v: [['0.3', 'visa', 'img-auroc'], ['0.5', 'visa', 'img-ap'], ['0.5', 'visa', 'pix-auroc'], ['0.2', 'visa', 'aupro']] },
    { set: 'MPDD', v: [['4.4', 'mpdd', 'img-auroc'], ['3.3', 'mpdd', 'img-ap'], ['0.8', 'mpdd', 'pix-auroc'], ['0.9', 'mpdd', 'aupro']] },
    { set: 'BTAD', v: [null, null, null, ['1.8', 'btad', 'aupro']] },
  ].map(r => ({ set: r.set, v: r.v.map(c => c && { v: c[0], id: g(c[1], c[2])[1] }) }));
  const FIG8 = [['object at test (as trained)', '91.5', 'anomalyclip-mvtec-img-auroc'], ['product name at test', '89.1', 'anomalyclip-re-mvtec-img-auroc']];
  const DEF = { view: 'transfer', metric: 'img' };
  const S = { ...DEF, u: 1, from: 'img' };
  let W = 0, stop = () => {};
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
  function header(W) {
    // The two learned prompts, shared by every dataset: green squares are learned vectors, the words are fixed tokens.
    const narrow = W < 520, sq = narrow ? 9 : 12, gap = 3, x0 = narrow ? 104 : 150, oy = narrow ? 30 : 0;
    const row = (y, lbl, col, link, words, linkWords) => {
      let s = `<g data-link="${link}"><text x="0" y="${y + sq - 1}" font-size="12" fill="${col}">${lbl}</text>`;
      for (let k = 0; k < 12; k++) s += `<rect x="${x0 + k * (sq + gap)}" y="${y}" width="${sq}" height="${sq}" rx="2" fill="var(--green)" opacity=".85" data-link="${link}"></rect>`;
      let x = x0 + 12 * (sq + gap) + 6;
      words.forEach((w, i) => { s += `<text x="${x}" y="${y + sq - 1}" font-size="12" fill="var(--fg)" data-link="${linkWords[i]}">${w}</text>`; x += w.length * 7 + 8; });
      return s + '</g>';
    };
    return row(oy + 4, 'normal prompt', 'var(--blue)', 'prompt-n', ['object'], ['object'])
      + row(oy + 10 + sq, 'abnormal prompt', 'var(--red)', 'prompt-a', ['damaged', 'object'], ['prompt-a', 'object'])
      + `<text x="0" y="${oy + 2 * sq + 34}" font-size="11" fill="var(--muted)">${narrow ? 'One pair of learned prompts for every category.' : 'One pair of learned prompts (12 green vectors each) for every category below.'}</text>`;
  }
  function transfer(W, y0) {
    const narrow = W < 520, labW = narrow ? 92 : 128, x0 = labW + 8, x1 = W - (narrow ? 44 : 64), X = v => x0 + (v - 50) / 50 * (x1 - x0);
    const key = S.metric, prev = S.from, u = S.u;
    let s = `<text x="${x0}" y="${y0}" font-size="11" fill="var(--muted)">${key === 'img' ? 'Image AUROC' : 'Pixel AUROC'} (50 = chance)</text>`, y = y0 + 18;
    const yTick = y; [50, 75, 100].forEach(t => { s += `<text x="${X(t)}" y="${y + 4}" font-size="11" fill="var(--muted)" text-anchor="middle">${t}</text>`; });
    y += 22;
    [['visa', 'Prompts trained on VisA (test split)'], ['mvtec', 'Prompts trained on MVTec AD (test split)']].forEach(([src, title]) => {
      s += `<text x="0" y="${y + 12}" font-size="12" font-weight="600" fill="var(--fg)" data-link="aux">${title}</text>`; y += 22;
      T.filter(r => r.src === src).forEach(r => {
        const a = r[key], b = r[prev], v = +b[0] + (+a[0] - +b[0]) * u;
        s += `<text x="${labW}" y="${y + 13}" font-size="12" fill="var(--fg)" text-anchor="end">${r.set}</text>`;
        s += `<rect x="${x0}" y="${y + 3}" width="${Math.max(0, X(v) - x0)}" height="13" rx="3" fill="var(--green)" opacity=".85"></rect>`;
        const w = r['w' + key], lx = w && Math.abs(X(+w[0]) - X(v)) < 34 ? Math.max(X(v), X(+w[0])) + 9 : X(v) + 5;
        s += `<text x="${lx}" y="${y + 14}" font-size="12" fill="var(--fg)" data-claim="${a[1]}">${a[0]}</text>`;
        if (w) { const wx = X(+w[0]);
          s += `<path d="M${wx} ${y + 1} l5 8 l-5 8 l-5 -8 z" fill="none" stroke="var(--brown)" stroke-width="1.8"></path>`;
          s += `<text x="${wx}" y="${y + 28}" font-size="10" fill="var(--brown)" text-anchor="middle"><tspan data-claim="${w[1]}">${w[0]}</tspan> ${w[2]}</text>`; }
        y += w ? 34 : 24;
      });
      y += 6;
    });
    s = [50, 75, 100].map(t => `<line x1="${X(t)}" x2="${X(t)}" y1="${yTick + 10}" y2="${y - 6}" stroke="var(--rule)"></line>`).join('') + s;
    const lx2 = narrow ? 6 : 222, ly2 = narrow ? 20 : 0;
    s += `<g transform="translate(0 ${y + 4})"><rect x="0" y="0" width="12" height="10" rx="2" fill="var(--green)" opacity=".85"></rect><text x="18" y="9" font-size="11" fill="var(--muted)">AnomalyCLIP (learned prompts)</text>`
      + `<path d="M${lx2} ${ly2 - 3} l5 8 l-5 8 l-5 -8 z" fill="none" stroke="var(--brown)" stroke-width="1.8"></path><text x="${lx2 + 12}" y="${ly2 + 9}" font-size="11" fill="var(--muted)">WinCLIP (hand prompts, no training)</text></g>`;
    return [s, y + (narrow ? 40 : 22)];
  }
  function agnostic(W, y0) {
    const narrow = W < 520, labW = narrow ? 76 : 104, x0 = labW + 10, x1 = W - 40, X = v => x0 + v / 5 * (x1 - x0);
    let s = `<text x="0" y="${y0 + 12}" font-size="12" font-weight="600" fill="var(--fg)" data-link="agnostic">${narrow ? 'Gain of "object" over the name (points)' : 'Gain of "object" over the product name in the learned prompts (points)'}</text>`, y = y0 + 26;
    [0, 1, 2, 3, 4, 5].forEach(t => { s += `<line x1="${X(t)}" x2="${X(t)}" y1="${y}" y2="${y + GAINS.length * 70 - 8}" stroke="var(--rule)"></line><text x="${X(t)}" y="${y - 2}" font-size="10" fill="var(--muted)" text-anchor="middle">${t}</text>`; });
    y += 6;
    GAINS.forEach(r => {
      s += `<text x="${labW}" y="${y + 28}" font-size="12" fill="var(--fg)" text-anchor="end">${r.set}</text>`;
      r.v.forEach((c, m) => { const yy = y + m * 15;
        if (!c) { s += `<text x="${x0 + 4}" y="${yy + 10}" font-size="10" fill="var(--muted)">${MET[m]}: not printed</text>`; return; }
        s += `<rect x="${x0}" y="${yy + 2}" width="${Math.max(1.5, X(+c.v * S.u) - x0)}" height="10" rx="2" fill="var(--green)" opacity="${.45 + .15 * m}" data-link="agnostic"></rect>`;
        s += `<text x="${Math.max(X(+c.v * S.u), x0 + 1.5) + 4}" y="${yy + 11}" font-size="10" fill="var(--fg)"><tspan data-claim="${c.id}">${c.v}</tspan> ${narrow ? '' : MET[m]}</text>`; });
      y += 70;
    });
    s += `<text x="0" y="${y + 10}" font-size="12" font-weight="600" fill="var(--fg)">${narrow ? 'MVTec AD image AUROC, name back at test' : 'MVTec AD image AUROC when "object" is swapped back to the product name at test time'}</text>`; y += 22;
    const X2 = v => x0 + (v - 80) / 20 * (x1 - x0);
    FIG8.forEach(([lbl, v, id], i) => {
      s += `<text x="${labW}" y="${y + 12}" font-size="11" fill="var(--fg)" text-anchor="end">${narrow ? (i ? 'name' : 'object') : (i ? 'name at test' : 'as trained')}</text>`;
      s += `<rect x="${x0}" y="${y + 2}" width="${X2(80 + (+v - 80) * S.u) - x0}" height="12" rx="3" fill="${i ? 'var(--brown)' : 'var(--green)'}" opacity=".85"></rect>`;
      s += `<text x="${X2(80 + (+v - 80) * S.u) + 5}" y="${y + 13}" font-size="11" fill="var(--fg)" data-claim="${id}">${v}</text>`; y += 20; });
    s += `<text x="${x0}" y="${y + 12}" font-size="10" fill="var(--muted)">axis from 80</text>`;
    return [s, y + 22];
  }
  function render() {
    W = wrap.clientWidth || 600;
    let body, h; const top = W < 520 ? 104 : 80;
    [body, h] = S.view === 'transfer' ? transfer(W, top) : agnostic(W, top);
    wrap.innerHTML = `<svg viewBox="0 0 ${W} ${h}" width="${W}" height="${h}" role="img" aria-label="${S.view === 'transfer' ? 'Bar chart: AnomalyCLIP zero-shot scores on seven datasets, grouped by the dataset its prompts were trained on, with WinCLIP marks where available.' : 'Bar chart: gains of object-agnostic over object-aware learned prompts on four datasets, and MVTec AD image AUROC with the product name swapped back at test time.'}">${header(W)}${body}</svg>`;
  }
  function say() {
    if (S.view === 'transfer') { const k = S.metric, m = k === 'img' ? 'Image AUROC' : 'Pixel AUROC';
      const a = T[0], rest = T.slice(1).map(r => `${r.set} ${r[k][0]}${r['w' + k] ? ` (WinCLIP ${r['w' + k][0]}, ${r['w' + k][2]})` : ''}`).join(', ');
      out.textContent = `${m}. Prompts trained on VisA: MVTec AD ${a[k][0]}${a['w' + k] ? ` (WinCLIP ${a['w' + k][0]}, ${a['w' + k][2]})` : ''}. Prompts trained on MVTec AD: ${rest}.`; }
    else out.textContent = `Gain of object-agnostic over object-aware prompts, in points: MVTec AD ${GAINS[0].v.map(c => c.v).join(' / ')}, VisA ${GAINS[1].v.map(c => c.v).join(' / ')}, MPDD ${GAINS[2].v.map(c => c.v).join(' / ')} (image AUROC / image AP / pixel AUROC / AUPRO); BTAD AUPRO ${GAINS[3].v[3].v}. Product name back at test time: ${FIG8[0][1]} to ${FIG8[1][1]}.`;
  }
  const press = (ids, on) => ids.forEach(k => document.getElementById('btn-aclip-' + k).setAttribute('aria-pressed', k === on));
  function setView(v) { if (v === S.view) return; stop(); S.view = v; S.from = S.metric; press(['transfer', 'agnostic'], v);
    stop = PE.animate(600, u => { S.u = u; render(); }, () => { S.u = 1; render(); }); say(); }
  function setMetric(m) { if (m === S.metric) return; stop(); S.from = S.metric; S.metric = m; press(['img', 'pix'], m);
    stop = PE.animate(600, u => { S.u = u; render(); }, () => { S.u = 1; S.from = m; render(); }); say(); }
  ['transfer', 'agnostic'].forEach(k => document.getElementById('btn-aclip-' + k).addEventListener('click', () => setView(k)));
  ['img', 'pix'].forEach(k => document.getElementById('btn-aclip-' + k).addEventListener('click', () => setMetric(k)));
  document.getElementById('btn-aclip-reset').addEventListener('click', () => { setView(DEF.view); setMetric(DEF.metric); });
  new ResizeObserver(() => { if (Math.abs((wrap.clientWidth || 0) - W) > 2) render(); }).observe(wrap);
  render(); say();
})();
</script>
