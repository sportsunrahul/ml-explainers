<style>
#viz-tree .tree-scroll { overflow-x:auto; border-radius:9px; background:var(--bg); }
#viz-tree #tree-chart { min-width:760px; }
#viz-tree #tree-chart svg { display:block; width:100%; height:auto; }
#viz-tree .nd { cursor:pointer; }
#viz-tree .nd:focus { outline:none; }
#viz-tree .nd:focus-visible circle, #viz-tree .nd.on circle { stroke:var(--yellow); stroke-width:2.5; }
#viz-tree .nd.on text { fill:var(--yellow); }
#viz-tree .tree-detail { min-height:4.4em; }
#viz-tree .tree-detail b { font-weight:600; }
#viz-tree details.tree-list { margin-top:.7rem; font:.9rem/1.5 system-ui, sans-serif; }
#viz-tree details.tree-list summary { cursor:pointer; color:var(--muted); }
#viz-tree details.tree-list summary:focus-visible { outline:2px solid var(--yellow); outline-offset:2px; }
#viz-tree details.tree-list h4 { font:600 .92rem system-ui, sans-serif; margin:.8rem 0 .2rem; }
#viz-tree details.tree-list ul { margin:0; padding-left:1.1rem; }
</style>
<script>
(() => {
  // ch-timeline: family tree. Nodes, dates, one-liners and key numbers come from notes/claims and notes/branches-*.json via
  // build/f7/gen.py (the JSON block in the figure). x = first arXiv date, rows = family lanes, sub-rows avoid label overlap.
  const fig = document.getElementById('viz-tree'), host = document.getElementById('tree-chart'), detail = fig.querySelector('.tree-detail');
  const T = JSON.parse(document.getElementById('tree-data').textContent);
  const DEF = { mode: 'all', sel: 'patchcore' };
  const S = { ...DEF };
  const yearOf = d => { const [y, m = '6', dd = '15'] = d.split('-'); return +y + (+m - 1) / 12 + (+dd - 1) / 365; };
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

  function layout() {
    const nodes = T.nodes.filter(n => S.mode === 'all' || n.spine);
    const t0 = Math.floor(Math.min(...nodes.map(n => yearOf(n.date))) * 2) / 2 - .2, t1 = Math.max(...nodes.map(n => yearOf(n.date))) + .35;
    const W = 900, left = 178, right = 12, top = 30, rowH = 19, lanePad = 12;
    const X = t => left + (t - t0) / (t1 - t0) * (W - left - right);
    const lanes = T.lanes.map(() => []);
    nodes.slice().sort((a, b) => a.date < b.date ? -1 : 1).forEach(n => {
      const x = X(yearOf(n.date)), w = n.name.length * (n.spine ? 7.2 : 6.4) + 14, L = lanes[n.lane];
      n._left = x + w > W - right;                               // label on the node's left near the right edge
      const a = n._left ? x - w : x - 4, e = n._left ? x + 6 : x + w;
      let r = 0; while (L[r] !== undefined && L[r] > a - 4) r++;
      L[r] = e; n._x = x; n._r = r;
    });
    let y = top; const laneY = [];
    lanes.forEach((L, i) => { const rows = Math.max(1, L.length); laneY.push({ y0: y, h: rows * rowH + lanePad }); y += rows * rowH + lanePad; });
    nodes.forEach(n => { n._y = laneY[n.lane].y0 + lanePad / 2 + n._r * rowH + rowH / 2; });
    return { nodes, W, H: y + 8, left, top, X, t0, t1, laneY };
  }

  function render() {
    const L = layout(), css = PE.css, fg = css('--fg'), mu = css('--muted'), ru = css('--rule');
    const byKey = Object.fromEntries(L.nodes.map(n => [n.key, n]));
    let s = `<svg viewBox="0 0 ${L.W} ${L.H}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">`;
    for (let yr = Math.ceil(L.t0); yr <= Math.floor(L.t1); yr++) {
      s += `<line x1="${L.X(yr)}" x2="${L.X(yr)}" y1="${L.top - 6}" y2="${L.H - 8}" stroke="${ru}"/><text x="${L.X(yr)}" y="${L.top - 12}" fill="${mu}" font-size="11" text-anchor="middle">${yr}</text>`;
    }
    L.laneY.forEach((ly, i) => {
      if (i) s += `<line x1="0" x2="${L.W}" y1="${ly.y0}" y2="${ly.y0}" stroke="${ru}"/>`;
      s += `<text x="8" y="${ly.y0 + 18}" fill="${mu}" font-size="11.5" font-weight="600">${esc(T.lanes[i])}</text>`;
    });
    T.edges.forEach(([a, b]) => { const p = byKey[a], q = byKey[b]; if (!p || !q) return;
      const mx = (p._x + q._x) / 2;
      s += `<path d="M${p._x} ${p._y} C${mx} ${p._y} ${mx} ${q._y} ${q._x} ${q._y}" fill="none" stroke="${mu}" stroke-width="1.4" opacity=".7"/>`; });
    L.nodes.forEach(n => {
      const on = n.key === S.sel;
      s += `<g class="nd${on ? ' on' : ''}" tabindex="0" data-key="${esc(n.key)}" aria-label="${esc(n.name + ', ' + n.date + ', ' + T.lanes[n.lane])}">`;
      s += n.spine ? `<circle cx="${n._x}" cy="${n._y}" r="5.5" fill="${fg}" stroke="${fg}"/>` : `<circle cx="${n._x}" cy="${n._y}" r="4" fill="${css('--bg')}" stroke="${mu}" stroke-width="1.5"/>`;
      const tw = n.name.length * (n.spine ? 7.2 : 6.4);
      s += `<text x="${n._left ? n._x - 9 : n._x + 9}" y="${n._y + 4}" fill="${n.spine ? fg : mu}" font-size="${n.spine ? 12 : 11}"${n.spine ? ' font-weight="600"' : ''}${n._left ? ' text-anchor="end"' : ''}>${esc(n.name)}</text>`;
      s += `<rect x="${n._left ? n._x - tw - 12 : n._x - 8}" y="${n._y - 9}" width="${tw + 20}" height="18" fill="transparent"/></g>`;
    });
    host.innerHTML = s + '</svg>';
  }

  function show(key) {
    const n = T.nodes.find(m => m.key === key); if (!n) return;
    S.sel = key; host.querySelectorAll('.nd').forEach(g => g.classList.toggle('on', g.dataset.key === key));
    const date = n.dateClaim ? `<span data-claim="${esc(n.dateClaim)}">${esc(n.dateV)}</span>` : esc(n.date);
    const num = n.kc ? ` Key number: <span data-claim="${esc(n.kc)}">${esc(n.kv)}</span> ${esc(n.klabel)}.` : '';
    const link = n.ch ? ` Covered in <a href="#${esc(n.ch)}">this chapter</a>.` : '';
    detail.innerHTML = `<b>${esc(n.name)}</b> (${date}; ${esc(T.lanes[n.lane])}${n.spine ? '' : ', branch'}). ${esc(n.line)}${num}${link} <a href="${esc(n.url)}">Paper</a>.`;
  }
  host.addEventListener('focusin', e => { const g = e.target.closest('.nd'); if (g) show(g.dataset.key); });
  host.addEventListener('click', e => { const g = e.target.closest('.nd'); if (g) { show(g.dataset.key); g.focus({ preventScroll: true }); } });
  function setMode(m) {
    S.mode = m; ['all', 'spine'].forEach(k => document.getElementById('btn-tree-' + k).setAttribute('aria-pressed', k === m));
    if (m === 'spine' && !T.nodes.find(n => n.key === S.sel && n.spine)) S.sel = DEF.sel;
    host.style.opacity = '0';
    PE.animate(350, u => { host.style.opacity = String(u); }, () => { host.style.opacity = ''; });
    render(); show(S.sel);
  }
  document.getElementById('btn-tree-all').addEventListener('click', () => setMode('all'));
  document.getElementById('btn-tree-spine').addEventListener('click', () => setMode('spine'));
  document.getElementById('btn-tree-reset').addEventListener('click', () => { S.sel = DEF.sel; setMode(DEF.mode); });
  PE.onTheme(() => { render(); show(S.sel); });
  render(); show(S.sel);
})();
</script>
