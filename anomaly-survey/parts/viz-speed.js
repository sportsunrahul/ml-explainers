<style>
#viz-speed .speed-chart svg { display:block; width:100%; height:auto; background:var(--bg); border-radius:9px; }
#viz-speed .pt { cursor:pointer; }
#viz-speed .pt:focus { outline:none; }
#viz-speed .pt:focus-visible .mk, #viz-speed .pt.on .mk { stroke:var(--yellow); stroke-width:2.5; }
#viz-speed .pt.on text { fill:var(--yellow); }
#viz-speed .controls.dim > span, #viz-speed .controls.dim button { opacity:.55; }
#viz-speed details.speed-data { margin-top:.7rem; font:.88rem system-ui, sans-serif; }
#viz-speed details.speed-data summary { cursor:pointer; color:var(--muted); }
#viz-speed details.speed-data summary:focus-visible { outline:2px solid var(--yellow); outline-offset:2px; }
#viz-speed table.speed-table td.src { color:var(--muted); font-size:.8rem; text-align:left; }
#viz-speed table.speed-table td:nth-child(2), #viz-speed table.speed-table td:nth-child(5) { text-align:left; }
</style>
<script>
(() => {
  // ch-speed: latency (log x) vs image AUROC (y), one point per (method, device, source). Data and every value come from
  // notes/claims via build/f7/gen.py (the JSON block in the figure); this file only draws. Buttons set state (cue-safe).
  const fig = document.getElementById('viz-speed'), host = document.getElementById('speed-chart');
  const D = JSON.parse(document.getElementById('speed-data').textContent);
  const out = fig.querySelector('.readout'), rng = document.getElementById('rng-speed-pps'), rOut = document.getElementById('out-speed-pps');
  const DEF = { view: 'desktop', gpu: 'a6000', acc: 'mvtec', v: 50 };
  const S = { ...DEF, focus: null };
  const XR = { desktop: [1, 300], jetson: [10, 2000], cpu: [5, 30000] };
  const VIEW = { desktop: 'Desktop GPU', jetson: 'Jetson-class boards', cpu: 'CPUs and a Raspberry Pi 5' };
  const GPU = { a6000: 'RTX A6000', a5000: 'RTX A5000', v100: 'Tesla V100', '3080': 'RTX 3080', '2080ti': 'RTX 2080 Ti' };
  const pps = v => Math.pow(10, -1 + 4 * v / 100);
  const fmtPps = p => p >= 10 ? Math.round(p).toString() : p >= 1 ? p.toFixed(1).replace(/\.0$/, '') : p.toFixed(2).replace(/0$/, '');
  const fmtMs = ms => ms >= 100 ? Math.round(ms).toLocaleString('en-US') : ms >= 10 ? ms.toFixed(0) : ms.toFixed(1);
  let shown = [], anim = null;

  function current() {
    const m = S.view === 'desktop' ? S.acc : 'mvtec';
    return D.points.filter(p => p.view === S.view && (S.view !== 'desktop' || p.sub === S.gpu))
      .map(p => ({ ...p, acc: p.accPct[m], accShow: p.accV[m] }));
  }
  const rug = () => D.rug.filter(r => r.view === S.view);
  function yRange(pts) { const lo = Math.min(...pts.map(p => p.acc)); return [Math.max(40, Math.floor((lo - 2) / 5) * 5), 100]; }

  function geometry() {
    const narrow = host.clientWidth < 520;
    const W = narrow ? 360 : 640, H = narrow ? 440 : 380;
    return { narrow, W, H, b: { x: narrow ? 40 : 50, y: 16, w: W - (narrow ? 52 : 66), h: H - (narrow ? 70 : 62) } };
  }
  function ticks(xr, narrow) {
    const t = [];
    for (let e = -1; e <= 5; e++) for (const m of narrow ? [1] : [1, 2, 5]) { const v = m * 10 ** e; if (v >= xr[0] && v <= xr[1]) t.push(v); }
    return t;
  }
  const fmtTick = v => v >= 1000 ? (v / 1000) + ' s' : v + (v === 1 ? ' ms' : '');

  function marker(proto, x, y, r) {
    if (proto === 'multi-class') return `<rect class="mk" x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}"/>`;
    if (proto === 'continual') return `<path class="mk" d="M${x} ${y - r * 1.3}L${x + r * 1.3} ${y}L${x} ${y + r * 1.3}L${x - r * 1.3} ${y}Z"/>`;
    if (proto === 'zero-shot') return `<path class="mk" d="M${x} ${y - r * 1.3}L${x + r * 1.2} ${y + r}L${x - r * 1.2} ${y + r}Z"/>`;
    return `<circle class="mk" cx="${x}" cy="${y}" r="${r}"/>`;
  }
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');
  const short = p => (p.label || p.name.replace(/ \(.*\)$/, '')).replace('TensorRT ', '').replace('Uninformed Students', 'US');

  function render(pts, alphaOf, xr, yr) {
    const G = geometry(), { b, W, H, narrow } = G, css = PE.css;
    const lx = v => Math.log10(v), X = v => b.x + (lx(v) - lx(xr[0])) / (lx(xr[1]) - lx(xr[0])) * b.w, Y = v => b.y + b.h - (v - yr[0]) / (yr[1] - yr[0]) * b.h;
    const budget = 1000 / pps(S.v), reserved = [];
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">`;
    const fg = css('--fg'), mu = css('--muted'), ru = css('--rule'), red = css('--red');
    // grid
    ticks(xr, narrow).forEach(v => { s += `<line x1="${X(v)}" x2="${X(v)}" y1="${b.y}" y2="${b.y + b.h}" stroke="${ru}"/><text x="${X(v)}" y="${b.y + b.h + 15}" fill="${mu}" font-size="${narrow ? 10 : 11}" text-anchor="middle">${fmtTick(v)}</text>`; });
    for (let v = yr[0]; v <= 100; v += yr[1] - yr[0] > 30 ? 10 : 5) s += `<line x1="${b.x}" x2="${b.x + b.w}" y1="${Y(v)}" y2="${Y(v)}" stroke="${ru}"/><text x="${b.x - 6}" y="${Y(v) + 4}" fill="${mu}" font-size="${narrow ? 10 : 11}" text-anchor="end">${v}</text>`;
    s += `<text x="${b.x + b.w / 2}" y="${H - (narrow ? 22 : 10)}" fill="${mu}" font-size="11" text-anchor="middle">latency per image (log scale)</text>`;
    const ylab = S.view === 'desktop' && S.acc === 'mean' ? 'mean image AUROC, 3 datasets (%)' : 'MVTec AD image AUROC (%)';
    s += `<text transform="translate(${narrow ? 11 : 14} ${b.y + b.h / 2}) rotate(-90)" fill="${mu}" font-size="11" text-anchor="middle">${ylab}</text>`;
    // time budget
    if (budget > xr[0]) {
      const bx = Math.min(X(Math.min(budget, xr[1])), b.x + b.w);
      if (budget < xr[1]) s += `<rect x="${bx}" y="${b.y}" width="${b.x + b.w - bx}" height="${b.h}" fill="${red}" opacity=".09"/>`;
      s += `<line x1="${bx}" x2="${bx}" y1="${b.y}" y2="${b.y + b.h}" stroke="${red}" stroke-width="2"/>`;
      // Label runs up along the line from the bottom, where the charts are emptiest.
      const lab = `budget ${fmtMs(budget)} ms per part`, lw = lab.length * 6.3, tx = bx < b.x + 22 ? bx + 14 : bx - 6;
      s += `<text transform="translate(${tx} ${b.y + b.h - 6}) rotate(-90)" fill="${red}" font-size="11" font-weight="600">${lab}</text>`;
      reserved.push({ x: tx - 12, y: b.y + b.h - 6 - lw, w: 15, h: lw });
    } else s += `<rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" fill="${red}" opacity=".09"/>`;
    const R = rug();
    if (R.length) { const x0 = Math.min(...R.map(r => X(r.latMs))), x1 = Math.max(...R.map(r => X(r.latMs)));
      s += `<text x="${x1 + 6}" y="${b.y + b.h - 4}" fill="${mu}" font-size="10">latency only</text>`; reserved.push({ x: x0 - 8, y: b.y + b.h - 18, w: x1 - x0 + 80, h: 18 }); }
    // latency-only rug
    rug().forEach((r, i) => { const x = X(r.latMs);
      s += `<g class="pt rug" tabindex="0" data-rug="${i}" aria-label="${esc(r.name + ', ' + r.device + ': ' + r.latV + ' ms, no accuracy for this run')}"><line class="mk" x1="${x}" x2="${x}" y1="${b.y + b.h - 14}" y2="${b.y + b.h}" stroke="${mu}" stroke-width="2"/><rect x="${x - 6}" y="${b.y + b.h - 16}" width="12" height="18" fill="transparent"/></g>`; });
    // points and labels
    const placed = reserved.slice(), r = narrow ? 4.5 : 5.5, fs = narrow ? 10 : 11;
    const order = pts.map((p, i) => i).sort((i, j) => pts[i].acc - pts[j].acc);
    const geo = pts.map(p => ({ x: X(p.latMs), y: Y(p.acc) }));
    const labels = new Array(pts.length).fill(null);
    pts.forEach((p, i) => placed.push({ x: geo[i].x - r, y: geo[i].y - r, w: 2 * r, h: 2 * r }));
    order.forEach(i => {
      const t = short(pts[i]), w = t.length * fs * .56, h = fs + 2, { x, y } = geo[i];
      for (const [dx, dy, anchor] of [[r + 3, 4, 'start'], [-r - 3, 4, 'end'], [r + 2, -r - 3, 'start'], [r + 2, r + 11, 'start'], [-r - 2, -r - 3, 'end'], [-r - 2, r + 11, 'end']]) {
        const bx = anchor === 'start' ? x + dx : x + dx - w, by = y + dy - fs;
        if (bx < b.x || bx + w > b.x + b.w || by < b.y || by + h > b.y + b.h) continue;
        if (placed.some(q => bx < q.x + q.w && bx + w > q.x && by < q.y + q.h && by + h > q.y)) continue;
        placed.push({ x: bx, y: by, w, h }); labels[i] = { x: x + dx, y: y + dy, anchor, t }; break;
      }
    });
    pts.forEach((p, i) => {
      const a = alphaOf(p), ok = p.latMs <= budget, { x, y } = geo[i];
      const style = ok ? `fill="${fg}" stroke="${fg}"` : `fill="none" stroke="${mu}" stroke-width="1.6"`;
      s += `<g class="pt${S.focus === p.key ? ' on' : ''}" tabindex="0" data-key="${p.key}" opacity="${a.toFixed(3)}" aria-label="${esc(p.name + ' on ' + p.device + ': ' + p.latV + (p.latScale === 1000 ? ' s' : ' ms') + ', image AUROC ' + p.accShow + ', ' + p.proto)}"><g ${style}>${marker(p.proto, x, y, r)}</g>`;
      if (labels[i]) s += `<text x="${labels[i].x}" y="${labels[i].y}" fill="${ok ? fg : mu}" font-size="${fs}" text-anchor="${labels[i].anchor}">${esc(labels[i].t)}</text>`;
      s += `<rect x="${x - 9}" y="${y - 9}" width="18" height="18" fill="transparent"/></g>`;
    });
    s += '</svg>';
    host.innerHTML = s;
  }

  function draw(animate) {
    const pts = current(), xr = XR[S.view], yr = yRange(pts);
    if (anim) anim();
    const prev = new Map(shown.map(p => [p.key + p.view, p]));
    const same = shown.length && shown[0].view === S.view;
    if (!animate || !same) { shown = pts; render(pts, () => 1, xr, yr); say(); return; }
    const from = pts.map(p => prev.get(p.key + p.view) || p);
    anim = PE.animate(600, u => {
      const mid = pts.map((p, i) => ({ ...p, latMs: 10 ** PE.lerp(Math.log10(from[i].latMs), Math.log10(p.latMs), u), acc: PE.lerp(from[i].acc, p.acc, u) }));
      render(mid, () => 1, xr, PE.lerp(0, 1, u) < 1 ? [Math.min(yRange(from)[0], yr[0]), 100] : yr);
    }, () => { anim = null; shown = pts; render(pts, () => 1, xr, yr); });
    shown = pts; say();
  }

  function say(extra) {
    const pts = current(), budget = 1000 / pps(S.v), n = pts.filter(p => p.latMs <= budget).length;
    const where = S.view === 'desktop' ? `${VIEW.desktop}, ${GPU[S.gpu]}` : VIEW[S.view];
    let t = `${where}. At ${fmtPps(pps(S.v))} parts per second each part gets ${fmtMs(budget)} ms: ${n} of ${pts.length} plotted methods keep up.`;
    if (extra) t += ' ' + extra;
    out.textContent = t;
  }
  function describe(el) {
    if (!el) return;
    if (el.dataset.rug) { const r = rug()[+el.dataset.rug]; say(`${r.name}, ${r.device}: ${r.latV} ms. Source: ${r.src}.`); return; }
    const p = current().find(q => q.key === el.dataset.key); if (!p) return;
    S.focus = p.key; host.querySelectorAll('.pt').forEach(g => g.classList.toggle('on', g === el));
    say(`${p.name} on ${p.device}: ${p.latV}${p.latScale === 1000 ? ' s' : ' ms'}, image AUROC ${p.accShow} (${p.proto}). Source: ${p.src}.`);
  }
  host.addEventListener('focusin', e => describe(e.target.closest('.pt')));
  host.addEventListener('mouseover', e => { const g = e.target.closest('.pt'); if (g) describe(g); });
  host.addEventListener('click', e => { const g = e.target.closest('.pt'); if (g) g.focus(); });

  // GPU and accuracy buttons stay visible in every view (dimmed off the desktop view); pressing one selects the desktop view.
  function sync() {
    const d = S.view === 'desktop';
    ['desktop', 'jetson', 'cpu'].forEach(v => document.getElementById('btn-speed-' + v).setAttribute('aria-pressed', v === S.view));
    Object.keys(GPU).forEach(g => document.getElementById('btn-speed-gpu-' + g).setAttribute('aria-pressed', d && g === S.gpu));
    ['mvtec', 'mean'].forEach(a => document.getElementById('btn-speed-acc-' + a).setAttribute('aria-pressed', d && a === S.acc));
    document.getElementById('speed-gpus').classList.toggle('dim', !d); document.getElementById('speed-acc').classList.toggle('dim', !d);
    fig.querySelectorAll('tbody[data-view]').forEach(tb => { tb.hidden = !(tb.dataset.view === S.view && (!d || tb.dataset.sub === S.gpu)); });
  }
  function setView(v) { if (S.view === v && shown.length) return; S.view = v; S.focus = null; sync(); draw(true); }
  function setGpu(g) { const was = S.view; if (S.gpu === g && was === 'desktop') return; S.gpu = g; S.view = 'desktop'; sync(); draw(was === 'desktop'); }
  function setAcc(a) { const was = S.view; if (S.acc === a && was === 'desktop') return; S.acc = a; S.view = 'desktop'; sync(); draw(was === 'desktop'); }
  function setV(v) { S.v = v; rng.value = v; rOut.textContent = `${fmtPps(pps(v))} parts per second`; render(shown.length ? shown : current(), () => 1, XR[S.view], yRange(current())); say(); }
  ['desktop', 'jetson', 'cpu'].forEach(v => document.getElementById('btn-speed-' + v).addEventListener('click', () => setView(v)));
  Object.keys(GPU).forEach(g => document.getElementById('btn-speed-gpu-' + g).addEventListener('click', () => setGpu(g)));
  ['mvtec', 'mean'].forEach(a => document.getElementById('btn-speed-acc-' + a).addEventListener('click', () => setAcc(a)));
  rng.addEventListener('input', () => setV(+rng.value));
  document.getElementById('btn-speed-reset').addEventListener('click', () => {
    S.focus = null; Object.assign(S, { view: DEF.view, gpu: DEF.gpu, acc: DEF.acc }); sync(); draw(true); const a = S.v;
    PE.animate(500, u => setV(Math.round(PE.lerp(a, DEF.v, u))), () => setV(DEF.v));
  });
  new ResizeObserver(() => render(shown.length ? shown : current(), () => 1, XR[S.view], yRange(current()))).observe(host);
  PE.onTheme(() => render(shown.length ? shown : current(), () => 1, XR[S.view], yRange(current())));
  sync(); setV(DEF.v); draw(false);
})();
</script>
