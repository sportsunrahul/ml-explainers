<style>
figure.archflow .cf-row { display:flex; flex-wrap:wrap; gap:.3rem; margin:0 0 .5rem; padding-right:9rem; font:.84rem system-ui, sans-serif; align-items:center; }
figure.archflow .cf-row span { color:var(--muted); margin-right:.2rem; }
figure.archflow .cf-row button { font:inherit; color:var(--muted); background:var(--bg); border:1px solid var(--rule); border-radius:999px; padding:.2rem .55rem; cursor:pointer; }
figure.archflow .cf-row button[aria-pressed="true"] { color:var(--bg); background:var(--blue); border-color:var(--blue); }
figure.archflow .cf-row button.cf-back[aria-pressed="true"] { background:var(--red); border-color:var(--red); }
figure.archflow .cf-row button:focus-visible { outline:2px solid var(--yellow); outline-offset:2px; }
figure.archflow .cf-text { font:.95rem/1.55 system-ui, sans-serif; margin:.6rem 0 0; min-height:9em; }
figure.archflow .cf-text h4 { font-size:1rem; margin:0 0 .3rem; }
figure.archflow .cf-text p { margin:0 0 .5rem; }
@media (max-width: 560px) { figure.archflow .cf-row { padding-right:0; } figure.archflow .cf-row:first-of-type { margin-top:1.9rem; } }
</style>
<script>
/* ArchFlow: forward/backward architecture stepper driven by a per-chapter spec. Needs PE (canvas, animate, label, arrow, dot, mix, css, clamp, lerp).
   Spec:
   { prefix, grid: { cols, rows },                      // desktop grid; mobile transposes it (rows -> columns, cols -> rows)
     boxes: { id: { c, r, lane: 'blue'|'yellow'|'green'|'red'|'mixed', title: [lines], glyph, data?: true } },
     edges: [[from, to], ...],                          // forward direction; the loss box is the one with no outgoing edge
     stages: [{ key, show: [ids] } | { key, back: true } | { key, back: true, done: true }],
     variants: { key: config }, variant: 'default key', // buttons btn-<prefix>-<key>; may be empty
     shape(id, v) -> string, sub?(id, v) -> string,     // forward label under each box ('|' splits lines)
     params(id, v) -> string|null,                      // trainable parameter names, or null when the box has none
     frozen(id, v) -> bool,                             // frozen weights (brown)
     readout(stage, v) -> string }
   Variants change shapes, labels and frozen sets only, never the graph or stage list: mount one figure per
   differently-shaped training stage, and keep any second button row in the chapter's own code.
   Ids used: viz-<prefix>, <prefix>-canvas, <prefix>-readout, btn-<prefix>-<stage>, btn-<prefix>-<variant>, btn-<prefix>-prev|next|reset,
   and stage texts as <div data-stage="..."> inside .cf-text (one with data-stage="overview"). */
const ArchFlow = (() => {
  function mount(spec) {
    const P = spec.prefix, fig = document.getElementById('viz-' + P), el = document.getElementById(P + '-canvas'), out = document.getElementById(P + '-readout');
    const ORDER = spec.stages.map(s => s.key), ST = Object.fromEntries(spec.stages.map((s, i) => [s.key, { ...s, i }]));
    const chips = ORDER.map(k => document.getElementById('btn-' + P + '-' + k));
    const VK = Object.keys(spec.variants || {}), vBtns = VK.map(k => document.getElementById('btn-' + P + '-' + k));
    const texts = [...fig.querySelectorAll('.cf-text > div')];
    const ids = Object.keys(spec.boxes), B = spec.boxes;
    const S = { scene: 'overview', v: spec.variant, u: 1 };
    let stop = () => {};
    // First forward stage that shows each box.
    const firstStep = {}; spec.stages.forEach((s, i) => (s.show || []).forEach(id => { if (!(id in firstStep)) firstStep[id] = i; }));
    // Distance from the loss box(es) (no outgoing edge) along reversed edges: the order in which the backward pass reaches boxes.
    const dist = Object.fromEntries(ids.filter(id => !spec.edges.some(e => e[0] === id)).map(id => [id, 0])); let changed = true;
    while (changed) { changed = false; for (const [a, b] of spec.edges) if (b in dist && (!(a in dist) || dist[a] > dist[b] + 1)) { dist[a] = dist[b] + 1; changed = true; } }
    const maxD = Math.max(...Object.values(dist));
    const conf = () => (spec.variants || {})[S.v];
    const trainable = id => !B[id].data && spec.params(id, conf()) != null && !spec.frozen(id, conf());
    // A box needs a gradient if it, or anything upstream of it, has trainable weights.
    function needsGrad(id, seen = {}) { if (seen[id]) return false; seen[id] = 1; return trainable(id) || spec.edges.some(([a, b]) => b === id && needsGrad(a, seen)); }

    let C = {};
    const hex = (c, d) => (c && c.length === 7 ? c : d);
    const tint = (c, a) => PE.mix(hex(C.BG, '#16181d'), hex(c, '#888888'), a);
    function rr(ctx, x0, y0, w, h, r, fill, stroke, lw = 1.5, dash) {
      ctx.beginPath(); ctx.moveTo(x0 + r, y0); ctx.arcTo(x0 + w, y0, x0 + w, y0 + h, r); ctx.arcTo(x0 + w, y0 + h, x0, y0 + h, r);
      ctx.arcTo(x0, y0 + h, x0, y0, r); ctx.arcTo(x0, y0, x0 + w, y0, r); ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.save(); ctx.strokeStyle = stroke; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash); ctx.stroke(); ctx.restore(); }
    }
    const laneC = id => ({ blue: C.B, yellow: C.Y, green: C.G, red: C.Rd, mixed: C.F })[B[id].lane] || C.F;
    const lines = s => String(s).split('|');
    const labN = {};
    const titleN = id => B[id].title.length;

    function glyph(ctx, id, b, col) {
      const g = B[id].glyph || 'layers', cx = b.x + b.w / 2, cy = b.y + b.h / 2, seed = id.length * 7 + id.charCodeAt(0);
      ctx.save(); ctx.fillStyle = col; ctx.strokeStyle = col; ctx.lineWidth = 1.3;
      if (g === 'photos') for (let k = 2; k >= 0; k--) { const s = 24; rr(ctx, cx - s / 2 + k * 5 - 5, cy - s / 2 - k * 4 + 4, s, s, 3, tint(C.B, .25 + .15 * (2 - k)), C.B, 1.2); }
      else if (g === 'photo') { rr(ctx, cx - 14, cy - 13, 28, 26, 3, tint(C.B, .35), C.B, 1.2); ctx.fillStyle = C.B; ctx.beginPath(); ctx.arc(cx + 4, cy - 3, 5, 0, 7); ctx.fill(); }
      else if (g === 'grid') { const s = 6; for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) rr(ctx, cx - 10 + c * (s + 1), cy - 14 + r * (s + 1), s, s, 1, tint(C.B, .55), null);
        if (B[id].cls !== false) rr(ctx, cx - 22, cy - 14, s, s, 1, C.F, null); }
      else if (g === 'layers') for (let k = 0; k < 4; k++) rr(ctx, cx - 22, cy - 16 + k * 8, 44, 6, 2, tint(col, .3 + .15 * k), null);
      else if (g === 'vector' || g === 'unit') { for (let k = 0; k < 8; k++) { const h = g === 'unit' ? 9 : 5 + 10 * Math.abs(Math.sin(k * 1.7 + seed)); ctx.fillRect(cx - 20 + k * 5.2, cy + 8 - h, 3.6, h); }
        if (g === 'unit') { ctx.beginPath(); ctx.arc(cx + 30, cy + 2, 7, 0, 7); ctx.stroke(); } }
      else if (g === 'text') for (let k = 0; k < 3; k++) rr(ctx, cx - 22, cy - 12 + k * 9, 44 - k * 8, 5, 2, tint(C.Y, .6), null);
      else if (g === 'tokens') for (let k = 0; k < 5; k++) rr(ctx, cx - 24 + k * 10, cy - 8, 8, 12, 2, k === 0 || k === 4 ? C.F : tint(C.Y, .55), null);
      else if (g === 'seq') { const pat = B[id].pattern || 'yybbbbyyy'; const n = pat.length, w = Math.min(8, (b.w - 12) / n - 1.5);
        for (let k = 0; k < n; k++) rr(ctx, cx - n * (w + 1.5) / 2 + k * (w + 1.5), cy - 7, w, 12, 2, pat[k] === 'b' ? C.B : pat[k] === 'g' ? C.G : tint(C.Y, .7), null); }
      else if (g === 'latents') for (let k = 0; k < 6; k++) { ctx.fillStyle = tint(C.G, .45 + .1 * (k % 2)); ctx.beginPath(); ctx.arc(cx - 20 + k * 8, cy, 3.4, 0, 7); ctx.fill(); }
      else if (g === 'matrix') { const s = 7; for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) rr(ctx, cx - 15 + c * (s + 1), cy - 17 + r * (s + 1), s, s, 1, r === c ? C.G : tint(C.Br, .5), null); }
      else if (g === 'loss') { rr(ctx, cx - 22, cy - 4, 44, 8, 3, tint(C.Rd, .25), null); rr(ctx, cx - 22, cy - 4, 30, 8, 3, C.Rd, null); }
      else if (g === 'gate') { ctx.beginPath(); for (let k = 0; k <= 20; k++) { const xx = -1 + k / 10, X = cx + xx * 20, Y = cy - 12 * Math.tanh(2.2 * xx); k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); } ctx.lineWidth = 2; ctx.stroke(); }
      else if (g === 'mlp') { for (const [xx, n] of [[-16, 4], [0, 6], [16, 4]]) for (let k = 0; k < n; k++) { ctx.beginPath(); ctx.arc(cx + xx, cy - (n - 1) * 3.5 + k * 7, 2.4, 0, 7); ctx.fill(); } }
      else if (g === 'codebook') { const s = 6; for (let r = 0; r < 3; r++) for (let c = 0; c < 5; c++) rr(ctx, cx - 18 + c * (s + 1.5), cy - 11 + r * (s + 1.5), s, s, 1, tint(col, .3 + .6 * (((r * 5 + c) * 7) % 5) / 5), null); }
      ctx.restore();
    }

    function layout(W, H) {
      const R = {}, { cols, rows } = spec.grid, horiz = W >= 600;
      if (horiz) {
        const cw = (W - 16) / cols, bw = Math.min(cw - 16, 104), bh = 50;
        const rowY = r => rows === 1 ? (H - bh) / 2 + 6 : 34 + r * (H - bh - 72) / (rows - 1);
        ids.forEach(id => { const b = B[id], cx = 8 + cw * (b.c + .5); R[id] = { x: cx - bw / 2, y: rowY(b.r), w: bw, h: bh }; });
        return { R, horiz };
      }
      const colW = W / rows, bw = Math.min(colW - 14, 150), bh = 50, rh = (H - 40) / cols;
      ids.forEach(id => { const b = B[id], cx = colW * (b.r + .5); R[id] = { x: cx - bw / 2, y: 36 + rh * b.c, w: bw, h: bh }; });
      return { R, horiz };
    }
    const aspect = w => { const { cols, rows } = spec.grid; return w >= 600 ? (rows === 1 ? 150 : 72 + 50 + (rows - 1) * 114) / w : (40 + cols * 134) / w; };
    function edgePts(L, a, b) {
      const A = L.R[a], Bx = L.R[b], down = L.horiz ? B[a].c === B[b].c : B[a].r !== B[b].r && B[a].c === B[b].c;
      if (L.horiz && !down) return B[b].c > B[a].c ? [A.x + A.w, A.y + A.h / 2, Bx.x, Bx.y + Bx.h / 2] : [A.x, A.y + A.h / 2, Bx.x + Bx.w, Bx.y + Bx.h / 2];
      if (!L.horiz && B[a].c === B[b].c) return B[b].r > B[a].r ? [A.x + A.w, A.y + A.h / 2, Bx.x, Bx.y + Bx.h / 2] : [A.x, A.y + A.h / 2, Bx.x + Bx.w, Bx.y + Bx.h / 2];
      const upward = L.horiz ? B[b].r < B[a].r : false;
      if (upward) return [A.x + A.w / 2, A.y - 8 - 12 * titleN(a), Bx.x + Bx.w / 2, Bx.y + Bx.h + 7 + 11 * (labN[b] || 1)];
      return [A.x + A.w / 2, A.y + A.h + 7 + 11 * (labN[a] || 1), Bx.x + Bx.w / 2, Bx.y - 8 - 12 * titleN(b)];
    }

    const cv = PE.canvas(el, aspect, (ctx, W, H) => {
      C = { B: PE.css('--blue'), Y: PE.css('--yellow'), G: PE.css('--green'), Rd: PE.css('--red'), Br: PE.css('--brown'), F: PE.css('--fg'), Mu: PE.css('--muted'), BG: PE.css('--bg') };
      ctx.fillStyle = C.BG; ctx.fillRect(0, 0, W, H);
      const L = layout(W, H), v = conf(), k = S.scene === 'overview' ? -1 : ST[S.scene].i, back = S.scene !== 'overview' && ST[S.scene].back;
      const alphaOf = id => { if (S.scene === 'overview' || back) return 1; const s = firstStep[id] ?? 0; return s < k ? .7 : s === k ? .25 + .75 * S.u : .12; };
      const gradAt = id => { if (!back) return 0; if (ST[S.scene].done) return 1; const d = (dist[id] ?? maxD) / Math.max(1, maxD); return PE.clamp((S.u * 1.15 - d) / .15, 0, 1); };
      const labelOf = id => {
        const fz = !B[id].data && spec.frozen(id, v);
        if (back) {
          if (B[id].data) return ['data, no weights', C.Mu];
          if (trainable(id)) return [spec.params(id, v), C.Rd];
          if (fz) return [needsGrad(id) ? 'frozen: gradient|passes, no update' : 'frozen: gradient|stops here', C.Br];
          return [spec.shape(id, v), C.Mu];
        }
        return [(spec.sub && spec.sub(id, v)) || spec.shape(id, v), fz ? C.Br : C.F];
      };
      ids.forEach(id => { labN[id] = lines(labelOf(id)[0]).length; });
      for (const [a, b] of spec.edges) {
        const [x1, y1, x2, y2] = edgePts(L, a, b), al = Math.min(alphaOf(a), alphaOf(b)), col = laneC(a);
        PE.arrow(ctx, x1 + 2 * Math.sign(x2 - x1), y1, x2 - 2 * Math.sign(x2 - x1), y2, col, 1.6, 7, al * .8);
        if (!back && (firstStep[b] ?? 0) === k && S.u < 1) PE.dot(ctx, PE.lerp(x1, x2, S.u), PE.lerp(y1, y2, S.u), 4, col);
        if (back && needsGrad(a) && gradAt(b) > 0) {
          const vert = Math.abs(x2 - x1) < Math.abs(y2 - y1), off = vert ? [9, 0] : [0, 9], t = Math.max(gradAt(a), .35);
          PE.arrow(ctx, x2 + off[0], y2 + off[1], PE.lerp(x2, x1, t) + off[0], PE.lerp(y2, y1, t) + off[1], C.Rd, 2, 7, .45 + .55 * gradAt(a));
        }
      }
      for (const id of ids) {
        const b = L.R[id], al = alphaOf(id), col = laneC(id), g = gradAt(id), fz = !B[id].data && spec.frozen(id, v), cur = !back && k >= 0 && firstStep[id] === k;
        ctx.save(); ctx.globalAlpha = al;
        rr(ctx, b.x, b.y, b.w, b.h, 7, fz ? tint(C.Br, .22) : tint(col, cur ? .2 : .1), fz ? C.Br : col, cur ? 2.4 : 1.4, fz ? [4, 3] : null);
        if (g > 0 && trainable(id)) rr(ctx, b.x - 3, b.y - 3, b.w + 6, b.h + 6, 9, null, C.Rd, 2.2 * g);
        glyph(ctx, id, b, fz ? C.Br : col);
        const T = B[id].title; T.forEach((t, j) => PE.label(ctx, t, b.x + b.w / 2, b.y - 5 - (T.length - 1 - j) * 12, col, 10.5, 'center'));
        ctx.restore();
        ctx.save(); ctx.globalAlpha = Math.max(al, .12);
        const [lab, lc] = labelOf(id);
        lines(lab).forEach((t, j) => PE.label(ctx, t, b.x + b.w / 2, b.y + b.h + 12 + 11 * j, lc, 10, 'center'));
        ctx.restore();
      }
    });
    function show() {
      texts.forEach(d => { d.hidden = d.dataset.stage !== S.scene; });
      chips.forEach((c, i) => c.setAttribute('aria-pressed', String(ORDER[i] === S.scene)));
      vBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(VK[i] === S.v)));
      out.textContent = spec.readout(S.scene, conf());
    }
    function go(scene) { stop(); S.scene = scene; S.u = 0; show(); stop = PE.animate(ST[scene].back && !ST[scene].done ? 2200 : 800, u => { S.u = u; cv.redraw(); }); }
    const idx = () => ORDER.indexOf(S.scene);
    ORDER.forEach((k, i) => chips[i].addEventListener('click', () => go(k)));
    VK.forEach((k, i) => vBtns[i].addEventListener('click', () => { S.v = k; show(); cv.redraw(); }));
    document.getElementById('btn-' + P + '-next').addEventListener('click', () => go(ORDER[Math.min(ORDER.length - 1, idx() + 1)]));
    document.getElementById('btn-' + P + '-prev').addEventListener('click', () => { const i = idx(); if (i <= 0) { stop(); S.scene = 'overview'; S.u = 1; show(); cv.redraw(); } else go(ORDER[i - 1]); });
    document.getElementById('btn-' + P + '-reset').addEventListener('click', () => { stop(); S.scene = 'overview'; S.v = spec.variant; S.u = 1; show(); cv.redraw(); });
    show();
    return { go, state: S };
  }
  return { mount };
})();
</script>
