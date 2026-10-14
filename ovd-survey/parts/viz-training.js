<script>
/* Interlude: one Grounding DINO training step on one GPU (2 images). Image sizes follow OV-DINO's [800, 1333];
   feature maps follow Grounding DINO T's strides (8, 16, 32 and a stride-2 copy for 64); map sizes round up
   (our reading). Scenes, names, heat, predictions and losses are toys; the matching is exact on the toys. */
(() => {
  const el = document.getElementById('train-canvas'), out = document.getElementById('train-shape');
  const fig = document.getElementById('viz-train-step');
  const chips = [...fig.querySelectorAll('#train-stages button')], texts = [...fig.querySelectorAll('.train-text > div')];
  const NS = chips.length, B = 2, NQ = 900, NL = 6, SHOWN = 12, SHORT = 800, LONG = 1333;
  const ASPECTS = [4 / 3, 3 / 4, 16 / 9, 1, 3 / 2];
  const PRESENT = [['person', 'person'], ['forklift', 'forklift'], ['pallet', 'pallet'], ['traffic cone', 'cone'], ['crate', 'crate']];
  const NEG = ['helmet', 'ladder', 'truck', 'hand truck', 'shelf', 'person', 'forklift', 'pallet', 'traffic cone', 'crate'];
  const S = { stage: 0, t: 1, seed: 1, running: false };
  let stopA = () => {}, runTimer = 0, batch;
  const fmt = n => n.toLocaleString('en-US');
  const C = () => DK.C();

  // boxes are { x1, y1, x2, y2 } in [0, 1] image coordinates
  const fromC = (cx, cy, w, h) => ({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  const cxcywh = b => [(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, b.x2 - b.x1, b.y2 - b.y1];
  const area = b => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  function iou(a, b) { const i = area({ x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1), x2: Math.min(a.x2, b.x2), y2: Math.min(a.y2, b.y2) }); return i / (area(a) + area(b) - i || 1); }
  function giou(a, b) {
    const i = area({ x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1), x2: Math.min(a.x2, b.x2), y2: Math.min(a.y2, b.y2) }), u = area(a) + area(b) - i;
    const c = area({ x1: Math.min(a.x1, b.x1), y1: Math.min(a.y1, b.y1), x2: Math.max(a.x2, b.x2), y2: Math.max(a.y2, b.y2) });
    return i / u - (c - u) / c;
  }
  // Exact assignment of m objects to distinct predictions by search; m <= 3 and 12 predictions keep it tiny.
  function assign(cost, m) {
    let best = Infinity, pick = null;
    const go = (j, used, sum, cur) => { if (sum >= best) return; if (j === m) { best = sum; pick = cur.slice(); return; }
      for (let i = 0; i < cost.length; i++) if (!used.has(i)) { used.add(i); cur.push(i); go(j + 1, used, sum + cost[i][j], cur); cur.pop(); used.delete(i); } };
    go(0, new Set(), 0, []); return pick; // pick[j] = prediction for object j
  }
  const words = n => n.split(' ').length;
  const shuffle = (a, r) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  function makeBatch(seed) {
    const r = PE.rng(seed * 7919 + 3), imgs = [];
    for (let i = 0; i < B; i++) {
      let ar = ASPECTS[Math.floor(r() * ASPECTS.length)];
      while (i > 0 && ar === imgs[0].ar) ar = ASPECTS[Math.floor(r() * ASPECTS.length)];
      const q = Math.max(ar, 1 / ar), s = SHORT * q > LONG ? LONG / q : SHORT; // short side 800 unless the long side would pass 1333
      const h = Math.round(ar >= 1 ? s : s / ar), w = Math.round(ar >= 1 ? s * ar : s);
      const kinds = shuffle(PRESENT.slice(), r).slice(0, i === 0 ? 2 + Math.floor(r() * 2) : 1 + Math.floor(r() * 3)), objs = [];
      kinds.forEach(([name, kind]) => {
        for (let tries = 0; tries < 60; tries++) {
          const bw = .16 + r() * .22, bh = (kind === 'person' || kind === 'cone' ? 1.6 : .9) * bw * (ar >= 1 ? ar : 1) * (.8 + .4 * r());
          if (bh > .8) continue;
          const b = fromC(bw / 2 + r() * (1 - bw), bh / 2 + r() * (1 - bh), bw, bh);
          if (objs.every(o => iou(o.b, b) < .02 && !(b.x1 < o.b.x2 && b.x2 > o.b.x1 && b.y1 < o.b.y2 && b.y2 > o.b.y1))) { objs.push({ name, kind, b }); break; }
        }
      });
      if (!objs.length) objs.push({ name: kinds[0][0], kind: kinds[0][1], b: fromC(.5, .55, .3, .4) });
      const here = objs.map(o => o.name), negs = shuffle(NEG.filter(n => !here.includes(n)), r).slice(0, 2 + Math.floor(r() * 3));
      const prompt = shuffle([...here.map(n => ({ n, pos: true })), ...negs.map(n => ({ n, pos: false }))], r);
      const nt = 2 + prompt.reduce((a, p) => a + words(p.n) + 1, 0); // [CLS] + words and one '.' per name + [SEP]
      imgs.push({ ar, h, w, objs, prompt, nt });
    }
    const H0 = Math.max(...imgs.map(i => i.h)), W0 = Math.max(...imgs.map(i => i.w));
    const maps = [8, 16, 32].map(s => [Math.ceil(H0 / s), Math.ceil(W0 / s)]); maps.push(maps[2].map(v => Math.ceil(v / 2)));
    const NI = maps.reduce((a, [p, q]) => a + p * q, 0), NT = Math.max(...imgs.map(i => i.nt));
    // Toy heat on image 1's stride-32 cells: overlap with the objects it names, plus noise; top cells seed queries.
    const im = imgs[0], gh = Math.ceil(im.h / 32), gw = Math.ceil(im.w / 32), cells = [];
    for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
      const cb = { x1: x * 32 / im.w, y1: y * 32 / im.h, x2: Math.min(1, (x + 1) * 32 / im.w), y2: Math.min(1, (y + 1) * 32 / im.h) };
      let best = 0, own = -1; im.objs.forEach((o, k) => { const ov = area({ x1: Math.max(cb.x1, o.b.x1), y1: Math.max(cb.y1, o.b.y1), x2: Math.min(cb.x2, o.b.x2), y2: Math.min(cb.y2, o.b.y2) }) / (area(cb) || 1); if (ov > best) { best = ov; own = k; } });
      cells.push({ x, y, cb, own: best > .3 ? own : -1, heat: PE.clamp(.12 + .78 * best + .12 * (r() - .5), 0, 1) });
    }
    const sel = cells.slice().sort((a, b) => b.heat - a.heat).slice(0, SHOWN);
    // Toy anchors and final predictions for the drawn queries; the first query on an object is the good one.
    const seen = {}, P = sel.map(c => {
      const [cx, cy, cw, ch] = cxcywh(c.cb), dflt = fromC(cx, cy, cw * 2.2, ch * 2.2);
      if (c.own < 0) return { cell: c, anchor: dflt, b: fromC(cx + (r() - .5) * .04, cy + (r() - .5) * .04, cw * 2, ch * 2), k: -1, p: .02 + .08 * r() };
      const o = im.objs[c.own].b, first = !seen[c.own], j = first ? .025 : .1; seen[c.own] = 1;
      const [ox, oy, ow, oh] = cxcywh(o), anchor = fromC(PE.lerp(cx, ox, .5), PE.lerp(cy, oy, .5), PE.lerp(cw * 2.2, ow, .5), PE.lerp(ch * 2.2, oh, .5));
      return { cell: c, anchor, b: fromC(ox + (r() - .5) * j, oy + (r() - .5) * j, ow * (1 + (r() - .5) * j * 2), oh * (1 + (r() - .5) * j * 2)), k: c.own, p: first ? .7 + .25 * r() : .25 + .25 * r() };
    });
    const l1 = (a, b) => { const A = cxcywh(a), Bc = cxcywh(b); return A.reduce((s, v, i) => s + Math.abs(v - Bc[i]), 0); };
    const costOf = wc => P.map(p => im.objs.map((o, j) => -wc * (p.k === j ? p.p : .03) + 5 * l1(p.b, o.b) + 2 * (1 - giou(p.b, o.b))));
    const cost = costOf(2), pick2 = assign(cost, im.objs.length), pick1 = assign(costOf(1), im.objs.length);
    const same = pick2.every((v, j) => v === pick1[j]);
    const losses = Array.from({ length: NL + 1 }, (_, l) => { const f = 1 - l * .09; return [(.5 + r() * .15) * f, (.35 + r() * .1) * f, (.3 + r() * .1) * f]; });
    const links = Array.from({ length: 60 }, () => [r(), r(), r()]);
    return { imgs, H0, W0, maps, NI, NT, cells, gh, gw, sel, P, cost, pick: pick2, same, losses, links };
  }

  const txt = (ctx, s, x, y, col, size = 11, weight = 600, align = 'left') => {
    ctx.save(); ctx.font = `${weight} ${size}px system-ui, sans-serif`; ctx.fillStyle = col; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); ctx.restore(); };
  const tw = (ctx, s, size, weight = 600) => { ctx.save(); ctx.font = `${weight} ${size}px system-ui, sans-serif`; const v = ctx.measureText(s).width; ctx.restore(); return v; };
  function hatch(ctx, x, y, w, h, k) {
    if (w <= 0 || h <= 0) return;
    ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); ctx.fillStyle = k.Br; ctx.globalAlpha = .14; ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = .45; ctx.strokeStyle = k.Br; ctx.lineWidth = 1;
    for (let d = -h; d < w; d += 7) { ctx.beginPath(); ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); ctx.stroke(); } ctx.restore();
  }
  // The padded H0 x W0 frame fitted into (bx, by, bw, bh); returns where image `im` sits inside it.
  function frame(ctx, k, bx, by, bw, bh, im, padA = 1) {
    const sc = Math.min(bw / batch.W0, bh / batch.H0), fw = batch.W0 * sc, fh = batch.H0 * sc, x = bx + (bw - fw) / 2, y = by;
    ctx.save(); ctx.strokeStyle = k.rule; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, fw - 1, fh - 1); ctx.restore();
    const iw = im.w * sc, ih = im.h * sc; ctx.fillStyle = k.p2; ctx.fillRect(x, y, iw, ih);
    if (padA > 0) { ctx.save(); ctx.globalAlpha = padA; hatch(ctx, x + iw, y, fw - iw, fh, k); hatch(ctx, x, y + ih, iw, fh - ih, k); ctx.restore(); }
    return { x, y, fw, fh, iw, ih, sc };
  }
  const R = (f, b) => [f.x + b.x1 * f.iw, f.y + b.y1 * f.ih, (b.x2 - b.x1) * f.iw, (b.y2 - b.y1) * f.ih];
  function objects(ctx, k, f, im, labels = true) {
    im.objs.forEach(o => { const [x, y, w, h] = R(f, o.b); DK.OBJ[o.kind](ctx, x + w * .08, y + h * .06, w * .84, h * .88, k);
      ctx.save(); ctx.strokeStyle = k.fg; ctx.globalAlpha = .7; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, w, h); ctx.restore();
      if (labels) txt(ctx, o.name, x + 2, y - 7, k.Y, 10.5); });
  }
  // Prompt chips; positives solid, negatives dashed. Returns the y below the last row.
  function promptChips(ctx, k, im, x0, y0, maxW, fs = 11) {
    let x = x0, y = y0; const hgt = fs * 1.7;
    im.prompt.forEach((p, i) => {
      const w = tw(ctx, p.n, fs) + fs * 1.2; if (x + w > x0 + maxW && x > x0) { x = x0; y += hgt + 4; }
      DK.rr(ctx, x, y, w, hgt, hgt / 2); ctx.save();
      if (p.pos) { ctx.fillStyle = DK.mixh(k.bg, k.Y, .22); ctx.fill(); } else ctx.setLineDash([3, 2]);
      ctx.strokeStyle = k.Y; ctx.globalAlpha = p.pos ? 1 : .75; ctx.lineWidth = 1.2; ctx.stroke(); ctx.restore();
      txt(ctx, p.n, x + w / 2, y + hgt / 2 + .5, k.Y, fs, p.pos ? 700 : 500, 'center'); x += w + 3;
      if (i < im.prompt.length - 1) { txt(ctx, '.', x + 1, y + hgt / 2, k.mu, fs + 1, 700); x += 8; }
    });
    return y + hgt;
  }
  function arrowR(ctx, x1, y1, x2, y2, col) { PE.arrow(ctx, x1, y1, x2, y2, col, 1.5, 8); }
  const narrowOf = W => W < 560;

  const DRAW = [
    (ctx, W, H, t, k) => { // batch and prompts
      const nar = narrowOf(W), gap = 14, cols = nar ? 1 : 2, cw = (W - gap * (cols - 1)) / cols, rh = H / (nar ? 2 : 1);
      batch.imgs.forEach((im, i) => { const bx = nar ? 0 : i * (cw + gap), by = nar ? i * rh : 0;
        const f = frame(ctx, k, bx, by + 14, cw, rh - (nar ? 78 : 88), im, PE.clamp(t * 2 - .5, 0, 1)); objects(ctx, k, f, im);
        txt(ctx, `img ${i + 1}: ${im.h} x ${im.w}, padded to ${batch.H0} x ${batch.W0}`, f.x, by + 5, k.fg, 11);
        ctx.save(); ctx.globalAlpha = PE.clamp(t * 2 - .8, 0, 1); promptChips(ctx, k, im, bx, f.y + f.fh + 10, cw, nar ? 10 : 11); ctx.restore(); });
    },
    (ctx, W, H, t, k) => { // Swin
      const nar = narrowOf(W), im = batch.imgs[0], f = frame(ctx, k, 0, 18, nar ? W : W * .3, nar ? H * .36 : H - 40, im); objects(ctx, k, f, im, false);
      txt(ctx, 'img 1 (padded)', f.x, 8, k.mu, 11);
      const x0 = nar ? 4 : f.x + f.fw + 40, y0 = nar ? f.y + f.fh + 40 : 40, aw = W - x0 - 6, ah = nar ? H - y0 - 40 : H - 90;
      const [h8, w8] = batch.maps[0], sc = Math.min(aw * .5 / w8, ah * .62 / h8), labs = ['1/8', '1/16', '1/32', '1/64'];
      if (!nar) arrowR(ctx, f.x + f.fw + 6, y0 + h8 * sc / 2, x0 - 8, y0 + h8 * sc / 2, k.B);
      let x = x0;
      batch.maps.forEach(([h, w], i) => { const a = PE.clamp(t * 5 - i, 0, 1), rw = Math.max(6, w * sc), rh = Math.max(6, h * sc);
        ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = DK.mixh(k.bg, k.B, .2); ctx.fillRect(x, y0, rw, rh); ctx.strokeStyle = k.B; ctx.lineWidth = 1.3; ctx.strokeRect(x + .5, y0 + .5, rw, rh); ctx.restore();
        txt(ctx, labs[i], x, y0 + rh + 12, k.B, 11); txt(ctx, `${h} x ${w}`, x, y0 + rh + 26, k.mu, 10.5, 500); x += Math.max(rw, 44) + 14; });
      const ty = Math.min(H - 12, y0 + h8 * sc + 70);
      txt(ctx, '1/64: a strided copy of the 1/32 map', x0, ty - 18, k.mu, 10.5, 500);
      txt(ctx, `${fmt(batch.NI)} image tokens per image after flattening`, x0, ty, k.B, 12);
    },
    (ctx, W, H, t, k) => { // BERT
      const nar = narrowOf(W), im = batch.imgs[0], fs = nar ? 10 : 11, toks = ['[CLS]'];
      im.prompt.forEach(p => { p.n.split(' ').forEach(w => toks.push(w)); toks.push('.'); }); toks.push('[SEP]');
      txt(ctx, `prompt 1 as ${toks.length} tokens (toy tokenizer)`, 0, 9, k.mu, 11);
      let x = 0, y = 22; const hgt = fs * 1.7;
      toks.forEach((s, i) => { const w = tw(ctx, s, fs) + 10, a = PE.clamp(t * 3 - i / toks.length * 2, 0, 1); if (x + w > W) { x = 0; y += hgt + 4; }
        const special = s === '[CLS]' || s === '[SEP]' || s === '.';
        ctx.save(); ctx.globalAlpha = a; DK.rr(ctx, x, y, w, hgt, 4); ctx.strokeStyle = special ? k.mu : k.Y; ctx.lineWidth = 1; ctx.stroke(); ctx.restore();
        txt(ctx, s, x + w / 2, y + hgt / 2 + .5, special ? k.mu : k.Y, fs, 600, 'center'); x += w + 3; });
      // mask over the name words only
      const wl = [], gi = []; im.prompt.forEach((p, g) => p.n.split(' ').forEach(w => { wl.push(w); gi.push(g); }));
      const my = y + hgt + 34, room = H - my - 26, n = wl.length, cs = Math.min(room / n, (nar ? W - 80 : W * .45 - 80) / n, 30), mx = 76;
      txt(ctx, 'sub-sentence mask (name words only)', 0, my - 16, k.fg, 11.5);
      wl.forEach((w, i) => { txt(ctx, w, mx - 6, my + (i + .5) * cs, k.Y, Math.min(10.5, cs * .6 + 4), 600, 'right');
        wl.forEach((_, j) => { const ok = gi[i] === gi[j], a = PE.clamp(t * 2 - .4, 0, 1);
          ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = ok ? DK.mixh(k.bg, k.Y, .55) : k.p2; ctx.fillRect(mx + j * cs + 1, my + i * cs + 1, cs - 2, cs - 2);
          if (!ok) hatch(ctx, mx + j * cs + 1, my + i * cs + 1, cs - 2, cs - 2, k); ctx.restore(); }); });
      if (!nar) { const ax = mx + n * cs + 30, ay = my + n * cs / 2; arrowR(ctx, ax, ay, ax + 60, ay, k.Y);
        txt(ctx, `BERT-base, then ${batch.imgs[0].nt} x 256`, ax + 70, ay - 8, k.Y, 12); txt(ctx, 'one feature per token', ax + 70, ay + 10, k.mu, 11, 500); }
      else txt(ctx, `BERT-base, then ${batch.imgs[0].nt} x 256 per prompt`, 0, H - 10, k.Y, 11.5);
    },
    (ctx, W, H, t, k) => { // enhancer
      const nar = narrowOf(W), im = batch.imgs[0], gx = 10, gy = 7, cell = Math.min((W * (nar ? .5 : .36)) / gx, (H - 70) / gy), ix = 4, iy = 30;
      txt(ctx, 'image tokens (sample)', ix, 12, k.B, 11.5);
      for (let r = 0; r < gy; r++) for (let c = 0; c < gx; c++) { ctx.fillStyle = k.B; ctx.globalAlpha = .35 + .4 * ((r * 7 + c * 3) % 5) / 5; ctx.fillRect(ix + c * cell + 1, iy + r * cell + 1, cell - 2, cell - 2); }
      ctx.globalAlpha = 1;
      const wl = []; im.prompt.forEach(p => p.n.split(' ').forEach(w => wl.push(w)));
      const tx = nar ? W - 92 : W * .62, ch = Math.min(22, (H - 60) / wl.length);
      txt(ctx, 'text tokens', tx, 12, k.Y, 11.5);
      wl.forEach((w, i) => { const y = iy + i * ch; DK.rr(ctx, tx, y, 86, ch - 4, (ch - 4) / 2); ctx.strokeStyle = k.Y; ctx.lineWidth = 1; ctx.stroke(); txt(ctx, w, tx + 43, y + (ch - 4) / 2, k.Y, 10, 600, 'center'); });
      const n = Math.round(batch.links.length * t);
      batch.links.slice(0, n).forEach(([u, v, a], i) => { const c = Math.floor(u * gx), r = Math.floor(v * gy), wi = Math.floor(a * wl.length);
        const x1 = ix + (c + .5) * cell, y1 = iy + (r + .5) * cell, x2 = tx, y2 = iy + wi * ch + (ch - 4) / 2;
        ctx.save(); ctx.strokeStyle = i % 2 ? k.B : k.Y; ctx.globalAlpha = .18 + .3 * a; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore(); });
      const lx = nar ? 4 : tx + 104, ly = nar ? iy + gy * cell + 22 : 36, L = [['self-attention: deformable (image),', k.fg], ['ordinary (text)', k.fg], ['image-to-text cross-attention', k.B], ['text-to-image cross-attention', k.Y], ['FFN on each side', k.fg], [`x 6 layers (Fig. 3 order)`, k.fg]];
      L.forEach(([s, c], i) => txt(ctx, s, lx, ly + i * 16, c, 11, i === 5 ? 700 : 500));
    },
    (ctx, W, H, t, k) => { // query selection
      const nar = narrowOf(W), im = batch.imgs[0], f = frame(ctx, k, 0, 18, nar ? W : W * .6, nar ? H * .7 : H - 36, im, 0); objects(ctx, k, f, im, false);
      const cw = 32 * f.sc;
      batch.cells.forEach(c => { ctx.fillStyle = k.G; ctx.globalAlpha = (.05 + .5 * Math.pow(c.heat, 1.6)) * PE.clamp(t * 2, 0, 1); ctx.fillRect(f.x + c.x * cw + .5, f.y + c.y * cw + .5, Math.min(cw, f.iw - c.x * cw) - 1, Math.min(cw, f.ih - c.y * cw) - 1); });
      ctx.globalAlpha = 1;
      batch.P.forEach((p, i) => { const a = PE.clamp(t * 2.5 - 1 - i / SHOWN, 0, 1); if (a <= 0) return; const c = p.cell;
        ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = k.G; ctx.lineWidth = 2; ctx.strokeRect(f.x + c.x * cw + 1.5, f.y + c.y * cw + 1.5, cw - 3, cw - 3); ctx.restore();
        const [x, y, w, h] = R(f, p.anchor); DK.brackets(ctx, x, y, w, h, k.B, { lw: 1.3, alpha: a * .85, len: .22 }); });
      txt(ctx, 'heat: best score over prompt 1, on the 1/32 cells (toy)', f.x, 8, k.mu, 11, 500);
      const lx = nar ? 4 : f.x + f.fw + 16, ly = nar ? f.y + f.fh + 18 : 40;
      [[`top 900 of ${fmt(batch.NI)} tokens`, k.G], [`drawn: the best ${SHOWN} cells`, k.mu], ['blue: anchor boxes', k.B], ['content queries: learned', k.Br]].forEach(([s, c], i) => txt(ctx, s, lx, ly + i * 17, c, 11.5));
    },
    (ctx, W, H, t, k) => { // decoder
      const nar = narrowOf(W), im = batch.imgs[0], cy = promptChips(ctx, k, im, 0, 2, nar ? W : W * .7, 10.5), f = frame(ctx, k, 0, cy + 12, nar ? W : W * .7, H - cy - (nar ? 64 : 22), im, 0); objects(ctx, k, f, im, false);
      const layer = Math.min(NL, 1 + Math.floor(t * NL)), u = PE.clamp(t, 0, 1);
      batch.P.forEach((p, i) => { const b = { x1: PE.lerp(p.anchor.x1, p.b.x1, u), y1: PE.lerp(p.anchor.y1, p.b.y1, u), x2: PE.lerp(p.anchor.x2, p.b.x2, u), y2: PE.lerp(p.anchor.y2, p.b.y2, u) };
        const [x, y, w, h] = R(f, b); DK.brackets(ctx, x, y, w, h, k.B, { lw: 1 + 1.6 * p.p, alpha: .3 + .7 * p.p, len: .22 });
        if (i < 4) { ctx.save(); ctx.strokeStyle = k.Y; ctx.globalAlpha = .25; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x + w / 2, y); ctx.lineTo(x + w / 2 + (i - 2) * 20, cy + 2); ctx.stroke(); ctx.restore(); } });
      const lx = nar ? 4 : f.x + f.fw + 16, ly = nar ? f.y + f.fh + 16 : cy + 30;
      [[`decoder layer ${layer} of ${NL}`, k.fg], ['boxes move toward objects', k.B], ['dotted: text cross-attention', k.Y]].forEach(([s, c], i) => txt(ctx, s, lx, ly + i * 17, c, 11.5));
    },
    (ctx, W, H, t, k) => { // matching
      const nar = narrowOf(W), im = batch.imgs[0], n = im.objs.length, rows = SHOWN, top = 26, lx = 30;
      const mh = nar ? H * .42 : H - 54, ch = (mh - top) / rows, cwid = Math.min(nar ? 60 : 64, (nar ? W - 40 : W * .3) / n);
      txt(ctx, 'query', 0, 12, k.B, 10.5);
      im.objs.forEach((o, j) => txt(ctx, o.name, lx + j * cwid + 2, 12, k.Y, 10, 600));
      const vals = batch.cost.flat(), cmin = Math.min(...vals), cmax = Math.max(...vals);
      for (let i = 0; i < rows; i++) { ctx.globalAlpha = 1; txt(ctx, `q${i + 1}`, 0, top + (i + .5) * ch, k.B, 10, 600);
        for (let j = 0; j < n; j++) { const sel = batch.pick[j] === i && t > .35;
          ctx.fillStyle = sel ? k.G : k.R; ctx.globalAlpha = sel ? .95 : .08 + .6 * (1 - (batch.cost[i][j] - cmin) / (cmax - cmin || 1));
          ctx.fillRect(lx + j * cwid, top + i * ch + 1, cwid - 3, ch - 2); } }
      ctx.globalAlpha = 1;
      txt(ctx, 'red: cost, darker = cheaper', 0, top + rows * ch + 12, k.mu, 10, 500); txt(ctx, 'green: assigned', 0, top + rows * ch + 26, k.G, 10, 600);
      const fx = nar ? 0 : Math.max(lx + n * cwid + 24, 180), fy = nar ? mh + 40 : 18, fw = nar ? W : W - fx, fh = nar ? H - mh - 128 : H - 110;
      const f = frame(ctx, k, fx, fy, fw, fh, im, 0); objects(ctx, k, f, im);
      batch.P.forEach((p, i) => { const j = batch.pick.indexOf(i), hit = j >= 0 && t > .35, [x, y, w, h] = R(f, p.b);
        DK.brackets(ctx, x, y, w, h, hit ? k.G : k.Br, { lw: hit ? 2.4 : 1, alpha: hit ? 1 : .5, dash: hit ? null : [3, 3], len: .22 }); });
      // token targets for one matched and one unmatched query
      const toks = []; im.prompt.forEach((p, g) => p.n.split(' ').forEach(w => toks.push({ w, g })));
      const j0 = 0, q0 = batch.pick[j0], g0 = im.prompt.findIndex(p => p.n === im.objs[j0].name), qx = batch.P.findIndex((_, i) => !batch.pick.includes(i));
      const sy = f.y + f.fh + 14, cs = Math.min(18, (fw - 150) / toks.length);
      [[`q${q0 + 1} (matched to ${im.objs[j0].name})`, tk => tk.g === g0], [`q${qx + 1} (unmatched)`, () => false]].forEach(([lab, on], r) => {
        const y = sy + r * (cs + 18); txt(ctx, `target of ${lab}`, fx, y + 4, k.mu, 10.5, 500);
        toks.forEach((tk, i) => { const v = on(tk) && t > .35; ctx.fillStyle = v ? k.G : k.p2; ctx.fillRect(fx + i * cs, y + 11, cs - 2, cs - 4);
          txt(ctx, v ? '1' : '0', fx + i * cs + (cs - 2) / 2, y + 11 + (cs - 4) / 2, v ? k.bg : k.mu, 9.5, 600, 'center'); }); });
    },
    (ctx, W, H, t, k) => { // loss and update
      const lx = 70, top = 20, bh = Math.min(16, (H - 120) / (NL + 3)), tot = batch.losses.reduce((a, l) => a.map((v, i) => v + l[i]), [0, 0, 0]);
      const scale = (W - lx - 20) / (tot[0] + tot[1] + tot[2]);
      const seg = (y, vals) => { let x = lx; vals.forEach((v, i) => { ctx.save(); ctx.fillStyle = k.R; ctx.globalAlpha = [.9, .55, .3][i]; ctx.fillRect(x, y, v * scale * t, bh); ctx.restore(); x += v * scale * t; }); };
      batch.losses.forEach((l, i) => { txt(ctx, i === 0 ? 'encoder' : `layer ${i}`, 0, top + i * (bh + 5) + bh / 2, k.fg, 10.5); seg(top + i * (bh + 5), l); });
      const ty = top + (NL + 1) * (bh + 5) + 6; txt(ctx, 'total', 0, ty + bh / 2, k.R, 11.5, 700); seg(ty, tot);
      ['class (focal)', 'L1 x 5.0', 'GIoU x 2.0'].forEach((s, i) => { const x = lx + i * 96; ctx.save(); ctx.fillStyle = k.R; ctx.globalAlpha = [.9, .55, .3][i]; ctx.fillRect(x, ty + bh + 8, 11, 11); ctx.restore(); txt(ctx, s, x + 15, ty + bh + 14, k.fg, 10.5, 500); });
      if (t > .5) {
        const flow = ['loss', 'heads', 'decoder', 'enhancer', 'Swin, BERT', 'AdamW step'], wide = W >= 560, per = wide ? 6 : 3;
        const y0 = ty + bh + 38, hgt = 24, sw = (W - 8) / per, bw = sw - 22;
        const at = i => { const r = Math.floor(i / per), c = i % per, col = r % 2 === 0 ? per - 1 - c : c; return { x: 4 + col * sw, y: y0 + r * (hgt + 22) }; };
        ctx.save(); ctx.globalAlpha = PE.clamp((t - .5) * 2, 0, 1);
        flow.forEach((s, i) => { const p = at(i); ctx.strokeStyle = i === flow.length - 1 ? k.G : k.R; ctx.lineWidth = 1.3; ctx.strokeRect(p.x + .5, p.y + .5, bw, hgt);
          txt(ctx, s, p.x + 6, p.y + hgt / 2, i === flow.length - 1 ? k.G : k.fg, 11);
          if (i > 0) { const q = at(i - 1);
            if (q.y === p.y) { const dir = Math.sign(p.x - q.x); arrowR(ctx, dir < 0 ? q.x - 2 : q.x + bw + 2, p.y + hgt / 2, dir < 0 ? p.x + bw + 4 : p.x - 4, p.y + hgt / 2, k.R); }
            else arrowR(ctx, q.x + bw / 2, q.y + hgt + 2, p.x + bw / 2, p.y - 3, k.R); } });
        const last = at(flow.length - 1);
        txt(ctx, 'gradients reach every part with weights, both backbones too', 4, last.y + hgt + 16, k.R, 11);
        txt(ctx, '(no part is frozen: our reading of Table 8)', 4, last.y + hgt + 32, k.mu, 10.5, 500); ctx.restore(); }
    },
  ];

  const cv = PE.canvas(el, w => (w < 560 ? 1.15 : .45), (ctx, W, H) => { if (!W) return; const k = C(); ctx.fillStyle = k.bg; ctx.fillRect(0, 0, W, H); DRAW[S.stage](ctx, W, H, S.t, k); });

  function readout() {
    const b = batch, [i1, i2] = b.imgs, pad = 1 - b.imgs.reduce((a, i) => a + i.h * i.w, 0) / (B * b.H0 * b.W0), m = i1.objs.length;
    const tot = b.imgs.reduce((a, i) => a + i.objs.length, 0), L = ['1/8', '1/16', '1/32', '1/64'];
    out.textContent = [
      `GPU batch 2 x 3 x ${b.H0} x ${b.W0} after padding (${Math.round(pad * 100)}% of its pixels are padding). Prompts: ${i1.nt} and ${i2.nt} tokens, padded to 2 x ${b.NT} (our reading). Objects: ${i1.objs.length} and ${i2.objs.length}.`,
      `Swin-T maps for the padded batch: ${b.maps.map(([h, w], i) => `${L[i]} ${h} x ${w}`).join(', ')}. Flattened and joined: N_I = ${fmt(b.NI)} tokens per image, 2 x ${fmt(b.NI)} x 256.`,
      `Text: 2 x ${b.NT} tokens -> BERT-base features 2 x ${b.NT} x 768 (BERT-base's width, not stated in this paper) -> 2 x ${b.NT} x 256. One ${i1.nt} x ${i1.nt} attention mask per prompt.`,
      `Image 2 x ${fmt(b.NI)} x 256 and text 2 x ${b.NT} x 256 through 6 layers. Cross-attention compares ${fmt(b.NI * b.NT)} image-text pairs per head per image; deformable self-attention avoids all ${fmt(b.NI)} x ${fmt(b.NI)} image pairs.`,
      `Scores 2 x ${fmt(b.NI)} x ${b.NT} -> max over text -> 2 x ${fmt(b.NI)} -> top 900 (${(NQ / b.NI * 100).toFixed(1)}% of the tokens) -> anchors 2 x 900 x 4. Content queries 900 x 256, learned, shared by both images.`,
      `Queries 2 x 900 x 256 through 6 layers; text cross-attention scores 900 x ${b.NT} per head per image. Each layer outputs boxes 2 x 900 x 4 and logits 2 x 900 x ${b.NT}.`,
      `7 prediction sets (6 decoder layers and the encoder outputs) x 2 images = 14 assignments this step. Image 1: ${m} of 900 queries get an object, ${NQ - m} get all-zero targets. On the 12 drawn queries, class weight 2.0 and 1.0 give ${b.same ? 'the same' : 'different'} assignments.`,
      `Focal loss over 2 x 900 x ${b.NT} query-token logits per set; L1 and GIoU on ${tot} matched pairs per set; 7 sets summed, one backward pass, one AdamW step.`,
    ][S.stage];
  }
  function show(i, animate = true) {
    stopA(); S.stage = (i + NS) % NS;
    chips.forEach((c, j) => { c.classList.toggle('done', j < S.stage); if (j === S.stage) c.setAttribute('aria-current', 'step'); else c.removeAttribute('aria-current'); });
    texts.forEach((d, j) => { d.hidden = j !== S.stage; });
    readout();
    if (animate) { S.t = 0; stopA = PE.animate(900, u => { S.t = u; cv.redraw(); }); } else { S.t = 1; cv.redraw(); }
  }
  const runBtn = document.getElementById('btn-train-run');
  function stopRun() { clearTimeout(runTimer); S.running = false; runBtn.setAttribute('aria-pressed', 'false'); }
  function run() { stopRun(); S.running = true; runBtn.setAttribute('aria-pressed', 'true');
    const still = PE.instant(); let i = 0; const tick = () => { show(i, !still); if (++i < NS) runTimer = setTimeout(tick, 2200); else stopRun(); }; tick(); }
  chips.forEach((c, i) => c.addEventListener('click', () => { stopRun(); show(i); }));
  document.getElementById('btn-train-prev').addEventListener('click', () => { stopRun(); show(S.stage - 1); });
  document.getElementById('btn-train-next').addEventListener('click', () => { stopRun(); show(S.stage + 1); });
  runBtn.addEventListener('click', () => { if (S.running) stopRun(); else run(); });
  document.getElementById('btn-train-batch').addEventListener('click', () => { S.seed++; batch = makeBatch(S.seed); show(S.stage); });
  document.getElementById('btn-train-reset').addEventListener('click', () => { stopRun(); S.seed = 1; batch = makeBatch(1); show(0, false); });
  batch = makeBatch(1); show(0, false);
})();
</script>
