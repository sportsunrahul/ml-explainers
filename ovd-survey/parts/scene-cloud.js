<script>
/* SceneCloud: the page's one 3D figure, a toy embedding sphere drawn with PE3. Shared by viz-vild3d (ch. 4),
   viz-gap3d (ch. 13) and the atlas. Everything is a toy: positions are hand-placed unit vectors, and every
   cosine in the readout is computed from those toy positions.
     const sc = SceneCloud(canvasEl, { aspect, onSay(text), start, jargon, describe });
       aspect    height / width (number or function of width); default .6 on desktop, .95 below 560 px
       onSay     receives one sentence per set() / reset() (write it to an aria-live readout)
       start     the state reset() returns to: 'start' (default, ch. 4) or 'gap' (ch. 13)
       jargon    text of the jargon prompt, default 'porosity'
       describe  text of the descriptive prompt, default 'small round holes in a weld'
     sc.set(state)  tweens to the state (states hold absolute targets, so setting one twice equals once)
     sc.reset(); sc.redraw(); sc.resize(); sc.destroy(); sc.state() -> current state name
   States:
     'start'    untrained detector (ViLD trains from scratch): region embeddings (blue) point anywhere
     'text'     ViLD-text: each base-class region moves next to its class's text point (yellow). The forklift
                regions have no label, so Eq. 2 calls them background and pulls them toward e_bg (brown)
     'image'    ViLD-image: every region, forklift included, moves onto CLIP's embedding of its crop (small faded blue)
     'both'     ViLD: base regions settle between their name and their CLIP target; the forklift regions stay
                near their CLIP targets, so 'forklift' becomes their nearest name
     'gap'      the trained cloud, dimmed, plus a cluster of weld-pore crop embeddings; no new prompt yet
     'jargon'   'gap' plus the text point of the jargon name, which lands far from the pore cluster
     'describe' 'jargon' plus the text point of the descriptive prompt, which lands close to the cluster
   Link words (PE3 pulses items whose link matches a 'pe-link' event on the enclosing figure):
     regions, text, novel, background, image, sphere, pores, prompt. */
const SceneCloud = (el, o = {}) => {
  const YAW = .6, PITCH = .58, RAD = Math.PI / 180;
  const nrm = p => { const L = Math.hypot(p[0], p[1], p[2]) || 1; return [p[0] / L, p[1] / L, p[2] / L]; };
  const dotp = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const sl = (a, b, u) => nrm([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u]);
  // Unit vector placed by its screen angle phi (0 = right, 90 = up) and its angle t from the centre of the default view.
  const cY = Math.cos(YAW), sY = Math.sin(YAW), cP = Math.cos(PITCH), sP = Math.sin(PITCH);
  const C0 = [sY * cP, sP, cY * cP], E1 = [cY, 0, -sY], E2 = [-sY * sP, cP, -cY * sP];
  const dir = (phi, t) => { const f = phi * RAD, a = t * RAD, c = Math.cos(a), s = Math.sin(a);
    return nrm([0, 1, 2].map(i => C0[i] * c + (E1[i] * Math.cos(f) + E2[i] * Math.sin(f)) * s)); };

  const TXT = [
    { n: 'person', phi: 140, t: 52 }, { n: 'pallet', phi: 198, t: 54 }, { n: 'traffic cone', phi: 252, t: 50 },
    { n: 'forklift', phi: 14, t: 50, novel: true },
  ].map(q => ({ ...q, p: dir(q.phi, q.t) }));
  const BG = dir(78, 56);
  // Two regions per class. Each CLIP crop target sits a little off its name: CLIP puts images and text near, not on, each other.
  const OFF = [[-15, -13], [13, -11]];
  const r0 = PE.rng(4);
  const REG = [];
  TXT.forEach((q, k) => OFF.forEach(([df, dt]) => REG.push({ k, novel: !!q.novel,
    tgt: dir(q.phi + df * (q.t > 45 ? 1.2 : 1), q.t + dt), s: dir(360 * r0(), 12 + 46 * r0()) })));
  const PORE_C = dir(318, 40), PORES = [[0, 0], [8, 5], [-7, 4], [5, -6], [-6, -5]].map(([a, b]) => dir(318 + a * 1.4, 40 + b));
  const JAR = dir(160, 40), DES = dir(334, 26);
  const jargon = o.jargon || 'porosity', describe = o.describe || 'small round holes in a weld';

  function spec(st) {
    const gapish = st === 'gap' || st === 'jargon' || st === 'describe';
    const pos = REG.map(r => {
      const T = TXT[r.k].p;
      if (st === 'start') return r.s;
      if (st === 'text') return r.novel ? sl(BG, r.s, .3) : sl(T, r.tgt, .16);
      if (st === 'image') return r.tgt;
      return r.novel ? sl(r.tgt, BG, .14) : sl(T, r.tgt, .5);   // 'both' and the ch. 13 states (a trained ViLD)
    });
    return { pos, dim: gapish ? 1 : 0, pores: gapish ? 1 : 0, jar: st === 'jargon' || st === 'describe' ? 1 : 0, des: st === 'describe' ? 1 : 0,
      lines: gapish || st === 'start' ? 'none' : st };
  }
  const start = o.start || 'start';
  let name = start, S = { ...spec(start) }, cancel = () => {};

  const s3 = PE3.scene(el, { aspect: o.aspect || (w => (w < 560 ? .95 : .6)), fov: 34, dist: 4.2, yaw: YAW, pitch: PITCH });
  // Wireframe sphere: meridians and three parallels, each split into arcs so the far side fades on its own.
  const SPHERE = [];
  const arcs = (f, n = 8, seg = 8) => { for (let a = 0; a < n; a++) { const pts = []; for (let j = 0; j <= seg; j++) pts.push(f((a + j / seg) / n * 2 * Math.PI)); SPHERE.push(pts); } };
  for (let m = 0; m < 6; m++) { const b = m * Math.PI / 6; arcs(t => [Math.cos(t) * Math.cos(b), Math.sin(t), Math.cos(t) * Math.sin(b)]); }
  [-.5, 0, .5].forEach(y => { const r = Math.sqrt(1 - y * y); arcs(t => [r * Math.cos(t), y, r * Math.sin(t)]); });

  const paleBlue = () => PE3.mix(PE.css('--blue'), PE.css('--bg'), .45);
  const lab = (at, text, color, dy, alpha = 1, dx = 0, size = 12) => ({ type: 'label', at, text, color, dy, dx, alpha, font: `600 ${size}px system-ui, sans-serif` });
  const narrow = () => (el.clientWidth || 600) < 560;
  // Great-circle arc between two unit vectors, for the prompt-to-cluster gap.
  const arc = (a, b, n = 16) => Array.from({ length: n + 1 }, (_, j) => sl(a, b, j / n));

  function build() {
    const fs = narrow() ? 10.5 : 12, dimA = 1 - .68 * S.dim, items = [];
    SPHERE.forEach(pts => items.push({ type: 'line', pts, color: '--rule', width: 1.1, link: 'sphere' }));
    // residual lines: what the active loss still wants
    const L = S.lines;
    if (L !== 'none') REG.forEach((r, i) => {
      const p = S.pos[i], toText = (L === 'text' || L === 'both') && !r.novel, toBg = (L === 'text' || L === 'both') && r.novel, toImg = L === 'image' || L === 'both';
      [[toText, TXT[r.k].p], [toBg, BG], [toImg, r.tgt]].forEach(([on, q]) => { if (on && dotp(p, q) < .9995) items.push({ type: 'line', pts: arc(p, q, 6), color: '--red', width: 1.3, dash: [3, 3], alpha: .75 * dimA }); });
    });
    items.push({ type: 'points', pts: REG.map(r => r.tgt), color: paleBlue, size: 4, link: 'image', alpha: (S.dim ? .5 : 1) * dimA });
    TXT.forEach(q => {
      items.push({ type: 'points', pts: [q.p], color: '--yellow', size: 8.5, link: q.novel ? 'novel' : 'text', alpha: Math.max(.55, dimA) });
      items.push(lab(q.p, q.novel ? 'forklift (novel)' : q.n, '--yellow', -14, Math.max(.55, dimA), 0, fs));
    });
    items.push({ type: 'points', pts: [BG], color: '--brown', size: 8.5, link: 'background', alpha: dimA });
    items.push(lab(BG, 'e_bg (learned background)', '--brown', -14, dimA, 0, fs));
    items.push({ type: 'points', pts: S.pos.filter((_, i) => !REG[i].novel), color: '--blue', size: 6, link: 'regions', alpha: dimA });
    const fk = S.pos.filter((_, i) => REG[i].novel);
    items.push({ type: 'points', pts: fk, color: '--blue', size: 7, link: 'regions', alpha: dimA });
    // Label below the lower of the two forklift points on screen; when they are apart (untrained), label each.
    if (!S.dim) {
      if (dotp(fk[0], fk[1]) > .97) { const lo = s3.project(fk[0]).y > s3.project(fk[1]).y ? fk[0] : fk[1]; items.push(lab(lo, 'forklift regions', '--blue', 17, dimA, 0, fs)); }
      else fk.forEach(p => items.push(lab(p, 'forklift region', '--blue', 17, dimA, 0, fs)));
    }
    if (S.pores > 0) {
      items.push({ type: 'points', pts: PORES, color: '--blue', size: 6.5, link: 'pores', alpha: S.pores });
      items.push(lab(PORE_C, 'weld-pore crops', '--blue', 22, S.pores, 0, fs));
    }
    [[S.jar, JAR, jargon], [S.des, DES, describe]].forEach(([u, P, txt]) => {
      if (u <= 0) return;
      const at = P.map(v => v * (1 + .7 * (1 - u)));
      items.push({ type: 'points', pts: [at], color: '--yellow', size: 8.5, link: 'prompt', alpha: u });
      items.push(lab(at, '"' + txt + '"', '--yellow', -14, u, 0, fs));
      if (u >= 1) { const c = dotp(P, PORE_C); items.push({ type: 'line', pts: arc(P, PORE_C), color: '--muted', width: 1.2, dash: [4, 4], alpha: .9 });
        items.push(lab(sl(P, PORE_C, .5), 'cos ' + c.toFixed(2), '--muted', -9, 1, 0, fs - 1)); }
    });
    s3.items = items; s3.redraw();
  }

  // The test-time classifier: background plus all four names. Returns the nearest row to p and its cosine.
  function nearest(p) {
    const rows = [['background', BG], ...TXT.map(q => [q.n, q.p])].map(([n, q]) => [n, dotp(p, q)]);
    return rows.reduce((a, b) => (b[1] > a[1] ? b : a));
  }
  function describeState(st) {
    const sp = spec(st), fk = sl(sp.pos[6], sp.pos[7], .5), [nn, nc] = nearest(fk), c = v => v.toFixed(2);
    const base = sp.pos.slice(0, 6).filter((p, i) => nearest(p)[0] === TXT[REG[i].k].n).length;
    const cJ = dotp(JAR, PORE_C), cD = dotp(DES, PORE_C);
    return {
      start: `Untrained detector: region embeddings point anywhere. ${base} of 6 base-class regions are nearest their own name; the forklift regions are nearest "${nn}".`,
      text: `ViLD-text: all ${base} base-class regions sit next to their names. The forklift regions have no label, so the loss calls them background: nearest row "${nn}" (cosine ${c(nc)}).`,
      image: `ViLD-image: every region moves onto CLIP's embedding of its crop. The forklift regions' nearest name is now "${nn}" (cosine ${c(nc)}), though no label named them.`,
      both: `ViLD: base regions settle between their name and their CLIP target (${base} of 6 still nearest their own name). The forklift regions keep "${nn}" as nearest name (cosine ${c(nc)}).`,
      gap: 'Weld-pore crops (blue cluster) in the same space as the trained cloud. No prompt is typed yet.',
      jargon: `"${jargon}" lands far from the pore crops: cosine ${c(cJ)}. A detector that scores crops against this name ranks the pores low.`,
      describe: `"${describe}" lands near the pore crops: cosine ${c(cD)}, against ${c(cJ)} for "${jargon}".`,
    }[st];
  }
  const say = st => o.onSay && o.onSay(describeState(st));

  function set(st) {
    cancel(); name = st;
    const to = spec(st), from = { ...S, pos: S.pos.slice() };
    S.lines = to.lines;
    cancel = s3.tween(st === 'jargon' || st === 'describe' ? 900 : 850, u => {
      S.pos = from.pos.map((p, i) => sl(p, to.pos[i], u));
      ['dim', 'pores', 'jar', 'des'].forEach(k => { S[k] = from[k] + (to[k] - from[k]) * u; });
      build();
    }, () => { Object.assign(S, to); build(); });
    say(st);
  }
  function reset() { cancel(); name = start; S = { ...spec(start) }; s3.setView({ yaw: YAW, pitch: PITCH }, 600); build(); say(start); }
  build(); say(start);
  return { set, reset, state: () => name, redraw: () => build(), resize: () => s3.redraw(),
    destroy: () => { cancel(); s3.destroy(); } };
};
</script>
