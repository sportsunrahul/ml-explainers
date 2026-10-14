/* ch-map family tree. Lanes are columns, time runs down. Rows group by year; inside a lane the order is exact first-
   submission order, across lanes only the year is shared. Dates are claim values (<key>-published, or the venue claim
   for the four papers that are not on arXiv). Edges are "builds on" links that the paper notes support. */
KIT.CTREE = {
  lanes: ['Negatives', 'Theory', 'Hard negatives', 'No negatives', 'Labels', 'Language', 'Dense'],
  // [key, lane, label, date claim id, date value, arXiv id or null, chapter id, spine?, one-line summary]
  nodes: [
    ['chopra2005', 4, 'Siamese net', 'chopra2005-venue', 'CVPR 2005', null, 'ch-compare', 0, 'Two copies of one network with shared weights map two faces to points; the distance is trained to be small for the same person and large otherwise.'],
    ['hadsell2006', 4, 'DrLIM', 'hadsell2006-venue', 'CVPR 2006', null, 'ch-compare', 1, 'The margin contrastive loss you met in chapter 1: pull similar pairs together, push dissimilar pairs only while they sit inside the margin.'],
    ['gutmann2010', 1, 'NCE', 'gutmann2010-venue', 'AISTATS 2010', null, 'ch-infonce', 0, 'Learn a density by training a classifier to tell real data from made-up noise. This is the "NCE" in InfoNCE.'],
    ['facenet', 4, 'FaceNet', 'facenet-published', '2015-03-12', '1503.03832', 'ch-compare', 0, 'The triplet loss: the anchor must be closer to its positive than to a negative by a margin, with useful negatives mined on the fly.'],
    ['sohn2016', 4, 'N-pair', 'sohn2016-venue', 'NIPS 2016', null, 'ch-compare', 0, 'Compare the anchor with one positive and many negatives at once through a softmax, reusing the other pairs in the batch as negatives.'],
    ['wu2018', 0, 'Memory bank', 'wu2018-published', '2018-05-05', '1805.01978', 'ch-moco', 1, 'Treat every image as its own class, and keep one stored vector per image in a memory bank so each step can compare against many negatives.'],
    ['cpc', 0, 'CPC', 'cpc-published', '2018-07-10', '1807.03748', 'ch-infonce', 1, 'Predict the future in a learned feature space and score the guess with a softmax over the true future and many negatives: the InfoNCE loss.'],
    ['arora', 1, 'Arora', 'arora-published', '2019-02-25', '1902.09229', 'ch-infonce', 0, 'The first theory: if positives share a hidden class, a low contrastive loss provably gives a low error for a simple linear classifier, plus a penalty when a negative shares that class.'],
    ['poole', 1, 'Poole', 'poole-published', '2019-05-16', '1905.06922', 'ch-infonce', 0, 'Puts the mutual-information bounds in one framework, and shows InfoNCE has low variance but can never exceed log N.'],
    ['cpcv2', 0, 'CPC v2', 'cpcv2-published', '2019-05-22', '1905.09272', 'ch-infonce', 0, 'CPC re-engineered for images (a bigger encoder, more augmentation) and tested with few labels and on detection.'],
    ['cmc', 0, 'CMC', 'cmc-published', '2019-06-13', '1906.05849', 'ch-infonce', 0, 'Views need not be crops: different channels of one scene, such as color channels, depth or motion, can form the positive pair.'],
    ['tschannen', 1, 'Tschannen', 'tschannen-published', '2019-07-31', '1907.13625', 'ch-infonce', 0, 'Tighter mutual-information bounds can give worse features. The choice of encoder and critic matters more than the information count.'],
    ['moco', 0, 'MoCo', 'moco-published', '2019-11-13', '1911.05722', 'ch-moco', 1, 'A queue of recent keys from a slowly moving copy of the encoder: many consistent negatives without a huge batch.'],
    ['simclr', 0, 'SimCLR', 'simclr-published', '2020-02-13', '2002.05709', 'ch-simclr', 1, 'No bank and no queue: the other views in one large batch are the negatives. Strong crop and color augmentation and a throwaway head do the rest.'],
    ['mocov2', 0, 'MoCo v2', 'mocov2-published', '2020-03-09', '2003.04297', 'ch-moco', 1, 'MoCo plus SimCLR\'s MLP head, blur augmentation and a cosine schedule. It keeps the queue and the momentum encoder.'],
    ['supcon', 4, 'SupCon', 'supcon-published', '2020-04-23', '2004.11362', 'ch-supcon', 1, 'With labels, every same-class sample in the batch is a positive. The classes form tight clusters.'],
    ['infomin', 0, 'InfoMin', 'infomin-published', '2020-05-20', '2005.10243', 'ch-simclr', 0, 'Good views share only what the task needs; stronger augmentation moves toward that sweet spot.'],
    ['wangisola', 1, 'Wang & Isola', 'wangisola-published', '2020-05-20', '2005.10242', 'ch-sphere', 1, 'With many negatives the loss splits into two forces: alignment (partners close) and uniformity (all points spread evenly over the sphere).'],
    ['byol', 3, 'BYOL', 'byol-published', '2020-06-13', '2006.07733', 'ch-beyond', 0, 'An online network predicts what a slowly moving target network outputs for another view. There are no negatives at all.'],
    ['swav', 3, 'SwAV', 'swav-published', '2020-06-17', '2006.09882', 'ch-beyond', 1, 'Predict one view\'s cluster assignment from the other view. A rule that every cluster gets used equally stops collapse, without negatives.'],
    ['simclrv2', 0, 'SimCLR v2', 'simclrv2-published', '2020-06-17', '2006.10029', 'ch-simclr', 0, 'Bigger SimCLR networks pretrained without labels, fine-tuned on a few labels, then distilled into a smaller student.'],
    ['chuang', 2, 'Debiased', 'chuang-published', '2020-07-01', '2007.00224', 'ch-negatives', 1, 'Without labels some "negatives" share the anchor\'s class. Correct the loss for them with an assumed class prior.'],
    ['robinson', 2, 'Hard negatives', 'robinson-published', '2020-10-09', '2010.04592', 'ch-negatives', 1, 'On top of that correction, give the closer, harder negatives more weight, with a knob for how strongly.'],
    ['densecl', 6, 'DenseCL', 'densecl-published', '2020-11-18', '2011.09157', 'ch-guide', 0, 'Add a dense head to MoCo v2: contrast local features across the two views, each matched to its most similar location.'],
    ['simsiam', 3, 'SimSiam', 'simsiam-published', '2020-11-20', '2011.10566', 'ch-beyond', 0, 'BYOL without the momentum encoder: a predictor on one side and a stop-gradient on the other are enough.'],
    ['wangliu', 1, 'Wang & Liu', 'wangliu-published', '2020-12-15', '2012.09740', 'ch-sphere', 1, 'Temperature sets how hard the push focuses on the nearest negatives. A small value spreads points, but also pushes similar images apart.'],
    ['align', 5, 'ALIGN', 'align-published', '2021-02-11', '2102.05918', 'ch-clip', 0, 'A CLIP-style loss on a large, noisy set of web images and their alt text: scale makes up for the noise.'],
    ['clip', 5, 'CLIP', 'clip-published', '2021-02-26', '2103.00020', 'ch-clip', 1, 'The positive is a caption: a softmax over every row and column of the image-text similarity matrix. Zero-shot classification picks the best caption.'],
    ['barlow', 3, 'Barlow Twins', 'barlow-published', '2021-03-04', '2103.03230', 'ch-beyond', 1, 'Drive the cross-correlation matrix of the two views\' embeddings to the identity: features that are invariant and not redundant.'],
    ['detcon', 6, 'DetCon', 'detcon-published', '2021-03-19', '2103.10957', 'ch-guide', 0, 'Contrast object-level features pooled inside unsupervised masks, so one image gives many training signals.'],
    ['mocov3', 0, 'MoCo v3', 'mocov3-published', '2021-04-05', '2104.02057', 'ch-moco', 1, 'MoCo for vision transformers: no queue, a large batch, a prediction head, and a frozen random patch layer for stable training.'],
    ['dino', 3, 'DINO', 'dino-published', '2021-04-29', '2104.14294', 'ch-beyond', 0, 'Self-distillation: a student matches a momentum teacher\'s output across crops. Centering and sharpening stop collapse.'],
    ['vicreg', 3, 'VICReg', 'vicreg-published', '2021-05-11', '2105.04906', 'ch-beyond', 0, 'Three explicit terms: keep each dimension\'s variance up, decorrelate the dimensions, and pull the two views together.'],
    ['haochen', 1, 'Spectral loss', 'haochen-published', '2021-06-08', '2106.04156', 'ch-sphere', 0, 'A spectral contrastive loss: minimizing it factorizes a graph of augmented images, which gives provable guarantees.'],
    ['dcl', 2, 'DCL', 'dcl-published', '2021-10-13', '2110.06848', 'ch-negatives', 0, 'Remove the positive from the denominator of InfoNCE. This helps most when the batch is small.'],
    ['jing', 1, 'Dim. collapse', 'jing-published', '2021-10-18', '2110.09348', 'ch-beyond', 1, 'Contrastive embeddings can still collapse partly: they fill only a low-dimensional subspace of the space they live in.'],
    ['ibot', 3, 'iBOT', 'ibot-published', '2021-11-15', '2111.07832', 'ch-beyond', 0, 'DINO plus masked image modeling: the student predicts the teacher\'s patch outputs where the input was masked.'],
    ['openclipscaling', 5, 'Scaling laws', 'openclipscaling-published', '2022-12-14', '2212.07143', 'ch-clip', 0, 'CLIP trained on public data at many sizes. Fitted scaling laws show that the training data changes how fast error falls.'],
    ['siglip', 5, 'SigLIP', 'siglip-published', '2023-03-27', '2303.15343', 'ch-clip', 1, 'Replace the softmax with an independent sigmoid per image-text pair and a learnable bias. No batch-wide normalization.'],
    ['dinov2', 3, 'DINOv2', 'dinov2-published', '2023-04-14', '2304.07193', 'ch-beyond', 0, 'DINO and iBOT scaled up on a curated image set. The frozen features work across many tasks.'],
    ['metaclip', 5, 'MetaCLIP', 'metaclip-published', '2023-09-28', '2309.16671', 'ch-clip', 0, 'CLIP\'s edge is its data: rebuild a balanced curation recipe and keep the model and loss fixed.'],
    ['siglip2', 5, 'SigLIP 2', 'siglip2-published', '2025-02-20', '2502.14786', 'ch-clip', 0, 'SigLIP\'s loss plus captioning, self-distillation and masked prediction in one recipe, with much better dense features.'],
    ['webssl', 3, 'Web-SSL', 'webssl-published', '2025-04-01', '2504.01017', 'ch-beyond', 0, 'DINOv2-style and CLIP models trained on the same web images: with matched data, the language-free model scales better.'],
    ['pe', 5, 'Percep. Enc.', 'pe-published', '2025-04-17', '2504.13181', 'ch-clip', 0, 'A carefully scaled CLIP-style recipe. Its best features for dense tasks sit inside the network, not at the output.'],
    ['dinov3', 3, 'DINOv3', 'dinov3-published', '2025-08-13', '2508.10104', 'ch-beyond', 0, 'Self-distillation scaled much further, with a new loss term that keeps the dense, per-patch features sharp.'],
  ],
  edges: [['chopra2005', 'hadsell2006'], ['facenet', 'sohn2016'], ['sohn2016', 'supcon'], ['facenet', 'supcon'], ['sohn2016', 'clip'], ['cpc', 'clip'],
    ['gutmann2010', 'wu2018'], ['gutmann2010', 'cpc'], ['wu2018', 'cmc'], ['wu2018', 'moco'], ['cpc', 'cpcv2'], ['cpc', 'poole'], ['cpc', 'tschannen'],
    ['tschannen', 'wangisola'], ['wangisola', 'wangliu'], ['cpc', 'simclr'], ['moco', 'simclr'], ['moco', 'mocov2'], ['simclr', 'mocov2'], ['mocov2', 'mocov3'],
    ['byol', 'mocov3'], ['simclr', 'simclrv2'], ['moco', 'simclrv2'], ['simclr', 'chuang'], ['chuang', 'robinson'], ['simclr', 'dcl'], ['simclr', 'jing'],
    ['moco', 'swav'], ['simclr', 'swav'], ['byol', 'simsiam'], ['byol', 'barlow'], ['simsiam', 'barlow'], ['barlow', 'vicreg'], ['dino', 'ibot'],
    ['dino', 'dinov2'], ['ibot', 'dinov2'], ['swav', 'dinov2'], ['dinov2', 'dinov3'], ['dinov2', 'webssl'], ['metaclip', 'webssl'], ['clip', 'openclipscaling'],
    ['clip', 'metaclip'], ['clip', 'siglip'], ['siglip', 'siglip2'], ['clip', 'pe'], ['metaclip', 'pe'], ['mocov2', 'densecl'], ['simclr', 'detcon'], ['byol', 'detcon']],
  chapters: { 'ch-compare': 'chapter 1', 'ch-infonce': 'chapter 3', 'ch-moco': 'chapter 4', 'ch-simclr': 'chapter 5', 'ch-sphere': 'chapter 6',
    'ch-negatives': 'chapter 7', 'ch-beyond': 'chapter 8', 'ch-supcon': 'chapter 9', 'ch-clip': 'chapter 10', 'ch-guide': '"When to reach for it"' },
};
KIT.ctreeYear = n => +n[4].slice(-4) || +n[4].slice(0, 4);
// Layout: rows grouped by year; within a year each lane stacks its papers in date order.
KIT.ctreeRows = () => {
  const T = KIT.CTREE, years = [...new Set(T.nodes.map(KIT.ctreeYear))].sort((a, b) => a - b), rows = {}, bands = [];
  let r = 0;
  years.forEach(y => { const inY = T.nodes.filter(n => KIT.ctreeYear(n) === y), slot = {};
    inY.forEach(n => { slot[n[1]] = (slot[n[1]] || 0); rows[n[0]] = r + slot[n[1]]; slot[n[1]]++; });
    const h = Math.max(...Object.values(slot)); bands.push({ year: y, row: r, h }); r += h; });
  return { rows, bands, total: r };
};
KIT.ctree = (host, opts = {}) => {
  const T = KIT.CTREE, idx = Object.fromEntries(T.nodes.map((n, i) => [n[0], i])), L = KIT.ctreeRows();
  host.style.position = 'relative';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); svg.setAttribute('aria-hidden', 'true');
  svg.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;overflow:visible';
  host.appendChild(svg);
  const pills = T.nodes.map(n => { const b = document.createElement('button'); b.type = 'button'; b.id = 'btn-tree-' + n[0]; b.className = 'tree-pill';
    b.textContent = n[2]; b.setAttribute('aria-pressed', 'false'); b.title = n[2] + ', ' + n[4];
    if (!n[7]) b.style.borderStyle = 'dashed'; else b.style.borderColor = 'var(--muted)';
    host.appendChild(b); return b; });
  const state = { sel: null, e: 0 };
  let narrow = false, cancel = () => {};
  const RH = 30, TOP = 40, LEFT = 50;
  const layout = () => {
    const w = host.clientWidth; if (!w) return;
    narrow = w < 640;
    if (narrow) {
      const byDate = T.nodes.slice().sort((a, b) => KIT.ctreeYear(a) - KIT.ctreeYear(b) || (a[4] < b[4] ? -1 : 1));
      byDate.forEach((n, k) => { const b = pills[idx[n[0]]]; b.style.left = LEFT + 'px'; b.style.top = (8 + k * 32) + 'px'; b.style.maxWidth = (w - LEFT - 92) + 'px'; });
      host.style.height = (byDate.length * 32 + 16) + 'px';
    } else {
      const lw = (w - LEFT) / T.lanes.length;
      T.nodes.forEach((n, i) => { const b = pills[i]; b.style.left = (LEFT + lw * n[1] + 4) + 'px'; b.style.top = (TOP + L.rows[n[0]] * RH) + 'px'; b.style.maxWidth = (lw - 10) + 'px'; });
      host.style.height = (TOP + L.total * RH + 8) + 'px';
    }
    redraw();
  };
  const redraw = () => {
    const w = host.clientWidth; if (!w) return;
    const mut = PE.css('--muted'), fg = PE.css('--fg'), rule = PE.css('--rule');
    const txt = (x, y, t, c, sz, wt) => `<text x="${x}" y="${y}" font-size="${sz}" font-weight="${wt || 400}" fill="${c}" font-family="system-ui, sans-serif">${t}</text>`;
    let s = '';
    if (narrow) {
      let last = '';
      T.nodes.slice().sort((a, b) => KIT.ctreeYear(a) - KIT.ctreeYear(b) || (a[4] < b[4] ? -1 : 1)).forEach(n => {
        const y = KIT.ctreeYear(n), b = pills[idx[n[0]]];
        if (String(y) !== last) { last = String(y); s += txt(0, b.offsetTop + 15, y, mut, 11); s += `<line x1="0" y1="${b.offsetTop - 3}" x2="${w}" y2="${b.offsetTop - 3}" stroke="${rule}" opacity=".7"/>`; }
        s += `<text x="${w - 2}" y="${b.offsetTop + 15}" font-size="10" fill="${mut}" text-anchor="end" font-family="system-ui, sans-serif">${T.lanes[n[1]]}</text>`;
      });
    } else {
      const lw = (w - LEFT) / T.lanes.length;
      T.lanes.forEach((l, i) => { s += txt(LEFT + lw * i + 4, 20, l, fg, 12, 600);
        if (i) s += `<line x1="${LEFT + lw * i - 2}" y1="28" x2="${LEFT + lw * i - 2}" y2="100%" stroke="${rule}"/>`; });
      L.bands.forEach(bd => { const y = TOP + bd.row * RH; s += txt(0, y + 16, bd.year, mut, 11);
        s += `<line x1="0" y1="${y - 4}" x2="${w}" y2="${y - 4}" stroke="${rule}" opacity=".6"/>`; });
    }
    // Edges: faint everywhere, emphasized (and tweened in) for the selected paper.
    T.edges.forEach(([a, b]) => {
      const A = pills[idx[a]], B = pills[idx[b]], on = state.sel && (state.sel === a || state.sel === b);
      if (narrow && !on) return;
      const op = on ? .25 + .75 * state.e : .28, sw = on ? 1 + 1.2 * state.e : 1;
      let d;
      if (narrow) { const x = LEFT - 3, ay = A.offsetTop + A.offsetHeight / 2, by = B.offsetTop + B.offsetHeight / 2;
        d = `M${x},${ay} C${x - 12},${ay} ${x - 12},${by} ${x},${by}`; }
      else if (A.offsetTop + A.offsetHeight <= B.offsetTop) { // B is below A: leave from A's bottom, enter B's top
        const ax = A.offsetLeft + A.offsetWidth / 2, ay = A.offsetTop + A.offsetHeight, bx = B.offsetLeft + B.offsetWidth / 2, by = B.offsetTop;
        d = `M${ax},${ay} C${ax},${(ay + by) / 2} ${bx},${(ay + by) / 2} ${bx},${by}`; }
      else { // same band: leave from the facing side
        const r = B.offsetLeft > A.offsetLeft, ax = r ? A.offsetLeft + A.offsetWidth : A.offsetLeft, bx = r ? B.offsetLeft : B.offsetLeft + B.offsetWidth;
        const ay = A.offsetTop + A.offsetHeight / 2, by = B.offsetTop + B.offsetHeight / 2;
        d = `M${ax},${ay} C${(ax + bx) / 2},${ay} ${(ax + bx) / 2},${by} ${bx},${by}`; }
      s += `<path d="${d}" fill="none" stroke="${on ? fg : mut}" stroke-width="${sw}" opacity="${op}"/>`;
    });
    svg.innerHTML = s;
  };
  const set = sel => {
    cancel(); state.sel = sel; state.e = 0;
    pills.forEach((b, i) => b.setAttribute('aria-pressed', String(T.nodes[i][0] === sel)));
    cancel = KIT.tween(state, { e: 1 }, 500, redraw);
  };
  const ro = new ResizeObserver(layout); ro.observe(host);
  const off = PE.onTheme(redraw);
  layout();
  return { set, state, pills, redraw, resize: layout, destroy: () => { cancel(); ro.disconnect(); off(); } };
};
KIT.ctreeText = key => {
  const T = KIT.CTREE, n = T.nodes.find(x => x[0] === key);
  if (!n) return 'Press any paper to see what it changed, what it builds on, and where this page teaches it.';
  const lab = k => T.nodes.find(x => x[0] === k)[2];
  const from = T.edges.filter(e => e[1] === key).map(e => lab(e[0])), to = T.edges.filter(e => e[0] === key).map(e => lab(e[1]));
  return '<b>' + n[2] + '</b> (' + (n[5] ? 'arXiv ' + n[5] + ', first version ' : '') + '<span data-claim="' + n[3] + '">' + n[4] + '</span>). ' + n[8] +
    (from.length ? ' Builds on: ' + from.join(', ') + '.' : '') + (to.length ? ' Leads to: ' + to.join(', ') + '.' : '') +
    ' On this page: <a href="#' + n[6] + '">' + T.chapters[n[6]] + '</a>.';
};
