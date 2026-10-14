<script>
(() => {
  const x = 'x';
  // Parameter counts and FPS from YOLO-World Table 2 (re-parameterized / original; one V100, no TensorRT).
  // Grids assume a 640 input and strides 8 / 16 / 32 (our reading, the YOLOv8 convention); the paper gives no channel widths.
  const V = {
    s: { n: 'S', pr: '13M', po: '77M', fr: '74.1', fo: '19.9' },
    m: { n: 'M', pr: '29M', po: '92M', fr: '58.1', fo: '18.5' },
    l: { n: 'L', pr: '48M', po: '110M', fr: '52.0', fo: '17.6' },
  };
  const R = 640, g = s => R / s, K = [8, 16, 32].reduce((a, s) => a + g(s) * g(s), 0);
  const grids = `${g(8)}${x}${g(8)}, ${g(16)}${x}${g(16)}, ${g(32)}${x}${g(32)}`;
  const SH = {
    prompt: () => 'C <= 80 nouns', img: () => `B${x}3${x}${R}${x}${R}`, tenc: () => `W: C${x}D`,
    bb: () => `C3-C5|${grids}`, neck: () => `P3-P5, gated|${grids}`, ipool: () => `27 tokens|W': C${x}D`,
    emb: () => `B${x}${K}${x}D`, box: () => `B${x}${K}${x}4`, con: () => `B${x}${K}${x}C`,
    lcon: () => 'L_con', lbox: () => 'L_iou + L_dfl',
  };
  const PARAMS = { bb: 'all conv weights', neck: 'C2f / T-CSP convs', ipool: 'attention weights', emb: 'two 3x3 convs', box: 'two 3x3 convs', con: 'alpha, beta' };
  ArchFlow.mount({
    prefix: 'yolowflow', grid: { cols: 6, rows: 3 },
    boxes: {
      prompt: { c: 0, r: 0, lane: 'yellow', title: ['vocabulary'], glyph: 'text', data: true },
      tenc: { c: 1, r: 0, lane: 'yellow', title: ['CLIP text', 'encoder'], glyph: 'layers' },
      ipool: { c: 3, r: 0, lane: 'green', title: ['I-Pooling', 'attention'], glyph: 'grid', cls: false },
      img: { c: 0, r: 1, lane: 'blue', title: ['mosaic of', '4 images'], glyph: 'photos', data: true },
      bb: { c: 1, r: 1, lane: 'blue', title: ['Darknet', 'backbone'], glyph: 'layers' },
      neck: { c: 2, r: 1, lane: 'green', title: ['RepVL-PAN', '(T-CSPLayers)'], glyph: 'gate' },
      emb: { c: 3, r: 1, lane: 'blue', title: ['embedding', 'branch'], glyph: 'vector' },
      con: { c: 4, r: 1, lane: 'green', title: ['text contrastive', 'head'], glyph: 'matrix' },
      lcon: { c: 5, r: 1, lane: 'red', title: ['contrastive', 'loss'], glyph: 'loss' },
      box: { c: 3, r: 2, lane: 'blue', title: ['box', 'branch'], glyph: 'layers' },
      lbox: { c: 4, r: 2, lane: 'red', title: ['box loss'], glyph: 'loss' },
    },
    edges: [['prompt', 'tenc'], ['tenc', 'neck'], ['tenc', 'ipool'], ['img', 'bb'], ['bb', 'neck'], ['neck', 'ipool'], ['neck', 'emb'], ['neck', 'box'],
      ['emb', 'con'], ['ipool', 'con'], ['con', 'lcon'], ['box', 'lbox']],
    stages: [
      { key: 'inputs', show: ['img', 'prompt'] }, { key: 'text', show: ['tenc'] }, { key: 'backbone', show: ['bb'] }, { key: 'neck', show: ['neck'] },
      { key: 'ipool', show: ['ipool'] }, { key: 'head', show: ['emb', 'box'] }, { key: 'score', show: ['con'] }, { key: 'loss', show: ['lcon', 'lbox'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'l',
    shape: id => SH[id](),
    sub: (id, v) => (id === 'bb' ? `YOLOv8-${v.n}|${grids}` : null),
    params: id => PARAMS[id] || null,
    frozen: id => id === 'tenc',
    readout: (s, v) => ({
      overview: `YOLO-World-${v.n}: ${v.pr} parameters re-parameterized, ${v.po} in the original version; ${v.fr} vs ${v.fo} FPS on one V100 (Table 2). At a ${R} input it predicts on ${grids} cells: ${K} per image (our reading).`,
      inputs: `Step 1: a mosaic of 4 images, B${x}3${x}${R}${x}${R} (640 is our reading), and its own vocabulary of C <= 80 nouns: the positives in the mosaic plus random negatives.`,
      text: `Step 2: the frozen CLIP text encoder maps the C nouns to W, C${x}D. D is not stated in the paper.`,
      backbone: `Step 3: YOLOv8-${v.n} Darknet backbone. C3, C4, C5 at strides 8, 16, 32: ${grids} (our reading).`,
      neck: `Step 4: top-down, then bottom-up over the three maps. Each T-CSPLayer scales every cell by sigmoid(max_j x . w_j) (Eq. 2). Out: P3, P4, P5 on ${grids}.`,
      ipool: `Step 5: max-pool P3, P4, P5 to 3${x}3 each: 27 tokens. W (C${x}D) attends to them: W' = W + MHA(W, X~, X~) (Eq. 3).`,
      head: `Step 6: two branches of two 3${x}3 convs on every cell: ${g(8) ** 2} + ${g(16) ** 2} + ${g(32) ** 2} = ${K} boxes (B${x}${K}${x}4) and embeddings (B${x}${K}${x}D).`,
      score: `Step 7: s = alpha * cos(e_k, w_j) + beta (Eq. 1): B${x}${K}${x}C scores, ${K}${x}80 per image when the vocabulary is full.`,
      loss: 'Step 8: task-aligned assignment marks positives and gives each a noun index. L = L_con + lambda_I (L_iou + L_dfl); lambda_I = 0 for pseudo-labelled image-text data.',
      backward: 'Backward: red reaches the contrastive head (alpha, beta), both head branches, I-Pooling, the neck and the backbone. The CLIP text encoder is brown: frozen, with only data upstream, so the gradient stops.',
      update: 'Update: AdamW, lr 0.002, weight decay 0.05, 100 epochs, 32 V100 GPUs, batch 512. The text encoder stays frozen in pretraining.',
    }[s]),
  });
})();
</script>
