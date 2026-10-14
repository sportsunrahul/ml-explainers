<script>
(() => {
  const x = '×';
  // SigLIP (PDF): Sec. 4.2 base recipe, Sec. 4.6 scaled-up recipe, Sec. 4.4 / Table 1 row 1 SigLiT. Vision widths from ViT Table 1 (SigLIP cites ViT [17]).
  // chips: Table 1 (null where the paper gives none); text: token cap; tw: text width, null when the paper does not state it.
  const V = {
    b16: { name: 'SigLIP B/16 at 224 px', P: 16, r: 224, tok: 16, tw: 768, d: 768, batch: '32 k', bn: 32768, chips: 32, lit: false },
    b16x: { name: 'SigLIP B/16 at 256 px, 64 tokens', P: 16, r: 256, tok: 64, tw: 768, d: 768, batch: '32 k', bn: 32768, chips: null, lit: false },
    siglit: { name: 'SigLiT: frozen B/8 image tower', P: 8, r: null, tok: 16, tw: null, d: null, batch: '32 k', bn: 32768, chips: 4, lit: true },
  };
  const np = m => m.r ? String((m.r / m.P) ** 2) : 'n';
  const W = m => m.tw ? String(m.tw) : 'W';
  const D = m => m.d ? String(m.d) : 'd';
  const SH = {
    i0: m => m.r ? `N${x}3${x}${m.r}${x}${m.r}` : 'N photos, offline', i1: m => `N${x}${np(m)}${x}768`, i2: m => `N${x}${D(m)}`, i3: m => `N${x}${D(m)}, unit`,
    t0: () => 'N strings', t1: m => `N${x}${m.tok}${x}${W(m)}`, t2: m => `N${x}${D(m)}`, t3: m => `N${x}${D(m)}, unit`, sim: () => `N${x}N, in blocks`, loss: () => 'one number' };
  const PARAMS = { i1: 'patch embedding', t1: 'token embeddings', i2: 'all layers', t2: 'all layers', sim: 't and b' };
  const fmt = v => Math.round(v).toLocaleString('en-US');
  ArchFlow.mount({
    prefix: 'siglipflow', grid: { cols: 6, rows: 2 },
    boxes: {
      i0: { c: 0, r: 0, lane: 'blue', title: ['photos'], glyph: 'photos', data: true },
      i1: { c: 1, r: 0, lane: 'blue', title: ['patches'], glyph: 'grid', cls: false },
      i2: { c: 2, r: 0, lane: 'blue', title: ['ViT', '+ pool'], glyph: 'layers' },
      i3: { c: 3, r: 0, lane: 'blue', title: ['L2 norm'], glyph: 'unit' },
      t0: { c: 0, r: 1, lane: 'yellow', title: ['captions'], glyph: 'text', data: true },
      t1: { c: 1, r: 1, lane: 'yellow', title: ['tokens', '+ embed'], glyph: 'tokens' },
      t2: { c: 2, r: 1, lane: 'yellow', title: ['text', 'transformer'], glyph: 'layers' },
      t3: { c: 3, r: 1, lane: 'yellow', title: ['L2 norm'], glyph: 'unit' },
      sim: { c: 4, r: .5, lane: 'green', title: ['logits', 't x·y + b'], glyph: 'matrix' },
      loss: { c: 5, r: .5, lane: 'red', title: ['sigmoid', 'per cell'], glyph: 'loss' },
    },
    edges: [['i0', 'i1'], ['i1', 'i2'], ['i2', 'i3'], ['i3', 'sim'], ['t0', 't1'], ['t1', 't2'], ['t2', 't3'], ['t3', 'sim'], ['sim', 'loss']],
    stages: [
      { key: 'batch', show: ['i0', 't0'] }, { key: 'tokens', show: ['i1', 't1'] }, { key: 'encode', show: ['i2', 't2'] }, { key: 'embed', show: ['i3', 't3'] },
      { key: 'sim', show: ['sim'] }, { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'b16',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'i2' ? `12 layers, pool|${SH.i2(m)}` : id === 't2' ? (m.lit ? `12 layers, pool|${SH.t2(m)}` : `B-sized, pool|${SH.t2(m)}`) : null,
    params: id => PARAMS[id] || null,
    frozen: (id, m) => m.lit && (id === 'i1' || id === 'i2'),
    readout: (s, m) => {
      const g = m.r ? m.r / m.P : null, cells = m.bn * m.bn, per = m.chips ? m.bn / m.chips : null;
      return {
        overview: `${m.name}: the same two lanes as CLIP. ` + (m.lit ? 'The image lane is frozen and its outputs are computed once, ahead of training; only the text lane, t and b learn.' : `A ${m.r} ${x} ${m.r} photo becomes ${np(m)} patch tokens, a caption at most ${m.tok} tokens; both end as one ${D(m)}-number unit vector.`),
        batch: `Step 1: a batch of N = ${m.batch} image-caption pairs. ` + (m.lit ? 'Images: already turned into embeddings by the frozen tower, read from disk.' : `Images ${SH.i0(m)}; captions are N strings.`),
        tokens: m.lit ? `Step 2: the B/8 tower cuts each photo into 8 ${x} 8-pixel patches (the paper does not give its input size, so the count is n). Caption: at most ${m.tok} ids (the Sec. 4.1 SigLiT setup, which Sec. 4.4 says it follows), embedded to ${SH.t1(m)} (width not stated).`
          : `Step 2: ${g} ${x} ${g} = ${np(m)} patches of ${m.P} ${x} ${m.P} ${x} 3 = ${m.P * m.P * 3} numbers, each mapped to 768 numbers: ${SH.i1(m)}. Caption: at most ${m.tok} sentencepiece ids, embedded to ${SH.t1(m)}.`,
        encode: m.lit ? `Step 3: frozen ViT-B/8 (12 layers, width 768), then a text transformer of 12 layers. Each ends with one vector per example, width d (not stated for SigLiT).`
          : `Step 3: ViT-B (12 layers, width 768, 12 heads) and a B-sized text transformer (width 768). Each lane pools to one vector: ${SH.i2(m)} and ${SH.t2(m)}.`,
        embed: `Step 4: each row is divided by its length: two ${SH.i3(m)} matrices. A dot product is now a cosine, between -1 and 1.`,
        sim: `Step 5: logits = t (x·y) + b, one N ${x} N table (taking 32 k as 32,768: about ${fmt(cells / 1e6)} million cells), but it never exists at once: it is computed one block at a time.` + (per ? ` Over ${m.chips} chips (one device per chip, our reading) each block is ${fmt(per)} ${x} ${fmt(per)} = ${(per * per / 1e6).toFixed(1)} million cells.` : ' The paper gives no device count for this run.'),
        loss: 'Step 6: every cell is its own yes/no question: log-sigmoid of (label ' + x + ' logit), label +1 on the diagonal and -1 elsewhere, summed over all N' + x + 'N cells and divided by N. One number.',
        backward: m.lit ? 'Backward: the gradient reaches t, b and the whole text lane. The image tower is brown and gets nothing: it is not even run during training.'
          : 'Backward: each cell sends back its own error, sigmoid(logit) minus 1 on the diagonal, 0 elsewhere, divided by N. Red outlines: everything learns, including t and b.',
        update: m.lit ? 'Update: LION, peak learning rate 1e-4 after 6.5k warm-up steps, weight decay 1e-7, cosine to zero; 65,000 steps at batch 32 k on four TPU-v4 chips.'
          : 'Update: ScalingViT-Adafactor with beta2 = 0.95, learning rate 0.001, weight decay 0.0001, cosine schedule.' + (m === V.b16x ? ` This recipe runs 40 billion examples at 32 k, about ${(40e9 / 32768 / 1e6).toFixed(2)} million steps.` : ''),
      }[s];
    },
  });
})();
</script>
