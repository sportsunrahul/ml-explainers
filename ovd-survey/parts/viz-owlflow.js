<script>
(() => {
  const x = ' x ';
  // Detection resolution and patch size: OWL-ViT Table A3 (CLIP rows) and OWLv2 Sec. 4.1. Tokens = (r / P)^2, printed for
  // B/32 at 768 (OWL-ViT Sec. 3.1) and L/14 at 1008 (OWLv2 Sec. 3.2), computed (our reading) for the other two.
  // Widths d and d_q are not printed in either paper, so they stay symbolic.
  const V = {
    b32: { name: 'CLIP B/32 at 768', P: 32, r: 768, lr: '5e-5', steps: '140k', how: 'printed in OWL-ViT Sec. 3.1' },
    b16: { name: 'CLIP B/16 at 768', P: 16, r: 768, lr: '5e-5', steps: '140k', how: 'computed, our reading' },
    l14: { name: 'CLIP L/14 at 840', P: 14, r: 840, lr: '2e-5', steps: '70k', how: 'computed, our reading' },
    v2: { name: 'OWLv2 L/14 at 1008', P: 14, r: 1008, lr: '2e-5', how: 'printed in OWLv2 Sec. 3.2', v2: true },
  };
  const g = m => m.r / m.P, n = m => g(m) ** 2;
  const SH = {
    img: m => `${m.r}${x}${m.r}${x}3`, vit: m => `${n(m)}${x}d`, tok: m => `${n(m)}${x}d`, cls: m => `${n(m)}${x}d_q`, box: m => `${n(m)}${x}4`,
    txt: () => 'Q strings', tenc: () => `Q${x}d_q`, logit: m => `${n(m)}${x}Q`, loss: () => 'one number' };
  const SUB = {
    vit: m => `${g(m)}${x}${g(m)} = ${n(m)} tokens`, tok: m => `${n(m)}${x}d|pooling removed`, cls: m => `${n(m)}${x}d_q|one per token`,
    box: m => `${n(m)}${x}4|offset from own patch`, txt: () => 'Q strings|16 tokens max each', tenc: () => `Q${x}d_q|each query alone`,
    logit: m => `${n(m)}${x}Q|sigmoid per entry`, loss: () => 'L1 + gIoU|+ focal sigmoid' };
  const PARAMS = { vit: 'patch embedding,|all layers', tok: null, cls: 'linear projection', box: 'small MLP', tenc: 'all layers,|low lr' };
  ArchFlow.mount({
    prefix: 'owlflow', grid: { cols: 6, rows: 3 },
    boxes: {
      img: { c: 0, r: 1, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      vit: { c: 1, r: 1, lane: 'blue', title: ['ViT image', 'tower'], glyph: 'layers' },
      tok: { c: 2, r: 1, lane: 'blue', title: ['every output', 'token'], glyph: 'grid', cls: false },
      cls: { c: 3, r: 1, lane: 'blue', title: ['class head'], glyph: 'vector' },
      box: { c: 3, r: 2, lane: 'blue', title: ['box MLP'], glyph: 'mlp' },
      txt: { c: 2, r: 0, lane: 'yellow', title: ['queries'], glyph: 'text', data: true },
      tenc: { c: 3, r: 0, lane: 'yellow', title: ['text tower'], glyph: 'layers' },
      logit: { c: 4, r: 1, lane: 'green', title: ['scores'], glyph: 'matrix' },
      loss: { c: 5, r: 1, lane: 'red', title: ['matching', '+ loss'], glyph: 'loss' },
    },
    edges: [['img', 'vit'], ['vit', 'tok'], ['tok', 'cls'], ['tok', 'box'], ['txt', 'tenc'], ['tenc', 'logit'], ['cls', 'logit'], ['logit', 'loss'], ['box', 'loss']],
    stages: [
      { key: 'image', show: ['img'] }, { key: 'vit', show: ['vit'] }, { key: 'nopool', show: ['tok'] }, { key: 'heads', show: ['cls', 'box'] },
      { key: 'queries', show: ['txt', 'tenc'] }, { key: 'scores', show: ['logit'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'l14',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => (SUB[id] ? SUB[id](m) : null),
    params: id => (id in PARAMS ? PARAMS[id] : null),
    frozen: () => false,
    readout: (s, m) => {
      const N = n(m), G = g(m);
      return {
        overview: `${m.name}: a ${m.r} x ${m.r} image becomes ${G} x ${G} = ${N} tokens (${m.how}). Each token gives one box and one embedding; each query gives one embedding; the scores form an ${N} x Q matrix.`,
        image: `Step 1: a batch of ${m.r} x ${m.r} x 3 images, padded to a square with gray at the bottom and right.`,
        vit: `Step 2: ${N} patches of ${m.P} x ${m.P} x 3 = ${m.P * m.P * 3} numbers each, embedded to width d with interpolated position embeddings; the transformer layers keep ${N} x d.`,
        nopool: `Step 3: CLIP would pool these ${N} tokens into one image vector. OWL-ViT removes the pooling and the final projection, so all ${N} tokens go on to the heads (${N} x d).`,
        heads: `Step 4: the class head maps each token to an embedding (${N} x d_q); the box MLP gives four numbers per token (${N} x 4), an offset from a default box on the token's own patch.` +
          (m.v2 ? ' OWLv2 in training: half of the patches are dropped (2592 kept, our reading) and boxes are computed only for the top 512 tokens by objectness; at inference all 5184 are used.' : ''),
        queries: 'Step 5: each query string (at most 16 text tokens) goes through the text tower alone, so Q queries give Q x d_q. No query sees the image or another query.',
        scores: `Step 6: every token embedding dotted with every query embedding: ${N} x Q logits, then a sigmoid on each entry. No softmax across queries.`,
        loss: `Step 7: bipartite matching pairs each labelled object with one of the ${N} predictions; all others learn "no object". Loss: L1 + gIoU on matched boxes, focal sigmoid cross-entropy on all scores, equal weights.`,
        backward: 'Backward: the gradient reaches the box MLP, the class head, every ViT layer down to the patch embedding, and, through the scores, the text tower. Nothing is frozen.',
        update: m.v2 ? 'Update (OWLv2): image lr 2e-5 for L/14, text lr 2e-6 for all models, an Adafactor variant with an inverse square-root schedule.'
          : `Update (OWL-ViT): Adam with cosine decay, image lr ${m.lr}, ${m.steps} steps at batch 256, gradient clipped to norm 1.0 per example. The text tower trains at a much lower lr.`,
      }[s];
    },
  });
})();
</script>
