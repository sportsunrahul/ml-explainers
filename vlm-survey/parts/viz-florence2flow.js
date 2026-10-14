<script>
(() => {
  const x = '×';
  // Florence-2 appendix Table 15 (DaViT stage dims/heads/params; encoder-decoder layers/width/params) and Sec. 6.1 Setup (lr, batch, samples).
  const V = {
    b: { name: 'Florence-2-B', dims: [128, 256, 512, 1024], heads: [4, 8, 16, 32], vp: 90, el: 6, dl: 6, D: 768, ep: 140, total: 232, lr: '1e-4', batch: 2048, hi: 0.5 },
    l: { name: 'Florence-2-L', dims: [256, 512, 1024, 2048], heads: [8, 16, 32, 64], vp: 360, el: 12, dl: 12, D: 1024, ep: 410, total: 771, lr: '1e-5', batch: 3072, hi: 0.1 },
  };
  const Dv = m => m.dims[3];
  const SH = {
    i0: () => `3${x}384${x}384`, i1: m => `N_v${x}${Dv(m)}`, i2: m => `N_v${x}${m.D}`, t0: () => 'prompt string', t1: m => `N_t${x}${m.D}`,
    cat: m => `(N_v+N_t)${x}${m.D}`, enc: m => `(N_v+N_t)${x}${m.D}`, y: () => 'L_y ids', dec: () => `L_y${x}|vocab|`, loss: () => 'one number' };
  const PARAMS = { i1: 'all 4 stages', i2: 'Linear, LN', t1: 'emb + loc rows', enc: 'all layers', dec: 'all layers' };
  const steps = m => Math.round(3e9 / m.batch).toLocaleString('en-US');
  ArchFlow.mount({
    prefix: 'florence2flow', grid: { cols: 7, rows: 2 },
    boxes: {
      i0: { c: 0, r: 0, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      i1: { c: 1, r: 0, lane: 'blue', title: ['DaViT'], glyph: 'layers' },
      i2: { c: 2, r: 0, lane: 'green', title: ['Linear', '+ LayerNorm'], glyph: 'vector' },
      t0: { c: 0, r: 1, lane: 'yellow', title: ['task prompt'], glyph: 'text', data: true },
      t1: { c: 1, r: 1, lane: 'yellow', title: ['tokens', '+ embed'], glyph: 'tokens' },
      cat: { c: 3, r: .5, lane: 'mixed', title: ['concat'], glyph: 'seq', pattern: 'bbbbbyyy' },
      enc: { c: 4, r: .5, lane: 'mixed', title: ['encoder'], glyph: 'layers' },
      y: { c: 5, r: 1, lane: 'yellow', title: ['targets,', 'shifted'], glyph: 'tokens', data: true },
      dec: { c: 5, r: 0, lane: 'yellow', title: ['decoder'], glyph: 'layers' },
      loss: { c: 6, r: 0, lane: 'red', title: ['loss'], glyph: 'loss' },
    },
    edges: [['i0', 'i1'], ['i1', 'i2'], ['i2', 'cat'], ['t0', 't1'], ['t1', 'cat'], ['cat', 'enc'], ['enc', 'dec'], ['y', 'dec'], ['dec', 'loss']],
    stages: [
      { key: 'inputs', show: ['i0', 't0'] }, { key: 'davit', show: ['i1'] }, { key: 'align', show: ['i2', 't1'] }, { key: 'concat', show: ['cat'] },
      { key: 'encoder', show: ['enc'] }, { key: 'decoder', show: ['y', 'dec'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'b',
    shape: (id, m) => SH[id](m),
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => ({
      overview: `${m.name}: a ${SH.i0(m)} image and a short task prompt go in; a sequence of L_y tokens (words and location tokens) comes out. N_v, N_t and L_y are not fixed numbers in the paper, so they stay symbols.`,
      inputs: `Step 1: one image, resized to ${SH.i0(m)} for main training (768 ${x} 768 in the short high-resolution phase), plus a task prompt string. The target answer is a token string too.`,
      davit: `Step 2: DaViT, 4 stages with widths ${m.dims.join(', ')}, blocks 1, 1, 9, 1, heads/groups ${m.heads.join(', ')}; ${m.vp}M parameters. Output: N_v visual tokens of width D_v = ${Dv(m)} (our reading: the last stage's width), ${SH.i1(m)}.`,
      align: `Step 3: V' = LayerNorm(Linear(V)): ${Dv(m)} -> ${m.D}, so ${SH.i2(m)}. The prompt's N_t tokens are embedded to the same width: ${SH.t1(m)}.`,
      concat: `Step 4: X = [V', T_prompt], visual tokens first: ${SH.cat(m)}.`,
      encoder: `Step 5: the transformer encoder, ${m.el} layers of width ${m.D}, attends over all N_v + N_t positions at once. Shape unchanged: ${SH.enc(m)}.`,
      decoder: `Step 6: the decoder, ${m.dl} layers of width ${m.D}, reads the target shifted right and cross-attends to the encoder output. It emits one row of logits per target position over the extended vocabulary: ${SH.dec(m)}, where the vocabulary is BART's text vocabulary plus the 1,000 location tokens (and any other added tokens; the paper gives no total).`,
      loss: 'Step 7: cross-entropy at each of the L_y target positions, summed (Eq. 1). A location token and a word are graded by the same formula. Nothing on the encoder side is graded.',
      backward: `Backward: from the loss into the decoder, across to the encoder, then split into the prompt embeddings and through the projection into DaViT. Nothing is frozen. Table 15's parts sum to ${m.vp + m.ep}M (${m.vp}M + ${m.ep}M); the paper's total is ${m.total}M.`,
      update: `Update: AdamW, cosine decay, peak lr ${m.lr}, 5,000 warm-up steps, batch ${m.batch}. 3 billion samples at 384 px is about ${steps(m)} steps (our arithmetic), then ${m.hi} billion samples at 768 px.`,
    }[s]),
  });
})();
</script>
