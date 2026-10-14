<script>
(() => {
  const x = '×';
  // Chameleon v2: Sec. 2.1 (tokenizers), Table 1 "llama_models" (context, GQA, lr, dropout, z-loss, QK-Norm), Sec. 2.3 (batch, norm reordering), Table 2 (GPUs).
  const V = {
    '7b': { name: 'Chameleon-7B', gqa: false, drop: '0.1', reorder: false, batch: '2^23', btok: 2 ** 23, gpus: 1024, gpuh: '856,481', frac: .55 },
    '34b': { name: 'Chameleon-34B', gqa: true, drop: '0.0', reorder: true, batch: '3 ' + x + ' 2^22', btok: 3 * 2 ** 22, gpus: 3072, gpuh: '4,282,407', frac: .8 },
  };
  const VOCAB = '65,536', CTX = 4096;
  const SH = {
    i0: () => `3${x}512${x}512`, tok: () => '1024 ids', t0: () => 'text', bpe: () => 'T_text ids',
    seq: () => `T${x}d`, tf: () => `T${x}d`, out: () => `T${x}${VOCAB}`, loss: () => 'CE + z-loss' };
  const PARAMS = { seq: 'embedding table', tf: 'all layers', out: 'output layer' };
  const fmt = n => Math.round(n).toLocaleString('en-US');
  ArchFlow.mount({
    prefix: 'chameleonflow', grid: { cols: 6, rows: 2 },
    boxes: {
      i0: { c: 0, r: 0, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      tok: { c: 1, r: 0, lane: 'blue', title: ['image', 'tokenizer'], glyph: 'codebook' },
      t0: { c: 0, r: 1, lane: 'yellow', title: ['text'], glyph: 'text', data: true },
      bpe: { c: 1, r: 1, lane: 'yellow', title: ['BPE'], glyph: 'tokens', data: true },
      seq: { c: 2, r: .5, lane: 'mixed', title: ['one sequence', '+ embed'], glyph: 'seq', pattern: 'yyybbbbbyy' },
      tf: { c: 3, r: .5, lane: 'mixed', title: ['transformer'], glyph: 'layers' },
      out: { c: 4, r: .5, lane: 'mixed', title: ['logits'], glyph: 'vector' },
      loss: { c: 5, r: .5, lane: 'red', title: ['loss'], glyph: 'loss' },
    },
    edges: [['i0', 'tok'], ['tok', 'seq'], ['t0', 'bpe'], ['bpe', 'seq'], ['seq', 'tf'], ['tf', 'out'], ['out', 'loss']],
    stages: [
      { key: 'inputs', show: ['i0', 't0'] }, { key: 'tokenize', show: ['tok', 'bpe'] }, { key: 'sequence', show: ['seq'] },
      { key: 'transformer', show: ['tf'] }, { key: 'logits', show: ['out'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: '7b',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'tf' ? (m.reorder ? 'QK-Norm, GQA,|norm reorder' : 'QK-Norm,|dropout 0.1') : null,
    params: id => PARAMS[id] || null,
    frozen: id => id === 'tok',
    readout: (s, m) => ({
      overview: `${m.name}: images and text both become integer ids from one ${VOCAB}-entry vocabulary, one transformer reads the mixed sequence (at most ${CTX} tokens, the 4k context), and the loss grades every next id. d and the layer count are not given in the paper.`,
      inputs: 'Step 1: a document: text, and images resized and center-cropped to 512 ' + x + ' 512. Any mix: text only, an image-caption pair (image first half the time), or interleaved.',
      tokenize: `Step 2: the image tokenizer (trained before, separately) maps 3${x}512${x}512 pixels to 1024 ids, each one of 8192 codebook entries; if the ids tile a square, that is 32 ${x} 32, one per 16 ${x} 16 pixels. BPE maps text to ids from the same vocabulary.`,
      sequence: `Step 3: one sequence of T ids (T <= ${CTX}); each id looks up a row of one ${VOCAB} ${x} d embedding table: T${x}d. A 4k context holds fewer than 4 images' worth of ids once text or delimiters are counted (our arithmetic).`,
      transformer: `Step 4: a LLaMa-2-style decoder (RMSNorm, SwiGLU, RoPE), causal attention, QK-Norm in every layer. ${m.name}: ${m.gqa ? 'grouped-query attention, norm reordering, no dropout' : 'no GQA, standard norm order, dropout 0.1'}. Shape T${x}d throughout.`,
      logits: `Step 5: at every position, one logit per vocabulary entry: T${x}${VOCAB}. Text words and image codes compete in the same softmax.`,
      loss: 'Step 6: next-token cross-entropy at every position, text or image, plus 10^-5 log^2 Z per position (the z-loss). One number.',
      backward: 'Backward: loss -> output layer -> every transformer layer -> the embedding rows of the ids that were used, text and image alike. It stops there: the tokenizer produced integers, and an integer has no gradient.',
      update: `Update: AdamW (0.9, 0.95), lr 1e-4, 4000 warm-up steps, exponential decay to 0, weight decay 0.1, clip 1.0. Batch ${m.batch} = ${fmt(m.btok)} tokens; 9.2T tokens / batch = about ${fmt(9.2e12 / m.btok)} steps (our arithmetic); the paper's "600k steps = ${Math.round(m.frac * 100)}%" gives about ${fmt(6e5 / m.frac)}. ${m.gpus} GPUs, ${m.gpuh} GPU hours.`,
    }[s]),
  });
})();
</script>
