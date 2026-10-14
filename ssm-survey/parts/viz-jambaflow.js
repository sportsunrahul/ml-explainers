<script>
(() => {
  // Jamba (Sec 3.1) and Jamba-1.5-Large (Sec 2). Jamba publishes no width or head counts; its 1024 KV channels per
  // attention layer are back-solved from Table 1 (our reading). Large's head dim 128 = 8192 / 64 is also our reading.
  const V = {
    v01: { name: 'Jamba', blocks: 4, d: 'd', dn: null, kv: 'K, V: 1024 ch.|(our reading)', vocab: '64K', total: '52B', active: '12B', kvTok: 16, kv256: '4 GiB' },
    large: { name: 'Jamba-1.5-Large', blocks: 9, d: '8192', dn: 8192, kv: 'Q 64 x 128,|KV 8 x 128', vocab: 'V', total: '398B', active: '94B', kvTok: 36, kv256: '9 GiB' },
  };
  const LAYER = [null, 'm', 'm', 'm', 'm', 'a', 'm', 'm', 'm'];
  const boxes = {
    tok: { c: 0, r: 0, lane: 'blue', title: ['tokens'], glyph: 'seq', pattern: 'bbbbb', data: true },
    emb: { c: 0, r: 1, lane: 'blue', title: ['embedding'], glyph: 'vector' },
    head: { c: 9, r: 0, lane: 'green', title: ['norm +', 'LM head'], glyph: 'vector' },
    loss: { c: 9, r: 1, lane: 'red', title: ['next-token', 'loss'], glyph: 'loss' },
  };
  const edges = [['tok', 'emb'], ['emb', 'x1']];
  for (let l = 1; l <= 8; l++) {
    boxes['x' + l] = LAYER[l] === 'a' ? { c: l, r: 0, lane: 'mixed', title: [`${l}: attention`], glyph: 'matrix' } : { c: l, r: 0, lane: 'yellow', title: [`${l}: Mamba`], glyph: 'codebook' };
    boxes['f' + l] = l % 2 ? { c: l, r: 1, lane: 'mixed', title: ['MLP'], glyph: 'mlp' } : { c: l, r: 1, lane: 'mixed', title: ['MoE'], glyph: 'mlp' };
    edges.push(['x' + l, 'f' + l]);
    if (l < 8) edges.push(['f' + l, 'x' + (l + 1)]);
  }
  edges.push(['f8', 'head'], ['head', 'loss']);
  const act = m => `B x T x ${m.d}`;
  ArchFlow.mount({
    prefix: 'jambaflow', grid: { cols: 10, rows: 2 }, boxes, edges,
    stages: [
      { key: 'tokens', show: ['tok', 'emb'] },
      { key: 'mamba', show: ['x1', 'x2', 'x3', 'x4', 'x6', 'x7', 'x8'] },
      { key: 'attention', show: ['x5'] },
      { key: 'mlp', show: ['f1', 'f3', 'f5', 'f7'] },
      { key: 'moe', show: ['f2', 'f4', 'f6', 'f8'] },
      { key: 'head', show: ['head'] },
      { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'v01',
    shape: (id, m) => id === 'tok' ? 'B x T ids' : id === 'loss' ? 'one number' : id === 'head' ? `B x T x ${m.vocab}` : act(m),
    sub: (id, m) => id === 'x5' ? m.kv : /^f[2468]$/.test(id) ? 'top-2 of 16' : /^x/.test(id) ? 'state: size|not published' : null,
    params: id => ({ emb: 'embedding table', head: 'output matrix', x5: 'W_Q, W_K,|W_V, W_O', f1: 'MLP', f3: 'MLP', f5: 'MLP', f7: 'MLP' })[id]
      || (/^x/.test(id) ? 'Mamba|weights' : /^f/.test(id) ? 'router + the|2 chosen experts' : null),
    frozen: () => false,
    readout: (s, m) => {
      const L = 8 * m.blocks, nA = m.blocks, gib = m.kvTok * 262144 / 1048576;
      return {
        overview: `${m.name}: ${m.blocks} blocks of 8 layers = ${L} layers, ${nA} of them attention and ${L - nA} Mamba; ${L / 2} MoE layers (our reading of Fig 1). ${m.total} parameters stored, ${m.active} used per token.`,
        tokens: `Step 1: B sequences of T token ids (context up to 256K) become vectors: ${act(m)}.${m.dn ? '' : ' Jamba does not publish its width d.'} No position encoding anywhere: the Mamba layers carry order.`,
        mamba: `Step 2: seven of the eight layers mix the sequence with Mamba. Each keeps a fixed state per channel and passes ${act(m)} on. Neither paper gives the state size N or the expansion.`,
        attention: `Step 3: layer 5 is attention with grouped-query heads. Its cache is 4 KiB per token at 16 bits (${m.dn ? '2 x 8 KV heads x 128 x 2 bytes' : 'back-solved from Table 1, our reading'}); ${nA} such layers give ${m.kvTok} KiB per token, ${m.kv256} at 256K tokens.`,
        mlp: 'Step 4: after every mixer comes a channel mixer. On the odd layers it is one dense MLP (SwiGLU), the same for every token.',
        moe: `Step 5: on the even layers the MLP is replaced by 16 expert MLPs. A router scores them per token and only the top 2 run, so ${m.active} of ${m.total} parameters work on each token.`,
        head: `Step 6: the ${m.blocks} blocks run in sequence; a final norm and the output matrix give ${act(m).replace(m.d, m.vocab)} scores.`,
        loss: 'Step 7: at every position, the score of the true next token is graded (cross-entropy), plus the MoE load-balancing term the paper cites. Neither formula is written in the paper.',
        backward: 'Backward: the gradient reaches every layer, Mamba and attention alike. In an MoE layer, a token sends gradient only to the router and to the 2 experts it chose; the other 14 get none from that token.',
        update: 'Update: every weight is trainable. The Jamba paper does not state the optimizer, schedule or batch size; it trained on NVIDIA H100 GPUs with an in-house framework (FSDP, tensor, sequence and expert parallelism).',
      }[s];
    },
  });
})();
</script>
