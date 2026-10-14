<script>
(() => {
  const x = '×', CTX = 76;
  // Rows of CLIP's "CLIP-ViT hyperparameters" appendix table: patch, input resolution, embedding dim, vision L/W/H, text L/W/H.
  const V = {
    b32: { name: 'ViT-B/32', P: 32, r: 224, de: 512, vl: 12, vw: 768, vh: 12, tl: 12, tw: 512, th: 8 },
    b16: { name: 'ViT-B/16', P: 16, r: 224, de: 512, vl: 12, vw: 768, vh: 12, tl: 12, tw: 512, th: 8 },
    l14: { name: 'ViT-L/14', P: 14, r: 224, de: 768, vl: 24, vw: 1024, vh: 16, tl: 12, tw: 768, th: 12 },
    l336: { name: 'ViT-L/14@336px', P: 14, r: 336, de: 768, vl: 24, vw: 1024, vh: 16, tl: 12, tw: 768, th: 12 },
  };
  const n = m => (m.r / m.P) ** 2 + 1;
  const SH = {
    i0: m => `N${x}3${x}${m.r}${x}${m.r}`, i1: m => `N${x}${n(m)}${x}${m.vw}`, i2: m => `N${x}${n(m)}${x}${m.vw}`, i3: m => `N${x}${m.vw}`, i4: m => `N${x}${m.de}`,
    t0: () => 'N strings', t1: m => `N${x}${CTX}${x}${m.tw}`, t2: m => `N${x}${CTX}${x}${m.tw}`, t3: m => `N${x}${m.tw}`, t4: m => `N${x}${m.de}`, sim: () => `N${x}N`, loss: () => 'one number' };
  const PARAMS = { i1: 'E, class, pos, LN', t1: 'token embeddings', i2: 'all layers', t2: 'all layers', i3: 'LN', t3: 'LN', i4: 'W_i', t4: 'W_t', sim: 't' };
  ArchFlow.mount({
    prefix: 'clipflow', grid: { cols: 7, rows: 2 },
    boxes: {
      i0: { c: 0, r: 0, lane: 'blue', title: ['photos'], glyph: 'photos', data: true },
      i1: { c: 1, r: 0, lane: 'blue', title: ['patches', '+ [class]'], glyph: 'grid' },
      i2: { c: 2, r: 0, lane: 'blue', title: ['ViT'], glyph: 'layers' },
      i3: { c: 3, r: 0, lane: 'blue', title: ['[class] out', '+ LN'], glyph: 'vector' },
      i4: { c: 4, r: 0, lane: 'blue', title: ['project', '+ L2 norm'], glyph: 'unit' },
      t0: { c: 0, r: 1, lane: 'yellow', title: ['captions'], glyph: 'text', data: true },
      t1: { c: 1, r: 1, lane: 'yellow', title: ['BPE tokens', '+ embed'], glyph: 'tokens' },
      t2: { c: 2, r: 1, lane: 'yellow', title: ['text', 'transformer'], glyph: 'layers' },
      t3: { c: 3, r: 1, lane: 'yellow', title: ['[EOS] out', '+ LN'], glyph: 'vector' },
      t4: { c: 4, r: 1, lane: 'yellow', title: ['project', '+ L2 norm'], glyph: 'unit' },
      sim: { c: 5, r: .5, lane: 'green', title: ['logits'], glyph: 'matrix' },
      loss: { c: 6, r: .5, lane: 'red', title: ['loss'], glyph: 'loss' },
    },
    edges: [['i0', 'i1'], ['i1', 'i2'], ['i2', 'i3'], ['i3', 'i4'], ['i4', 'sim'], ['t0', 't1'], ['t1', 't2'], ['t2', 't3'], ['t3', 't4'], ['t4', 'sim'], ['sim', 'loss']],
    stages: [
      { key: 'batch', show: ['i0', 't0'] }, { key: 'tokens', show: ['i1', 't1'] }, { key: 'encode', show: ['i2', 't2'] }, { key: 'pool', show: ['i3', 't3'] },
      { key: 'embed', show: ['i4', 't4'] }, { key: 'sim', show: ['sim'] }, { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'l14',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'i2' ? `${m.vl} layers` : id === 't2' ? `${m.tl} layers, masked` : null,
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => {
      const g = m.r / m.P;
      return {
        overview: `${m.name}: an image of ${m.r} ${x} ${m.r} pixels becomes ${g * g} patch tokens plus one [class] token; a caption becomes at most ${CTX} tokens. Both end as one ${m.de}-number unit vector.`,
        batch: `Step 1: a batch of N image-caption pairs. Images ${SH.i0(m)}; captions are N strings. In CLIP, N = 32,768.`,
        tokens: `Step 2: image cut into ${g} ${x} ${g} = ${g * g} patches of ${m.P} ${x} ${m.P} ${x} 3 = ${m.P * m.P * 3} numbers, each mapped to ${m.vw} numbers, plus the [class] token: ${SH.i1(m)}. Caption: ids N${x}${CTX}, embedded to ${SH.t1(m)}.`,
        encode: `Step 3: the vision transformer (${m.vl} layers, width ${m.vw}, ${m.vh} heads) and the text transformer (${m.tl} layers, width ${m.tw}, ${m.th} heads) run side by side. Shapes do not change inside an encoder.`,
        pool: `Step 4: keep one vector per example. Image: the [class] token's output, layer-normalized, ${SH.i3(m)}. Text: the [EOS] token's output, layer-normalized, ${SH.t3(m)}.`,
        embed: `Step 5: W_i is ${m.vw} ${x} ${m.de}, W_t is ${m.tw} ${x} ${m.de}. After the projection both sides are N${x}${m.de}, then each row is scaled to length 1.`,
        sim: `Step 6: logits = I_e T_e^T ${x} exp(t): one N ${x} N table. At N = 32,768 that is about 1.07 billion numbers per step.`,
        loss: 'Step 7: cross-entropy down every row (which caption?) and every column (which image?), averaged: one number.',
        backward: 'Backward: the gradient starts at the loss and flows right to left into both lanes. Red outlines mark what it updates; nothing in CLIP is frozen.',
        update: 'Update: every red-outlined parameter takes one AdamW step, including the temperature t, then the next batch starts.' +
          (m === V.l336 ? ' (ViT-L/14@336px is not trained this way from scratch: it is ViT-L/14 continued for one epoch at 336 px.)' : ''),
      }[s];
    },
  });
})();
</script>
