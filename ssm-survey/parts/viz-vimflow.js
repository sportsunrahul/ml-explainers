<script>
(() => {
  const x = '×';
  // Vim Sec. 3.4: D and E for Ti and S; L = 24 blocks and N = 16 for all sizes. Params and top-1 from Table 1.
  const V = {
    ti: { name: 'Vim-Ti', D: 192, E: 384, N: 16, L: 24, P: 16, r: 224, params: '7M', top1: '76.1' },
    s: { name: 'Vim-S', D: 384, E: 768, N: 16, L: 24, P: 16, r: 224, params: '26M', top1: '80.3' },
  };
  const n = m => (m.r / m.P) ** 2, M = m => n(m) + 1, st = m => (m.E * m.N).toLocaleString('en-US');
  const SH = {
    img: m => `B${x}${m.r}${x}${m.r}${x}3`,
    embed: m => `${n(m)} + 1 tokens|B${x}${M(m)}${x}${m.D}`,
    inproj: m => `x, z: B${x}${M(m)}${x}${m.E}`,
    fwd: m => `state ${m.E}${x}${m.N}|y: B${x}${M(m)}${x}${m.E}`,
    bwd: m => `reversed order|y: B${x}${M(m)}${x}${m.E}`,
    gate: m => `B${x}${M(m)}${x}${m.D}|repeat ${x}${m.L}`,
    head: m => `B${x}${m.D} -> B${x}1000`,
    loss: () => 'one number',
  };
  const PARAMS = {
    embed: 'W, class token,|position embeddings', inproj: 'norm, Linear x,|Linear z',
    fwd: 'Conv1d, Linear B, C, Δ,|A, Δ bias', bwd: 'Conv1d, Linear B, C, Δ,|A, Δ bias',
    gate: 'Linear T', head: 'norm, MLP',
  };
  ArchFlow.mount({
    prefix: 'vimflow', grid: { cols: 7, rows: 2 },
    boxes: {
      img: { c: 0, r: .5, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      embed: { c: 1, r: .5, lane: 'blue', title: ['patches + class', 'token + positions'], glyph: 'seq', pattern: 'bbbbgbbbb' },
      inproj: { c: 2, r: .5, lane: 'mixed', title: ['norm, then', 'x and z'], glyph: 'layers' },
      fwd: { c: 3, r: 0, lane: 'yellow', title: ['forward scan'], glyph: 'layers' },
      bwd: { c: 3, r: 1, lane: 'yellow', title: ['backward scan'], glyph: 'layers' },
      gate: { c: 4, r: .5, lane: 'green', title: ['gate by z, add,', 'project, residual'], glyph: 'gate' },
      head: { c: 5, r: .5, lane: 'green', title: ['class-token', 'head'], glyph: 'mlp' },
      loss: { c: 6, r: .5, lane: 'red', title: ['cross-entropy'], glyph: 'loss' },
    },
    edges: [['img', 'embed'], ['embed', 'inproj'], ['inproj', 'fwd'], ['inproj', 'bwd'], ['inproj', 'gate'], ['fwd', 'gate'], ['bwd', 'gate'], ['gate', 'head'], ['head', 'loss']],
    stages: [
      { key: 'image', show: ['img'] }, { key: 'tokens', show: ['embed'] }, { key: 'inproj', show: ['inproj'] }, { key: 'scans', show: ['fwd', 'bwd'] },
      { key: 'gate', show: ['gate'] }, { key: 'head', show: ['head'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'ti',
    shape: (id, m) => SH[id](m),
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => ({
      overview: `${m.name}: a ${m.r} ${x} ${m.r} image becomes ${n(m)} patch tokens plus 1 class token = ${M(m)} tokens of width ${m.D}, through ${m.L} blocks of two scans (state ${m.E} ${x} ${m.N} = ${st(m)} numbers per scan), then a head on the class token. ${m.params} parameters, ${m.top1}% top-1 (Table 1).`,
      image: `Step 1: B images, ${SH.img(m)}.`,
      tokens: `Step 2: ${n(m)} patches of ${m.P} ${x} ${m.P} ${x} 3 = ${m.P * m.P * 3} numbers, each times W (${m.P * m.P * 3} ${x} ${m.D}): B${x}${n(m)}${x}${m.D}. Class token at the middle, then E_pos (${M(m)} ${x} ${m.D}) added: B${x}${M(m)}${x}${m.D}.`,
      inproj: `Step 3: norm, then two linear maps to E = 2D = ${m.E}: x and z, each B${x}${M(m)}${x}${m.E}.`,
      scans: `Step 4: per direction, Conv1d + SiLU, then B and C (B${x}${M(m)}${x}${m.N}) and Δ (B${x}${M(m)}${x}${m.E}) from the tokens; the scan carries h of ${m.E} ${x} ${m.N} = ${st(m)} numbers per image, whatever the length.`,
      gate: `Step 5: y_fwd ${x} SiLU(z) + y_bwd ${x} SiLU(z), projected ${m.E} -> ${m.D}, plus the residual: B${x}${M(m)}${x}${m.D}. Repeated ${m.L} times.`,
      head: `Step 6: the class token's output, B${x}${m.D}, normalized, then an MLP to 1000 class scores.`,
      loss: 'Step 7: cross-entropy against the label (smoothed, mixed up). Only the class token position is graded.',
      backward: `Backward: loss -> head -> class token in block ${m.L} -> both scans -> every token, block by block, down to the class token, the position embeddings and W. All trainable, nothing frozen.`,
      update: 'Update: AdamW (momentum 0.9, weight decay 0.05), batch 1024, 300 epochs, learning rate 1e-3 with cosine decay, EMA weights; 8 A800 GPUs.',
    }[s]),
  });
})();
</script>
