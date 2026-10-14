<script>
(() => {
  const x = '×';
  // 125M and 360M: Table 5 (heads, layers, N = P = 64, one B/C head); width from H x P = 2D (expansion 2, Sec 8.1), our reading.
  // 2.7B: width 2560 and 64 layers from App D.3 and Sec 9.2.3; P = N = 64 and one B/C group are assumptions (code-only README example d_state=64).
  const V = {
    m125: { name: 'Mamba-2 125M (Table 5)', D: 768, L: 24, H: 24, P: 64, N: 64, G: 1, src: 'Table 5' },
    m360: { name: 'Mamba-2 360M (Table 5)', D: 1024, L: 48, H: 32, P: 64, N: 64, G: 1, src: 'Table 5' },
    m27: { name: 'Mamba-2 2.7B (App D.3)', D: 2560, L: 64, H: 80, P: 64, N: 64, G: 1, src: 'App D.3; P, N and groups assumed' },
  };
  const E = m => m.H * m.P, W = m => 2 * E(m) + 2 * m.G * m.N + m.H, f = n => n.toLocaleString('en-US');
  const state = m => m.H * m.P * m.N;
  const SH = {
    u: m => `T${x}${m.D}`, proj: m => `T${x}${f(W(m))}`, z: m => `T${x}${E(m)}`, X: m => `T${x}${m.H}${x}${m.P}`,
    bcd: m => `B, C: T${x}${m.G}${x}${m.N}|Δ: T${x}${m.H}`, conv: m => `T${x}${m.H}${x}${m.P}`, disc: m => `a: T${x}${m.H}|B, C: T${x}${m.G}${x}${m.N}`,
    ssd: m => `T${x}${m.H}${x}${m.P}|state ${m.H}${x}${m.P}${x}${m.N}`, gate: m => `T${x}${E(m)}`, norm: m => `T${x}${E(m)}`, out: m => `T${x}${m.D}`, loss: () => 'one number' };
  const PARAMS = { proj: 'one matrix:|z, X, B, C, Δ rows', conv: 'conv weights', disc: 'A, one per head', norm: 'norm scale', out: 'W_out' };
  ArchFlow.mount({
    prefix: 'mamba2flow', grid: { cols: 7, rows: 3 },
    boxes: {
      u: { c: 0, r: 1, lane: 'blue', title: ['input u'], glyph: 'seq', pattern: 'bbbbbb', data: true },
      proj: { c: 1, r: 1, lane: 'mixed', title: ['one input', 'projection'], glyph: 'layers' },
      z: { c: 2, r: 0, lane: 'mixed', title: ['gate branch z'], glyph: 'vector' },
      X: { c: 2, r: 1, lane: 'blue', title: ['X, in H heads'], glyph: 'vector' },
      bcd: { c: 2, r: 2, lane: 'yellow', title: ['B, C, Δ'], glyph: 'vector' },
      conv: { c: 3, r: 1, lane: 'blue', title: ['conv1d', '+ SiLU'], glyph: 'layers' },
      disc: { c: 3, r: 2, lane: 'yellow', title: ['decay per head', 'a = exp(ΔA)'], glyph: 'gate' },
      ssd: { c: 4, r: 1, lane: 'yellow', title: ['SSD', '(chunked)'], glyph: 'matrix' },
      gate: { c: 5, r: 0, lane: 'green', title: ['gate', 'y · SiLU(z)'], glyph: 'gate' },
      norm: { c: 6, r: 0, lane: 'green', title: ['extra norm'], glyph: 'unit' },
      out: { c: 6, r: 1, lane: 'green', title: ['out_proj'], glyph: 'layers' },
      loss: { c: 6, r: 2, lane: 'red', title: ['next-token loss'], glyph: 'loss' },
    },
    edges: [['u', 'proj'], ['proj', 'z'], ['proj', 'X'], ['proj', 'bcd'], ['X', 'conv'], ['bcd', 'disc'], ['conv', 'ssd'], ['disc', 'ssd'],
      ['ssd', 'gate'], ['z', 'gate'], ['gate', 'norm'], ['norm', 'out'], ['out', 'loss']],
    stages: [
      { key: 'input', show: ['u'] }, { key: 'project', show: ['proj', 'z', 'X', 'bcd'] }, { key: 'conv', show: ['conv', 'disc'] },
      { key: 'ssd', show: ['ssd'] }, { key: 'gate', show: ['gate', 'norm'] }, { key: 'output', show: ['out', 'loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'm125',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'conv' ? `T${x}${m.H}${x}${m.P}|width 4 (code-only)` : id === 'loss' ? 'after the LM head' : null,
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => {
      const kv = 2 * m.D;
      return {
        overview: `${m.name}: width ${m.D}, ${m.L} blocks; inside each, ${m.H} heads of dimension P = ${m.P} with state size N = ${m.N}. Shapes are per sequence of T tokens.${m.src === 'Table 5' ? '' : ' Heads, P, N and the single B/C group are assumed here: the paper gives only the width and depth at this size.'}`,
        input: `Step 1: the residual stream enters the block as u, T ${x} ${m.D}.`,
        project: `Step 2: one matrix of ${m.D} ${x} ${f(W(m))} makes everything at once: z and X (${E(m)} each, twice the width), B and C (${m.G} group of ${m.N} each, shared by all ${m.H} heads) and one Δ per head (${m.H}). The width is our count from the paper's shapes.`,
        conv: `Step 3: a causal depthwise convolution of width 4 (code-only) and SiLU on X; Δ and the per-head scalar A give the decay a = exp(ΔA), one number per head per token.`,
        ssd: `Step 4: SSD runs ${m.H} heads in parallel, each y = (L ∘ C Bᵀ) X on its own P = ${m.P} channels with chunks of 64 (Listing 1's default). State per layer: ${m.H} ${x} ${m.P} ${x} ${m.N} = ${f(state(m))} numbers at any length, as much as an attention layer of width ${m.D} caches for ${state(m) / kv} tokens (${f(kv)} numbers per token).`,
        gate: `Step 5: multiply by SiLU(z), then the extra normalization over ${E(m)} channels, right before the output projection.`,
        output: `Step 6: out_proj maps ${E(m)} back to ${m.D} and the result is added to the residual stream. After all ${m.L} blocks, the LM head scores the next token; the loss is cross-entropy at every position.`,
        backward: `Backward: the gradient leaves the loss, crosses out_proj and the norm, splits at the gate into the z branch and the SSD output, runs back through SSD to X, B, C and the decay, and meets again at the input projection. Every box with weights is red: nothing is frozen.`,
        update: `Update: AdamW with gradient clip 1.0 and weight decay 0.1, warmup then cosine decay to 1e-5, peak learning rate 5x the GPT-3 value. One step changes the projection, conv, A, norm and out_proj of all ${m.L} blocks.`,
      }[s];
    },
  });
})();
</script>
