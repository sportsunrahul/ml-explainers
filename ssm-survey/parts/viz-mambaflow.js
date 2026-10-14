<script>
(() => {
  const x = '×';
  // Layer counts and widths: official README table (code-only). E = 2 from Sec. 3.4; N = 16 and conv width 4 are README defaults.
  const V = {
    m130: { name: 'Mamba-130M', L: 24, D: 768 }, m370: { name: 'Mamba-370M', L: 48, D: 1024 }, m790: { name: 'Mamba-790M', L: 48, D: 1536 },
    m1p4: { name: 'Mamba-1.4B', L: 48, D: 2048 }, m2p8: { name: 'Mamba-2.8B', L: 64, D: 2560 },
  };
  const E = 2, N = 16, K = 4, f = n => n.toLocaleString('en-US'), ED = m => E * m.D;
  const SH = {
    tok: m => `B${x}L ids -> B${x}L${x}${m.D}`, inx: m => `B${x}L${x}${ED(m)}`, inz: m => `B${x}L${x}${ED(m)}`,
    conv: m => `B${x}L${x}${ED(m)}`, sel: m => `Δ: B${x}L${x}${ED(m)}|B, C: B${x}L${x}${N}`,
    scan: m => `B${x}L${x}${ED(m)}|state B${x}${ED(m)}${x}${N}`, gate: m => `B${x}L${x}${ED(m)}`,
    out: m => `B${x}L${x}${m.D}`, head: () => `B${x}L${x}V`, loss: () => 'one number' };
  const SUB = { conv: () => `width ${K} (code-only), SiLU`, out: m => `${m.L} blocks, width ${m.D}`, inx: () => `E = ${E}` };
  const PARAMS = { tok: 'embedding', inx: 'in_proj, x half', inz: 'in_proj, z half', conv: 'conv weights', sel: 's_B, s_C, s_Δ,|Δ bias', scan: 'A', out: 'out_proj, norm', head: 'final norm,|LM head' };
  ArchFlow.mount({
    prefix: 'mambaflow', grid: { cols: 9, rows: 2 },
    boxes: {
      tok: { c: 0, r: .5, lane: 'blue', title: ['tokens,', 'norm'], glyph: 'seq', pattern: 'bbbbbb' },
      inx: { c: 1, r: 0, lane: 'blue', title: ['in_proj', 'main branch'], glyph: 'layers' },
      inz: { c: 1, r: 1, lane: 'green', title: ['in_proj', 'gate branch'], glyph: 'layers' },
      conv: { c: 2, r: 0, lane: 'blue', title: ['causal conv1d', '+ SiLU'], glyph: 'layers' },
      sel: { c: 3, r: 0, lane: 'yellow', title: ['x_proj:', 'Δ, B, C'], glyph: 'vector' },
      scan: { c: 4, r: 0, lane: 'yellow', title: ['selective', 'scan'], glyph: 'gate' },
      gate: { c: 5, r: .5, lane: 'green', title: ['y · SiLU(z)'], glyph: 'gate' },
      out: { c: 6, r: .5, lane: 'mixed', title: ['out_proj', '+ residual'], glyph: 'layers' },
      head: { c: 7, r: .5, lane: 'green', title: ['LM head'], glyph: 'vector' },
      loss: { c: 8, r: .5, lane: 'red', title: ['cross-entropy'], glyph: 'loss' },
    },
    edges: [['tok', 'inx'], ['tok', 'inz'], ['inx', 'conv'], ['conv', 'sel'], ['sel', 'scan'], ['scan', 'gate'], ['inz', 'gate'], ['gate', 'out'], ['out', 'head'], ['head', 'loss']],
    stages: [
      { key: 'tokens', show: ['tok'] }, { key: 'expand', show: ['inx', 'inz'] }, { key: 'conv', show: ['conv'] }, { key: 'select', show: ['sel'] },
      { key: 'scan', show: ['scan'] }, { key: 'gate', show: ['gate', 'out'] }, { key: 'loss', show: ['head', 'loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'm2p8',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => SUB[id] ? SUB[id](m) : null,
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => ({
      overview: `${m.name} (sizes from the README, code-only): ${m.L} blocks of width ${m.D}; inside each block the branches are ${ED(m)} wide and the scan keeps a state of ${ED(m)} ${x} ${N} = ${f(ED(m) * N)} numbers per sequence.`,
      tokens: `Step 1: token ids B ${x} L become vectors B ${x} L ${x} ${m.D}; RMSNorm on the residual stream before each block.`,
      expand: `Step 2: two projections ${m.D} -> ${ED(m)}, one per branch: 2 ${x} ${m.D} ${x} ${ED(m)} = ${f(2 * m.D * ED(m))} weights per block.`,
      conv: `Step 3: each of the ${ED(m)} channels mixes its last ${K} inputs (causal, code-only width), then SiLU. Shape stays B ${x} L ${x} ${ED(m)}.`,
      select: `Step 4: per token, B and C of length ${N} (shared by all ${ED(m)} channels) and one Δ per channel, B ${x} L ${x} ${ED(m)}, through a softplus.`,
      scan: `Step 5: per token, Abar = exp(ΔA) and Bbar by zero-order hold, B ${x} L ${x} ${ED(m)} ${x} ${N} in on-chip memory only; out comes y, B ${x} L ${x} ${ED(m)}.`,
      gate: `Step 6: y times SiLU(z), then ${ED(m)} -> ${m.D} (${f(ED(m) * m.D)} weights) and add to the residual. Repeated ${m.L} times.`,
      loss: 'Step 7: final norm and LM head give B x L x V logits; the loss is the cross-entropy of the next token at every position, averaged: one number.'.replace(/ x /g, ` ${x} `),
      backward: 'Backward: the gradient reaches every box with weights (red). Nothing is frozen; A learns only through exp(ΔA).',
      update: 'Update: AdamW (betas .9 and .95, weight decay 0.1, clip 1.0), warmup then cosine to 1e-5. Init: A_n = -(n+1); Δ bias so Δ starts in [0.001, 0.1].',
    })[s],
  });
})();
</script>
