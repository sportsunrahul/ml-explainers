<script>
(() => {
  const x = '×', n = v => v.toLocaleString('en-US');
  // Picture 3 of the paper's architecture figure. Its three image token counts all equal (w/32)(h/32), which with the
  // stated 2x2 merge implies 16-pixel patches; the text never states the patch size.
  const IMG = { w: 1440, h: 800 }, P = 16, gw = IMG.w / P, gh = IMG.h / P, NP = gw * gh, NV = NP / 4;
  const VID = { w: 736, h: 448 }, NVF = (VID.w / 32) * (VID.h / 32);
  // Table training_setup: what trains, token budget, sequence length.
  const V = {
    s0: { name: 'S0', all: false, budget: '67B', seq: 8192, goal: 'vision-language alignment' },
    s1: { name: 'S1', all: true, budget: '~1T', seq: 8192, goal: 'multimodal pre-training' },
    s2: { name: 'S2', all: true, budget: '~1T', seq: 32768, goal: 'long-context pre-training' },
    s3: { name: 'S3', all: true, budget: '100B', seq: 262144, goal: 'ultra-long-context adaptation' },
  };
  const S = `(L+${n(NV)})|${x}d_LLM`;
  const SH = {
    patch: () => `${n(NP)}${x}d_ViT`, vit: () => `${n(NP)}${x}d_ViT`,
    merger: () => `${n(NV)}|${x}d_LLM`, ds1: () => `${n(NV)}|${x}d_LLM`, ds2: () => `${n(NV)}|${x}d_LLM`, ds3: () => `${n(NV)}|${x}d_LLM`,
    text: () => `L${x}d_LLM`, l1: () => '+ DeepStack 1', l2: () => '+ DeepStack 2', l3: () => '+ DeepStack 3', rest: () => S, loss: () => 'one number' };
  const PARAMS = { patch: 'patch embed|+ pos emb', vit: 'all blocks', merger: '2-layer MLP', ds1: 'its merger', ds2: 'its merger', ds3: 'its merger',
    text: 'embeddings', l1: 'layer 1', l2: 'layer 2', l3: 'layer 3', rest: 'layers 4..N' };
  const MERGERS = { merger: 1, ds1: 1, ds2: 1, ds3: 1 };
  ArchFlow.mount({
    prefix: 'qwen3vlflow', grid: { cols: 7, rows: 4 },
    // Placed so no two frozen boxes are neighbours in the transposed mobile layout (their two-line labels would collide).
    boxes: {
      patch: { c: 0, r: 1, lane: 'blue', title: ['image,', '16 px patches'], glyph: 'grid', cls: false },
      vit: { c: 1, r: 1, lane: 'blue', title: ['SigLIP-2', 'ViT'], glyph: 'layers' },
      merger: { c: 2, r: 0, lane: 'green', title: ['merger', '2x2 MLP'], glyph: 'mlp' },
      ds3: { c: 2, r: 1, lane: 'green', title: ['DeepStack', 'merger 3'], glyph: 'mlp' },
      ds1: { c: 2, r: 2, lane: 'green', title: ['DeepStack', 'merger 1'], glyph: 'mlp' },
      ds2: { c: 2, r: 3, lane: 'green', title: ['DeepStack', 'merger 2'], glyph: 'mlp' },
      text: { c: 3, r: 0, lane: 'yellow', title: ['text tokens'], glyph: 'tokens' },
      l1: { c: 3, r: 2, lane: 'yellow', title: ['LLM layer 1'], glyph: 'layers' },
      l2: { c: 4, r: 3, lane: 'yellow', title: ['LLM layer 2'], glyph: 'layers' },
      l3: { c: 5, r: 1, lane: 'yellow', title: ['LLM layer 3'], glyph: 'layers' },
      rest: { c: 6, r: 1, lane: 'yellow', title: ['LLM layers', '4 to N'], glyph: 'layers' },
      loss: { c: 6, r: 0, lane: 'red', title: ['next-token', 'loss'], glyph: 'loss' },
    },
    edges: [['patch', 'vit'], ['vit', 'ds3'], ['vit', 'ds2'], ['vit', 'ds1'], ['vit', 'merger'],
      ['ds1', 'l1'], ['ds2', 'l2'], ['ds3', 'l3'], ['merger', 'l1'], ['text', 'l1'], ['l1', 'l2'], ['l2', 'l3'], ['l3', 'rest'], ['rest', 'loss']],
    stages: [
      { key: 'input', show: ['patch'] }, { key: 'vit', show: ['vit'] }, { key: 'merge', show: ['merger'] },
      { key: 'splice', show: ['text'] }, { key: 'deepstack', show: ['ds1', 'ds2', 'ds3'] }, { key: 'llm', show: ['l1', 'l2', 'l3', 'rest'] },
      { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 's0',
    shape: id => SH[id](),
    params: id => PARAMS[id] || null,
    frozen: (id, v) => !!PARAMS[id] && !v.all && !MERGERS[id],
    readout: (s, v) => ({
      overview: `Stage ${v.name} (${v.goal}): the figure's 1440 ${x} 800 picture becomes ${n(NP)} patches and ${n(NV)} visual tokens; three more copies of those ${n(NV)} positions arrive, from intermediate ViT layers, inside the first three LLM layers.`,
      input: `Input and patches: 3 ${x} ${IMG.h} ${x} ${IMG.w} at native size, cut into ${gh} ${x} ${gw} = ${n(NP)} patches of 16 ${x} 16 (derived from the figure, see text) and embedded to ${n(NP)} ${x} d_ViT with 2D-RoPE plus interpolated absolute positions.`,
      vit: `ViT (SigLIP2-SO-400M; SigLIP2-Large, 300M, for the 2B and 4B models): ${n(NP)} ${x} d_ViT in and out. Three intermediate levels are tapped on the way (layer indices not stated).`,
      merge: `Merger: each 2 ${x} 2 block of final ViT features goes through a two-layer MLP into one token at the LLM width: ${n(NP)} / 4 = ${n(NV)} ${x} d_LLM.`,
      splice: `Splice: text tokens (L ${x} d_LLM) and the ${n(NV)} visual tokens form one sequence of (L + ${n(NV)}) ${x} d_LLM; a video adds one text timestamp such as <3.0 seconds> before each temporal patch (for the figure's video, ${n(NVF)} tokens per temporal patch). Stage ${v.name} trains at sequence length ${n(v.seq)}.`,
      deepstack: `DeepStack: three more mergers turn three intermediate ViT levels into ${n(NV)} ${x} d_LLM each; each is added to the hidden states at the ${n(NV)} visual positions of LLM layer 1, 2 or 3. The sequence stays L + ${n(NV)} long.`,
      llm: `LLM (Qwen3, dense or MoE): (L + ${n(NV)}) ${x} d_LLM through every layer, with interleaved MRoPE; layers 1 to 3 also receive the DeepStack additions.`,
      loss: 'Loss: next-token cross-entropy, square-root-normalized per token; which positions are graded is not spelled out (our reading: text positions, as in Qwen2-VL).',
      backward: v.all ? `Backward, stage ${v.name}: everything trains; the gradient reaches the ViT by four routes, through the main merger and the three DeepStack mergers.`
        : 'Backward, stage S0: only the mergers train. The frozen LLM passes the gradient down to the four mergers; the frozen ViT stops it.',
      update: `Update, stage ${v.name}: ${v.budget} tokens at sequence length ${n(v.seq)}. The paper gives no optimizer, learning rate or batch size.`,
    })[s],
  });
})();
</script>
