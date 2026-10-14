<script>
(() => {
  const x = '×', P = 14, CW = 1024;   // ViT-L/14: patch 14, width 1024 (CLIP appendix table; same for the 336px model)
  // LLaVA v2 Sec. 4.2 + App. "Training Details"; LLaVA-1.5 v2 App. "Hyperparameters". Steps are computed as examples / batch x epochs.
  const V = {
    v1s1: { name: 'LLaVA, stage 1', r: 224, mlp: false, s2: false, llm: 'Vicuna', data: '595K caption pairs', n: 595e3, bs: 128, ep: 1, lr: '2e-3', opt: 'Adam', time: 'within 4 hours' },
    v1s2: { name: 'LLaVA, stage 2', r: 224, mlp: false, s2: true, llm: 'Vicuna', data: '158K instruction samples', n: 158e3, bs: 32, ep: 3, lr: '2e-5', opt: 'Adam', time: 'within 10 hours' },
    v15s1: { name: 'LLaVA-1.5, stage 1', r: 336, mlp: true, s2: false, llm: 'Vicuna v1.5', data: '558K caption pairs', n: 558e3, bs: 256, ep: 1, lr: '1e-3', opt: 'AdamW', time: '~6 hours' },
    v15s2: { name: 'LLaVA-1.5, stage 2', r: 336, mlp: true, s2: true, llm: 'Vicuna v1.5', data: '665K instruction mix', n: 665e3, bs: 128, ep: 1, lr: '2e-5', opt: 'AdamW', time: '~20 hours' },
  };
  const g = m => (m.r / P) ** 2;
  const SH = {
    img: m => `3${x}${m.r}${x}${m.r}`, clip: m => `${g(m) + 1}${x}${CW}`, grid: m => `${g(m)}${x}${CW}`, proj: m => `${g(m)}${x}d_LLM`,
    text: () => 'T token ids', embed: () => `T${x}d_LLM`, splice: m => `(${g(m)}+T)${x}d_LLM`, llm: m => `(${g(m)}+T)${x}d_LLM`, loss: () => 'one number' };
  const SUB = {
    clip: m => `${SH.clip(m)}|24 layers`, proj: m => `${SH.proj(m)}|${m.mlp ? '2-layer MLP' : 'one matrix W'}`, llm: m => `${SH.llm(m)}|${m.llm}`, loss: () => 'answer + STOP|tokens' };
  const steps = m => Math.round(m.n / m.bs) * m.ep;
  ArchFlow.mount({
    prefix: 'llavaflow', grid: { cols: 7, rows: 2 },
    boxes: {
      img: { c: 0, r: 0, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      clip: { c: 1, r: 0, lane: 'blue', title: ['CLIP', 'ViT-L/14'], glyph: 'layers' },
      grid: { c: 2, r: 0, lane: 'blue', title: ['grid before', 'last layer'], glyph: 'grid', cls: false },
      proj: { c: 3, r: 0, lane: 'green', title: ['projector'], glyph: 'mlp' },
      text: { c: 0, r: 1, lane: 'yellow', title: ['conversation'], glyph: 'text', data: true },
      embed: { c: 1, r: 1, lane: 'yellow', title: ['LLM token', 'embeddings'], glyph: 'tokens' },
      splice: { c: 4, r: .5, lane: 'mixed', title: ['splice'], glyph: 'seq', pattern: 'yybbbbbyyy' },
      llm: { c: 5, r: .5, lane: 'yellow', title: ['LLM'], glyph: 'layers' },
      loss: { c: 6, r: .5, lane: 'red', title: ['loss'], glyph: 'loss' },
    },
    edges: [['img', 'clip'], ['clip', 'grid'], ['grid', 'proj'], ['proj', 'splice'], ['text', 'embed'], ['embed', 'splice'], ['splice', 'llm'], ['llm', 'loss']],
    stages: [
      { key: 'inputs', show: ['img', 'text'] }, { key: 'encode', show: ['clip', 'grid'] }, { key: 'project', show: ['proj'] }, { key: 'tokens', show: ['embed'] },
      { key: 'splice', show: ['splice'] }, { key: 'llm', show: ['llm'] }, { key: 'loss', show: ['loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'v1s1',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => SUB[id] ? SUB[id](m) : null,
    params: (id, m) => ({ proj: m.mlp ? 'MLP weights' : `W: d_LLM${x}${CW}`, embed: m.s2 ? 'embedding table' : null, llm: m.s2 ? 'all LLM weights' : null })[id] || null,
    frozen: (id, m) => id === 'clip' || (!m.s2 && (id === 'embed' || id === 'llm')),
    readout: (s, m) => {
      const k = m.r / P;
      return {
        overview: `${m.name}: a ${m.r} ${x} ${m.r} image becomes ${g(m)} visual tokens of the LLM's width, spliced into the conversation. ${m.s2 ? 'The projector and the LLM learn.' : 'Only the projector learns.'} Pick a variant to change the shapes or the frozen set.`,
        inputs: `Step 1: one ${m.r} ${x} ${m.r} RGB image (${SH.img(m)}) and one conversation of T tokens. This variant trains on ${m.data}.`,
        encode: `Step 2: ${k} ${x} ${k} = ${g(m)} patches of 14 ${x} 14 pixels, plus the [class] token: ${SH.clip(m)} through 24 layers of width ${CW}. The grid is read one layer before the end: ${SH.grid(m)} (our reading: the [class] token is not part of the grid).`,
        project: m.mlp ? `Step 3: a two-layer MLP maps each of the ${g(m)} patch vectors from ${CW} to d_LLM (hidden width and activation not stated): ${SH.proj(m)}.`
          : `Step 3: one matrix W (d_LLM ${x} ${CW}) applied to every patch vector: H_v = W Z_v, ${SH.grid(m)} -> ${SH.proj(m)}.`,
        tokens: `Step 4: the conversation (system message, Human / Assistant turns, each closed by <STOP>${m.mlp ? '' : ' = ###'}) becomes T ids, embedded by the LLM's own table: ${SH.embed()}.`,
        splice: `Step 5: the ${g(m)} visual tokens take the image's place before or after the first question (Eq. 2): one sequence of ${g(m)} + T vectors, ${SH.splice(m)}.`,
        llm: `Step 6: ${m.llm} reads the whole sequence with causal self-attention; its depth and width d_LLM are not given in these papers. Output ${SH.llm(m)}, then next-token logits.`,
        loss: 'Step 7: cross-entropy on the assistant\'s answer tokens and the <STOP> markers; the visual tokens and the human\'s words are context only, never targets.',
        backward: m.s2 ? 'Backward (stage 2): the gradient updates every LLM layer on its way down, then the projector. It stops at CLIP, which stays frozen.'
          : 'Backward (stage 1): the gradient runs down through the whole frozen LLM (no update) only to reach the projector. It stops at CLIP.',
        update: `Update: ${m.opt}, lr ${m.lr}, batch ${m.bs}, ${m.ep} epoch${m.ep > 1 ? 's' : ''} over ${m.data}: about ${steps(m).toLocaleString('en-US')} steps (computed), ${m.time} on 8 A100s.`,
      }[s];
    },
  });
})();
</script>
