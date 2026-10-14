<script>
(() => {
  const x = '×', n = v => v.toLocaleString('en-US');
  // Inputs: the paper's own worked examples (Sec. 2.1 text and the framework figure). Sides are w x h in pixels.
  const V = {
    photo: { name: 'a 224 ' + x + ' 224 photo', w: 224, h: 224 },
    strip: { name: 'a 224 ' + x + ' 28 formula strip', w: 224, h: 28 },
    wide: { name: 'a 1260 ' + x + ' 700 landscape', w: 1260, h: 700 },
    page: { name: 'a 1092 ' + x + ' 8204 web page', w: 1092, h: 8204 },
    video: { name: 'a 16 s video at 644 ' + x + ' 336', w: 644, h: 336, sec: 16 },
  };
  // Training stages. Qwen2-VL: Sec. 2.2 para 1. Qwen-VL: Sec. 3.1-3.3 and the Appendix C hyperparameter table.
  const TS = {
    q2s1: { q2: true, n: 1, fz: { emb: 1, llm: 1 } },
    q2s2: { q2: true, n: 2, fz: {} },
    q2s3: { q2: true, n: 3, fz: { patch: 1, vit: 1, merge: 1 } },
    qvs1: { q2: false, n: 1, fz: { emb: 1, llm: 1 }, res: 224, lr: '2e-4', batch: '30720', steps: '50k', seq: 512 },
    qvs2: { q2: false, n: 2, fz: {}, res: 448, lr: '5e-5', batch: '4096', steps: '19k', seq: 2048 },
    qvs3: { q2: false, n: 3, fz: { patch: 1, vit: 1 }, res: 448, lr: '1e-5', batch: '128', steps: '8k', seq: 2048 },
  };
  const TK = Object.keys(TS);
  let ts = 'q2s1';
  const T = () => TS[ts];
  const shapes = m => {
    const t = T();
    if (!t.q2) { const g = t.res / 14; return { ok: !m.sec, img: `3${x}${t.res}${x}${t.res}`, np: g * g, g, vis: 256, seqv: 258 }; }
    const frames = m.sec ? 2 * m.sec : 2, tt = frames / 2, gh = m.h / 14, gw = m.w / 14, np = tt * gh * gw;
    return { ok: true, frames, tt, gh, gw, np, vis: np / 4, seqv: np / 4 + 2, img: m.sec ? `${frames}${x}3${x}${m.h}${x}${m.w}` : `3${x}${m.h}${x}${m.w}` };
  };
  const BOXES = {
    img: { c: 0, r: 0, lane: 'blue', title: ['image', 'or video'], glyph: 'photo', data: true },
    patch: { c: 1, r: 0, lane: 'blue', title: ['14 px patches', '(frame pairs)'], glyph: 'grid', cls: false },
    vit: { c: 2, r: 0, lane: 'blue', title: ['ViT', '+ 2D-RoPE'], glyph: 'layers' },
    merge: { c: 3, r: 0, lane: 'green', title: ['2x2 merge', '+ MLP'], glyph: 'mlp' },
    txt: { c: 2, r: 1, lane: 'yellow', title: ['prompt +', 'answer'], glyph: 'text', data: true },
    emb: { c: 3, r: 1, lane: 'yellow', title: ['token', 'embeddings'], glyph: 'tokens' },
    seq: { c: 4, r: .5, lane: 'mixed', title: ['splice', '+ M-RoPE ids'], glyph: 'seq', pattern: 'yyybbbbyyy', data: true },
    llm: { c: 5, r: .5, lane: 'yellow', title: ['Qwen2', 'LLM'], glyph: 'layers' },
    loss: { c: 6, r: .5, lane: 'red', title: ['next-token', 'loss'], glyph: 'loss' },
  };
  const TITLES = {
    q2: { patch: ['14 px patches', '(frame pairs)'], vit: ['ViT', '+ 2D-RoPE'], merge: ['2x2 merge', '+ MLP'], seq: ['splice', '+ M-RoPE ids'], llm: ['Qwen2', 'LLM'] },
    qv: { patch: ['14 px patches', '(one frame)'], vit: ['ViT-bigG', '(OpenCLIP)'], merge: ['cross-attn', '256 queries'], seq: ['splice', '<img> </img>'], llm: ['Qwen-7B', 'LLM'] },
  };
  const setTitles = () => { const t = TITLES[T().q2 ? 'q2' : 'qv']; for (const k in t) BOXES[k].title = t[k]; };
  const PARAMS = { patch: 'patch embed', vit: 'all layers', merge: 'MLP', emb: 'embeddings', llm: 'all layers' };
  const PARAMS_QV = { patch: 'patch embed', vit: 'all layers', merge: 'queries + attn', emb: 'embeddings', llm: 'all layers' };
  function shape(id, m) {
    const s = shapes(m);
    if (!s.ok) return id === 'img' ? 'video: no' : id === 'txt' ? 'L ids' : 'images only';
    return {
      img: s.img, patch: `${n(s.np)}${x}d_ViT`, vit: `${n(s.np)}${x}d_ViT`, merge: `${n(s.vis)}${x}d_LLM`,
      txt: 'L ids', emb: `L${x}d_LLM`, seq: `(L+${n(s.seqv)})|${x}d_LLM`, llm: `(L+${n(s.seqv)})|${x}d_LLM`, loss: 'one number',
    }[id];
  }
  function readout(stage, m) {
    const t = T(), s = shapes(m), who = t.q2 ? `Qwen2-VL stage ${t.n}` : `Qwen-VL stage ${t.n}`;
    if (!s.ok) return `${who}: Qwen-VL has no video pathway; at evaluation it reads video as four sampled frames. Pick an image, or switch to a Qwen2-VL stage.`;
    const tok = `${n(s.vis)} visual tokens + 2 markers = ${n(s.seqv)}`;
    if (!t.q2) return {
      overview: `${who}, input ${m.name}: resized to ${t.res} ${x} ${t.res} whatever its shape, ${n(s.np)} patches, always 256 visual tokens (258 with <img> and </img>).`,
      input: `Input: ${m.name} is resized to a fixed ${s.img} (stage ${t.n} runs at ${t.res} px). A tall page and a small photo get the same square.`,
      patch: `Patches: ${s.g} ${x} ${s.g} = ${n(s.np)} patches of 14 ${x} 14 ${x} 3 = 588 numbers, each embedded to d_ViT: ${n(s.np)} ${x} d_ViT.`,
      vit: `ViT (OpenCLIP ViT-bigG, 1.9B parameters): ${n(s.np)} ${x} d_ViT in, the same shape out.`,
      merge: `Adapter: 256 learned queries cross-attend to the ${n(s.np)} patch features (keys), with 2D absolute positions added: 256 ${x} d_LLM. ${s.np === 256 ? 'At 224 px that is no compression at all.' : `A ${s.np / 256}${x} compression.`}`,
      splice: `Splice: <img> 256 image tokens </img> then the text: (L + 258) ${x} d_LLM. Stage ${t.n} trains at an LLM sequence length of ${t.seq}.`,
      llm: `LLM (Qwen-7B, 7.7B parameters): causal attention over (L + 258) ${x} d_LLM.`,
      loss: 'Loss: cross-entropy of each next text token, only where the target is graded text; the image tokens and ungraded prefix text get no loss.',
      backward: `Backward, stage ${t.n}: ${t.n === 1 ? 'LLM frozen; the gradient passes through it into the adapter and the ViT.' : t.n === 2 ? 'nothing frozen; every box is updated.' : 'ViT frozen; the adapter is the last box that learns.'}`,
      update: `Update, stage ${t.n}: AdamW, peak lr ${t.lr}, cosine decay, global batch ${t.batch}, ${t.steps} steps (Appendix C).`,
    }[stage];
    const grid = m.sec ? `${s.tt} slices of ${s.gh} ${x} ${s.gw}` : `${s.gh} ${x} ${s.gw}`;
    return {
      overview: `${who}, input ${m.name}: ${n(s.np)} patches in the ViT, ${tok} tokens for the LLM. Press 1 Input, then Next.`,
      input: m.sec ? `Input: ${m.name}, sampled at 2 frames per second = ${s.frames} frames, ${s.img}, kept at native size.` : `Input: ${m.name}, kept at native size: ${s.img}. Like every image, it is treated as 2 identical frames, for consistency with video.`,
      patch: `Patches: frame pairs ${x} ${m.h / 14} ${x} ${m.w / 14} patches (${grid}) = ${n(s.np)} tubes of 2 ${x} 14 ${x} 14 ${x} 3 = 1,176 numbers, each embedded to d_ViT by the depth-2 3D conv: ${n(s.np)} ${x} d_ViT.`,
      vit: `ViT (675M parameters, same for every LLM size): ${n(s.np)} ${x} d_ViT in and out; 2D-RoPE gives each patch its row and column, so any grid works.`,
      merge: `Merge: each 2 ${x} 2 block of neighbours becomes one token through an MLP: ${n(s.np)} / 4 = ${n(s.vis)} ${x} d_LLM (merged grid ${m.h / 28} ${x} ${m.w / 28}${m.sec ? ` per slice, ${s.tt} slices` : ''}).`,
      splice: `Splice: <|vision_start|> ${n(s.vis)} tokens <|vision_end|> inside the text: (L + ${n(s.seqv)}) ${x} d_LLM. M-RoPE ids for the block span only ${m.sec ? `t 0..${s.tt - 1}, ` : ''}h 0..${m.h / 28 - 1}, w 0..${m.w / 28 - 1} past its start, not ${n(s.vis)} positions.`,
      llm: `LLM (Qwen2: 1.5B, 7.6B or 72B parameters; width d_LLM is not in the paper): causal attention over (L + ${n(s.seqv)}) ${x} d_LLM with M-RoPE.`,
      loss: 'Loss: cross-entropy of each next token at text positions only; the visual tokens are read, never predicted. In stage 3 only the supervised (assistant) parts of the chat are graded.',
      backward: `Backward, stage ${t.n}: ${t.n === 1 ? 'LLM frozen; the gradient passes through it into the merger (trainable as drawn, our reading) and the ViT.' : t.n === 2 ? 'nothing frozen; every box is updated.' : 'ViT (and, as drawn, the merger) frozen; the gradient stops at the merger and only the LLM learns.'}`,
      update: `Update, stage ${t.n}: ${['', 'about 600B tokens', 'about 800B more tokens', 'instruction data in ChatML format'][t.n]}. The paper names only a fused Adam operator (cited to AdamW); no learning rate or batch size.`,
    }[stage];
  }
  const flow = ArchFlow.mount({
    prefix: 'qwen2vlflow', grid: { cols: 7, rows: 2 },
    boxes: BOXES,
    edges: [['img', 'patch'], ['patch', 'vit'], ['vit', 'merge'], ['merge', 'seq'], ['txt', 'emb'], ['emb', 'seq'], ['seq', 'llm'], ['llm', 'loss']],
    stages: [
      { key: 'input', show: ['img', 'txt'] }, { key: 'patch', show: ['patch'] }, { key: 'vit', show: ['vit'] }, { key: 'merge', show: ['merge', 'emb'] },
      { key: 'splice', show: ['seq'] }, { key: 'llm', show: ['llm'] }, { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'photo',
    shape,
    params: id => (T().q2 ? PARAMS : PARAMS_QV)[id] || null,
    frozen: id => !!T().fz[id],
    readout,
  });
  const tBtns = TK.map(k => document.getElementById('btn-qwen2vlflow-' + k));
  // ArchFlow has one variant axis (the input); the training stage is a second axis kept here. Re-pressing the
  // current input button is the public way to make ArchFlow redraw and recompute the readout.
  const redraw = () => { setTitles(); tBtns.forEach((b, i) => b.setAttribute('aria-pressed', String(TK[i] === ts))); document.getElementById('btn-qwen2vlflow-' + flow.state.v).click(); };
  TK.forEach((k, i) => tBtns[i].addEventListener('click', () => { ts = k; redraw(); }));
  document.getElementById('btn-qwen2vlflow-reset').addEventListener('click', () => { ts = 'q2s1'; redraw(); });
  redraw();
})();
</script>
