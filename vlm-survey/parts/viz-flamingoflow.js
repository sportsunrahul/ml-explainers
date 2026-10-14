<script>
(() => {
  const x = '×', L = 256, NI = 5, R = 64, RW = 1536;
  // Flamingo v2 Table 4 (App. A.1.4): frozen LM and GATED XATTN-DENSE L/D/H; Table 5 / App. B.1.1: insertion stride and parameter counts.
  const V = {
    f3b: { name: 'Flamingo-3B', lmL: 24, D: 2048, H: 16, xL: 24, every: 1, lmP: '1.4B', xP: '1.2B' },
    f9b: { name: 'Flamingo-9B', lmL: 40, D: 4096, H: 32, xL: 10, every: 4, lmP: '7.1B', xP: '1.6B' },
    f80b: { name: 'Flamingo-80B', lmL: 80, D: 8192, H: 64, xL: 12, every: 7, lmP: '70B', xP: '10B' },
  };
  const SH = {
    img: () => `N${x}T${x}3${x}320${x}320`, nfnet: () => `N${x}T${x}S${x}d_v`, resampler: () => `N${x}${R}${x}${RW}`,
    text: () => `${L} token ids`, embed: m => `${L}${x}${m.D}`, xattn: m => `${L}${x}${m.D}`, lm: m => `${L}${x}${m.D}`, head: () => `${L}${x}V`, loss: () => 'one number' };
  const SUB = {
    xattn: m => `${L}${x}${m.D}|${m.xL} layers`, lm: m => `${L}${x}${m.D}|${m.lmL} blocks` };
  const PARAMS = { resampler: 'latents,|6 layers', xattn: 'xattn, FFW,|2 alphas each' };
  const FROZEN = { nfnet: 1, embed: 1, lm: 1, head: 1 };
  const where = m => { const a = []; for (let i = 0; i < m.lmL; i += m.every) a.push(i + 1); return a; };
  ArchFlow.mount({
    prefix: 'flamingoflow', grid: { cols: 7, rows: 2 },
    boxes: {
      img: { c: 0, r: 0, lane: 'blue', title: ['images,', 'video frames'], glyph: 'photos', data: true },
      nfnet: { c: 1, r: 0, lane: 'blue', title: ['NFNet-F6'], glyph: 'layers' },
      resampler: { c: 2, r: 0, lane: 'green', title: ['Perceiver', 'Resampler'], glyph: 'latents' },
      text: { c: 0, r: 1, lane: 'yellow', title: ['interleaved', 'text'], glyph: 'text', data: true },
      embed: { c: 1, r: 1, lane: 'yellow', title: ['LM token', 'embeddings'], glyph: 'tokens' },
      xattn: { c: 3, r: .5, lane: 'green', title: ['gated', 'xattn-dense'], glyph: 'gate' },
      lm: { c: 4, r: .5, lane: 'yellow', title: ['LM blocks'], glyph: 'layers' },
      head: { c: 5, r: .5, lane: 'yellow', title: ['LM head'], glyph: 'vector' },
      loss: { c: 6, r: .5, lane: 'red', title: ['loss'], glyph: 'loss' },
    },
    edges: [['img', 'nfnet'], ['nfnet', 'resampler'], ['resampler', 'xattn'], ['text', 'embed'], ['embed', 'xattn'], ['xattn', 'lm'], ['lm', 'head'], ['head', 'loss']],
    stages: [
      { key: 'inputs', show: ['img', 'text'] }, { key: 'vision', show: ['nfnet'] }, { key: 'resample', show: ['resampler'] }, { key: 'tokens', show: ['embed'] },
      { key: 'xattn', show: ['xattn'] }, { key: 'lm', show: ['lm'] }, { key: 'loss', show: ['head', 'loss'] },
      { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'f80b',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => SUB[id] ? SUB[id](m) : null,
    params: id => PARAMS[id] || null,
    frozen: id => !!FROZEN[id],
    readout: (s, m) => {
      const w = where(m), ws = w.length > 6 ? w.slice(0, 4).join(', ') + ', ..., ' + w.at(-1) : w.join(', ');
      return {
        overview: `${m.name}: a ${m.lmP} frozen language model with ${m.xL} new gated xattn-dense layers (${m.xP} parameters) and a resampler that turns each image into ${R} tokens of width ${RW}. The drawing folds the interleaved stack into one green box and one brown box.`,
        inputs: `Step 1: one web-page example is ${L} text tokens with up to N = ${NI} images (${SH.img()}; T = 1 for an image, 8 frames for a training video). Images are resized to 320 ${x} 320. Batch size B is left out of the shapes.`,
        vision: `Step 2: the frozen NFNet-F6 encodes every frame separately. Its final-stage grid has S positions of width d_v per frame; the paper states neither number, so both stay symbolic: ${SH.nfnet()}.`,
        resample: `Step 3: add a learned time embedding per frame, flatten to T${x}S features per image, and let ${R} learned latents (width ${RW}, 6 layers, 16 heads of width 96) cross-attend to them: ${SH.resampler()}, i.e. ${NI * R} visual tokens for ${NI} images, whatever S and T are.`,
        tokens: `Step 4: the text becomes ${L} ids (with <image> and <EOC> markers) and the frozen LM's embedding table maps them to ${SH.embed(m)}.`,
        xattn: `Step 5: each gated xattn-dense layer: queries ${L}${x}${m.D} from the text, keys and values from the visual tokens, ${m.H} heads of width ${m.D / m.H}. Per head the logits are ${L} ${x} ${NI * R} for ${NI} images, but the mask keeps only the ${R} columns of each token's last preceding image (in training, for half of the web pages, the next image instead). FFW hidden width ${4 * m.D}. Output ${L}${x}${m.D}, times tanh(alpha).`,
        lm: `Step 6: the frozen LM has ${m.lmL} blocks (width ${m.D}, ${m.H} heads). A gated layer sits in front of block${w.length > 1 ? 's' : ''} ${ws}: ${m.every === 1 ? 'every block' : 'every ' + m.every + 'th block'}, ${m.xL} in all. Shape stays ${L}${x}${m.D}.`,
        loss: `Step 7: the frozen head gives next-token logits at every text position (${L}${x}V, V = vocabulary size); cross-entropy against the next text token, summed over the sequence.`,
        backward: `Backward: the gradient crosses the frozen LM blocks (brown, no update) to reach the ${m.xL} gated layers and, through their keys and values, the resampler. It stops at the NFNet: nothing below it is trainable.`,
        update: `Update: AdamW on the resampler and the ${m.xL} gated layers only (about ${m.xP} + 194M parameters, + the <EOC> row); learning rate warmed up from 0 to 1e-4 over 5000 steps, then constant, for 500k steps (gradient steps or updates: not stated).`,
      }[s];
    },
  });
})();
</script>
