<script>
(() => {
  const x = 'x';
  // Sec. 4.1 and Table 8. The training image size is not stated; token counts at 800 x 1333 are our reading.
  const V = {
    t: { name: 'Grounding DINO T', bb: 'Swin-T', strides: [8, 16, 32, 64], strideTxt: '8, 16, 32 + a 64 copy', gpus: '16 V100', batch: 32, perGpu: '2 images', params: '172M' },
    l: { name: 'Grounding DINO L', bb: 'Swin-L', strides: [4, 8, 16, 32], strideTxt: '4, 8, 16, 32', gpus: '64 A100', batch: 64, perGpu: '1 image', params: '341M' },
  };
  const tokensAt = (m, H, W) => m.strides.reduce((s, st) => s + Math.ceil(H / st) * Math.ceil(W / st), 0);
  const frac = m => m.strides.reduce((s, st) => s + 1 / (st * st), 0);
  const fmt = n => n.toLocaleString('en-US');
  const FWD = {
    img: () => `B ${x} 3 ${x} H ${x} W|(size not stated)`, prompt: () => 'name . name . ...|max 256 tokens',
    swin: m => `${m.bb}: 4 maps|strides ${m.strides[0]} to ${m.strides[3]}`, bert: () => `BERT-base|B ${x} N_T ${x} 256`,
    enh: () => `B ${x} N_I ${x} 256|B ${x} N_T ${x} 256`, qsel: () => `scores N_I ${x} N_T|keep the top 900`,
    dec: () => `B ${x} 900 ${x} 256|6 layers`, head: () => `900 boxes|900 ${x} N_T logits`, loss: () => 'focal + 5 L1|+ 2 GIoU',
  };
  const PARAMS = { swin: 'all weights|lr 1e-5', bert: 'all weights|lr 1e-5', enh: 'all 6 layers|lr 1e-4', dec: 'all 6 layers|lr 1e-4', head: 'box head|lr 1e-4' };
  ArchFlow.mount({
    prefix: 'gdinoflow', grid: { cols: 7, rows: 2 },
    boxes: {
      prompt: { c: 0, r: 0, lane: 'yellow', title: ['prompt'], glyph: 'text', data: true },
      img: { c: 0, r: 1, lane: 'blue', title: ['image'], glyph: 'photo', data: true },
      bert: { c: 1, r: 0, lane: 'yellow', title: ['text', 'backbone'], glyph: 'tokens' },
      swin: { c: 1, r: 1, lane: 'blue', title: ['image', 'backbone'], glyph: 'layers' },
      enh: { c: 2, r: .5, lane: 'mixed', title: ['feature', 'enhancer'], glyph: 'layers' },
      qsel: { c: 3, r: 1, lane: 'green', title: ['query', 'selection'], glyph: 'grid', cls: false },
      dec: { c: 4, r: .5, lane: 'mixed', title: ['cross-modality', 'decoder'], glyph: 'layers' },
      head: { c: 5, r: .5, lane: 'green', title: ['box head,', 'dot with text'], glyph: 'matrix' },
      loss: { c: 6, r: .5, lane: 'red', title: ['match', 'and loss'], glyph: 'loss' },
    },
    edges: [['prompt', 'bert'], ['img', 'swin'], ['bert', 'enh'], ['swin', 'enh'], ['enh', 'qsel'], ['enh', 'dec'], ['qsel', 'dec'], ['dec', 'head'], ['head', 'loss']],
    stages: [
      { key: 'inputs', show: ['img', 'prompt'] }, { key: 'backbones', show: ['swin', 'bert'] }, { key: 'enhancer', show: ['enh'] }, { key: 'select', show: ['qsel'] },
      { key: 'decoder', show: ['dec'] }, { key: 'heads', show: ['head'] }, { key: 'loss', show: ['loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 't',
    sub: (id, m) => FWD[id](m),
    // shape() is what a box without weights shows in the backward stages (sub() is the forward label)
    shape: (id, m) => id === 'qsel' ? 'no weights:|a top-k pick' : id === 'loss' ? 'the gradient|starts here' : FWD[id](m),
    params: id => PARAMS[id] || null,
    frozen: () => false,
    readout: (s, m) => {
      const t800 = tokensAt(m, 800, 1333);
      return {
        overview: `${m.name}: ${m.bb} image backbone, BERT-base text backbone, 6 enhancer and 6 decoder layers at width 256, 900 queries; ${m.params} parameters.`,
        inputs: `Step 1: B images, B ${x} 3 ${x} H ${x} W (H and W not stated in the paper), and B prompts of N_T tokens each, N_T at most 256.`,
        backbones: `Step 2: ${m.bb} feature maps at strides ${m.strideTxt}, each B ${x} 256 ${x} H/s ${x} W/s after projection. BERT-base gives B ${x} N_T ${x} 256 after projection.`,
        enhancer: `Step 3: N_I = HW ${x} ${frac(m).toFixed(4)} image tokens (the four maps flattened). At an assumed 800 ${x} 1333 input (our reading) that is ${fmt(t800)}; the paper says only "more than 10,000". Image B ${x} N_I ${x} 256, text B ${x} N_T ${x} 256, 6 layers.`,
        select: `Step 4: scores B ${x} N_I ${x} N_T (at 800 ${x} 1333 and 256 text tokens: ${fmt(t800 * 256)} per image), max over text -> B ${x} N_I, top 900 -> B ${x} 900 indices -> anchors B ${x} 900 ${x} 4; content queries 900 ${x} 256, learned.`,
        decoder: `Step 5: queries B ${x} 900 ${x} 256 through 6 layers: self-attention (900 ${x} 900), deformable image cross-attention, text cross-attention (900 ${x} N_T per head), FFN.`,
        heads: `Step 6: per layer, boxes B ${x} 900 ${x} 4 and logits B ${x} 900 ${x} N_T (query . text token). No class matrix: the logits have one column per token of this prompt.`,
        loss: 'Step 7: Hungarian matching per image and per layer, then focal loss on all query-token logits plus 5.0 L1 and 2.0 GIoU on matched boxes, for 6 decoder layers and the encoder outputs.',
        backward: `Backward: every box with weights is red, including ${m.bb} and BERT. Query selection has no weights; the gradient reaches the enhancer through the decoder and through the auxiliary loss on the encoder outputs.`,
        update: `Update: AdamW, lr 1e-4, backbones 1e-5, weight decay 0.0001, clip 0.1. ${m.name}: ${m.gpus} GPUs, total batch ${m.batch} (${m.perGpu} per GPU, our reading). Epochs: not stated.`,
      }[s];
    },
  });
})();
</script>
