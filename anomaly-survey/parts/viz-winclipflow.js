<script>
(() => {
  // WinCLIP zero-shot pipeline. Config from WinCLIP v1: 240 px input, ViT-B/16+ (225 tokens, image width 896, text width 640),
  // windows 2x2 and 3x3 patches at stride 1 plus the whole image, 7 + 4 states x 22 templates. Window and prompt totals are
  // our computation (no padding assumed); the joint embedding size d is not stated.
  const x = '×', g = 15, small = (g - 1) ** 2, mid = (g - 2) ** 2, nwin = small + mid + 1;
  const SH = {
    prm: `154 + 88 prompts`, img: `3${x}240${x}240`, win: `${small} + ${mid} + 1 windows`, venc: `ViT-B/16+, width 896`, tenc: `width 640`,
    wemb: `${nwin}${x}d`, proto: `2${x}d`, sm: `${nwin} probabilities`, agg: `240${x}240 map + 1 score` };
  ArchFlow.mount({
    prefix: 'winclipflow', grid: { cols: 6, rows: 2 },
    boxes: {
      prm: { c: 0, r: 1, lane: 'mixed', title: ['prompts', '7+4 states x 22'], glyph: 'text', data: true },
      img: { c: 0, r: 0, lane: 'yellow', title: ['image under', 'inspection'], glyph: 'photo', data: true },
      win: { c: 1, r: 0, lane: 'yellow', title: ['windows'], glyph: 'grid' },
      venc: { c: 2, r: 0, lane: 'yellow', title: ['CLIP image', 'encoder'], glyph: 'layers' },
      tenc: { c: 2, r: 1, lane: 'mixed', title: ['CLIP text', 'encoder'], glyph: 'layers' },
      wemb: { c: 3, r: 0, lane: 'yellow', title: ['window', 'embeddings'], glyph: 'vector' },
      proto: { c: 3, r: 1, lane: 'mixed', title: ['prototypes', 'g_n, g_a'], glyph: 'vector' },
      sm: { c: 4, r: .5, lane: 'red', title: ['two-class', 'softmax'], glyph: 'gate' },
      agg: { c: 5, r: .5, lane: 'red', title: ['harmonic', 'aggregation'], glyph: 'grid' },
    },
    edges: [['img', 'win'], ['win', 'venc'], ['venc', 'wemb'], ['prm', 'tenc'], ['tenc', 'proto'], ['wemb', 'sm'], ['proto', 'sm'], ['sm', 'agg']],
    stages: [
      { key: 'prompts', show: ['prm'] }, { key: 'image', show: ['img'] }, { key: 'windows', show: ['win'] }, { key: 'encoders', show: ['venc', 'tenc'] },
      { key: 'embed', show: ['wemb', 'proto'] }, { key: 'score', show: ['sm'] }, { key: 'map', show: ['agg'] }, { key: 'backward', back: true }],
    variants: {}, variant: null,
    shape: id => SH[id],
    params: id => id === 'venc' || id === 'tenc' ? 'all CLIP weights' : null,
    frozen: id => id === 'venc' || id === 'tenc',
    readout: s => ({
      overview: `Zero-shot WinCLIP: ${nwin} forward passes of the frozen image encoder per image (our count), one pass of the text encoder per product, no trainable weight.`,
      prompts: `Step 1: 7 normal and 4 anomalous states in 22 templates: 154 + 88 prompts (our computation), with the product name filled in.`,
      image: `Step 2: one image, ${SH.img}, after a bicubic resize of the shorter edge to 240.`,
      windows: `Step 3: ${small} windows of 2${x}2 patches (32 px), ${mid} of 3${x}3 (48 px), and the whole image: 15${x}15 grid, stride 16 px (our count, no padding).`,
      encoders: `Step 4: the image encoder (ViT-B/16+, width 896) runs once per window on that window's tokens only; the text encoder (width 640) runs once per prompt.`,
      embed: `Step 5: ${nwin} window class tokens, each a d-vector; two prototypes, each the mean of its class's prompt embeddings.`,
      score: `Step 6: cosine similarity to g_n and g_a, divided by the temperature, softmax over the two: ${nwin} probabilities of "anomalous".`,
      map: `Step 7: each pixel takes the harmonic mean of the windows covering it; the three scale maps combine by another harmonic mean. Image score: the whole-image probability.`,
      backward: 'Backward: no loss, no gradient. Both encoders are brown (frozen) and nothing upstream of them is trainable, so no arrow lights up.',
    })[s],
  });
})();
</script>
