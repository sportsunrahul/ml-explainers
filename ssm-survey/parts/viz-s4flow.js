<script>
(() => {
  const x = '\u00d7';
  // Path-X configs: original results (S4 Table 11) and the updated row (S4D Table 12). N from S4 App D.2 / S4D Table 12.
  const V = {
    upd: { name: 'updated row (S4D Table 12)', B: 32, L: 16384, H: 256, N: 64, depth: 6, bi: true, glu: true, ep: 50, wd: .05, sched: 'cosine' },
    orig: { name: 'original row (S4 Table 11)', B: 32, L: 16384, H: 256, N: 64, depth: 6, bi: false, glu: false, ep: 100, wd: 0, sched: 'constant, decay on plateau' },
  };
  const f = n => n.toLocaleString('en-US');
  const act = m => `${m.B}${x}${f(m.L)}${x}${m.H}`;
  const kern = m => `${m.bi ? '2' + x : ''}${m.H}${x}${f(m.L)}`;
  const SH = {
    inp: m => `${m.B}${x}${f(m.L)}${x}1`, enc: act, ssmp: m => `${m.H}${x}(${5 * m.N} complex + \u0394)`, kern, conv: act, skip: act,
    act: act, mix: act, pool: m => `${m.B}${x}${m.H} -> ${m.B}${x}2`, loss: () => 'one number' };
  ArchFlow.mount({
    prefix: 's4flow', grid: { cols: 8, rows: 2 },
    boxes: {
      inp: { c: 0, r: 1, lane: 'blue', title: ['pixels as', 'a sequence'], glyph: 'seq', pattern: 'bbbbbbbb', data: true },
      enc: { c: 1, r: 1, lane: 'blue', title: ['input map', 'to H channels'], glyph: 'vector' },
      ssmp: { c: 1, r: 0, lane: 'yellow', title: ['SSM parameters', 'per channel'], glyph: 'matrix' },
      kern: { c: 2, r: 0, lane: 'yellow', title: ['kernel', 'generation'], glyph: 'vector' },
      conv: { c: 2, r: 1, lane: 'yellow', title: ['FFT', 'convolution'], glyph: 'layers' },
      skip: { c: 3, r: 1, lane: 'yellow', title: ['skip', '+ D u'], glyph: 'unit' },
      act: { c: 4, r: 1, lane: 'green', title: ['nonlinearity'], glyph: 'gate' },
      mix: { c: 5, r: 1, lane: 'green', title: ['output', 'mixing'], glyph: 'mlp' },
      pool: { c: 6, r: 1, lane: 'green', title: [`${'\u00d7'}6 layers, then`, 'mean pool + head'], glyph: 'layers' },
      loss: { c: 7, r: 1, lane: 'red', title: ['classification', 'loss'], glyph: 'loss' },
    },
    edges: [['inp', 'enc'], ['enc', 'conv'], ['ssmp', 'kern'], ['kern', 'conv'], ['conv', 'skip'], ['skip', 'act'], ['act', 'mix'], ['mix', 'pool'], ['pool', 'loss']],
    stages: [
      { key: 'input', show: ['inp', 'enc'] }, { key: 'kernel', show: ['ssmp', 'kern'] }, { key: 'conv', show: ['conv'] }, { key: 'skip', show: ['skip'] },
      { key: 'act', show: ['act', 'mix'] }, { key: 'loss', show: ['pool', 'loss'] }, { key: 'backward', back: true }, { key: 'update', back: true, done: true }],
    variants: V, variant: 'upd',
    shape: (id, m) => SH[id](m),
    sub: (id, m) => id === 'act' ? (m.glu ? 'GLU gate \u03c3(W2 y)' : 'not named in S4 v3') : id === 'mix' ? (m.glu ? 'W1 y, times the gate' : `W, ${m.H}${x}${m.H}`) : id === 'kern' ? `${kern(m)}${m.bi ? '|two directions' : ''}` : null,
    params: (id, m) => ({ enc: 'input map (our reading)', ssmp: '\u039b, P, Q (= A), B, C, \u0394', skip: 'D', act: m.glu ? 'W2' : null, mix: m.glu ? 'W1' : 'W', pool: 'head weights' })[id] || null,
    frozen: () => false,
    readout: (s, m) => ({
      overview: `Path-X, ${m.name}: ${m.B} images of 128 ${x} 128 = ${f(m.L)} pixels, read as sequences. ${m.depth} layers, each with H = ${m.H} channels; every channel has its own SSM with state size N = ${m.N}.`,
      input: `Step 1: ${SH.inp(m)} pixel values, mapped position by position to ${m.H} channels: ${act(m)}.`,
      kernel: `Step 2: each of the ${m.H} channels turns its parameters into a kernel of length ${f(m.L)}${m.bi ? ', one per direction' : ''}: ${kern(m)} = ${f((m.bi ? 2 : 1) * m.H * m.L)} numbers per layer, rebuilt every training step.`,
      conv: `Step 3: per channel, pad to ${f(2 * m.L)}, FFT the input and the kernel, multiply, inverse FFT: ${act(m)}. About L log L work per channel, not L\u00b2.`,
      skip: `Step 4: add D times the channel's own input (one number D per channel): ${act(m)}.`,
      act: m.glu ? `Step 5: GLU, (W1 y) \u2218 \u03c3(W2 y) with W1, W2 of ${m.H} ${x} ${m.H}: the gate is the nonlinearity and W1 mixes the channels. ${f(2 * m.H * m.H)} weights.` : `Step 5: a pointwise nonlinearity (S4 v3 does not name it), then the position-wise linear map W of ${m.H} ${x} ${m.H} that mixes the channels: ${f(m.H * m.H)} weights.`,
      loss: `Step 6: after ${m.depth} such layers (batch norm before each; residual connections are our reading), average over all ${f(m.L)} positions, ${m.B}${x}${m.H}, and map to two scores: connected or not.`,
      backward: 'Backward: the gradient leaves the loss, crosses every layer\'s mixing, gate and skip, goes through the FFT convolution into the kernel, and through the kernel algorithm into the SSM parameters. Nothing in S4 is frozen.',
      update: `Update: AdamW, learning rate 0.0005, batch ${m.B}, ${m.ep} epochs, weight decay ${m.wd}, schedule ${m.sched}. The learning rate on the SSM parameters is capped at 0.001.`,
    })[s],
  });
})();
</script>
