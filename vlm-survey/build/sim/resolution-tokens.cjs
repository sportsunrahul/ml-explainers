// Token counts per preset x strategy, the same formulas viz-resolution.js uses.
const presets = { doc: [992, 1404], wide: [1920, 1080], photo: [400, 300] };
const grids = []; for (let c = 1; c <= 6; c++) for (let r = 1; r <= 6; r++) if (c * r <= 6 && c * 224 <= 672 && r * 224 <= 672 && !(c * 224 > 448 && r * 224 > 448)) grids.push([c, r]);
function fixed(S, W, H) { return { w: S, h: S, vit: (S / 14) ** 2, llm: S === 448 ? 256 : (S / 14) ** 2, sx: S / W, sy: S / H }; }
function tile(W, H) {
  let best = null;
  for (const [c, r] of grids) { const s = Math.min(c * 224 / W, r * 224 / H), n = c * r;
    const key = [Math.min(s, 1), -n];
    if (!best || key[0] > best.key[0] + 1e-9 || (Math.abs(key[0] - best.key[0]) < 1e-9 && key[1] > best.key[1])) best = { c, r, s, key }; }
  const w = Math.round(W * best.s), h = Math.round(H * best.s), cols = Math.ceil(w / 14), rows = Math.ceil(h / 14);
  return { grid: best.c + 'x' + best.r, w, h, vit: best.c * best.r * 256 + 256, llm: cols * rows + rows + 256, sx: best.s, sy: best.s };
}
function native(W, H) { const w = Math.max(28, Math.round(W / 28) * 28), h = Math.max(28, Math.round(H / 28) * 28);
  return { w, h, vit: (w / 14) * (h / 14), llm: (w / 28) * (h / 28) + 2, sx: w / W, sy: h / H }; }
console.log('grids', grids.map(g => g.join('x')).join(' '));
for (const [k, [W, H]] of Object.entries(presets)) {
  console.log(k, W, H);
  for (const [n, f] of [['fixed224', fixed(224, W, H)], ['fixed448', fixed(448, W, H)], ['tile', tile(W, H)], ['native', native(W, H)]])
    console.log('  ', n.padEnd(9), JSON.stringify(f), 'min scale', Math.min(f.sx, f.sy).toFixed(3));
}
console.log('224 native', native(224, 224));
