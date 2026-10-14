// Check the token rule against every worked example the papers print.
const q2 = (w, h, frames = 2) => { const T = frames / 2, p = (w / 14) * (h / 14) * T; return { patches: p, tokens: p / 4 }; };
const q3 = (w, h) => (w / 32) * (h / 32);
const ok = (name, got, want) => console.log((got === want ? 'OK  ' : 'FAIL'), name, got, want === undefined ? '' : 'paper ' + want);
ok('Qwen2-VL 224x224 +2 markers', q2(224, 224).tokens + 2, 66);
ok('Qwen2-VL fig page 1092x8204', q2(1092, 8204).tokens, 11427);
ok('Qwen2-VL fig strip 224x28', q2(224, 28).tokens, 8);
ok('Qwen2-VL fig wide 1260x700', q2(1260, 700).tokens, 1125);
// Expected mismatch: at the stated 2 fps the figure's clip gives 4416; the figure prints 2208 (= 16 frames, or a reduced frame size).
console.log('EXPECTED MISMATCH Qwen2-VL fig video 644x336 16s at 2 fps', q2(644, 336, 32).tokens, 'paper 2208');
ok('  same video if 16 frames', q2(644, 336, 16).tokens, 2208);
console.log('  patches: page', q2(1092, 8204).patches, 'wide', q2(1260, 700).patches, 'strip', q2(224, 28).patches, 'photo', q2(224, 224).patches, 'video', q2(644, 336, 32).patches);
console.log('  grids (merged h x w): page', 8204 / 28, 1092 / 28, 'wide', 700 / 28, 1260 / 28, 'video', 336 / 28, 644 / 28);
ok('Qwen3-VL fig page 1248x9376 @32px', q3(1248, 9376), 11427);
ok('Qwen3-VL fig strip 256x32', q3(256, 32), 8);
ok('Qwen3-VL fig 1440x800', q3(1440, 800), 1125);
console.log('  Qwen3 video 736x448 per temporal patch', q3(736, 448), 'patches', (736 / 16) * (448 / 16));
console.log('  Qwen-VL 448: patches', (448 / 14) ** 2, ' 224:', (224 / 14) ** 2);
