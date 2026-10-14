/* CoreRecall: K key-value pairs written into an N x N state by outer products (linear attention's write rule, no decay),
   against a cache that keeps every pair. Each stored key is then used as a query; the state's answer counts as right when
   the stored value nearest to it is the one written with that key. Exact arithmetic on one fixed random draw per (N, keys).
   Shared by the long page (viz-recall) and atlas stage 10. */
const CoreRecall = (() => {
  const NS = [4, 8, 16, 32], KMAX = 256;
  const GRID = [];
  for (let k = 1; k <= KMAX; k += k < 16 ? 1 : k < 32 ? 2 : k < 64 ? 4 : k < 128 ? 8 : 16) GRID.push(k);
  if (GRID[GRID.length - 1] !== KMAX) GRID.push(KMAX);
  const draws = {}, memo = {};

  function unit(N, g) {
    const v = new Float64Array(N); let s = 0;
    for (let i = 0; i < N; i++) { v[i] = g(); s += v[i] * v[i]; }
    s = Math.sqrt(s); for (let i = 0; i < N; i++) v[i] /= s; return v;
  }
  // "ortho": keys come in blocks of N mutually orthogonal vectors (Gram-Schmidt), so the first N keys never interfere.
  // An N-dimensional space has room for only N such directions, so key N + 1 must overlap some earlier key.
  function pairsOf(N, ortho) {
    const id = N + (ortho ? 'o' : 'r');
    if (draws[id]) return draws[id];
    const g = PE.gauss(PE.rng(1000 + N + (ortho ? 50000 : 0))), ks = [], vs = [];
    while (ks.length < KMAX) {
      if (!ortho) { ks.push(unit(N, g)); continue; }
      const B = [];
      while (B.length < N) {
        const v = unit(N, g);
        for (const b of B) { let d = 0; for (let i = 0; i < N; i++) d += v[i] * b[i]; for (let i = 0; i < N; i++) v[i] -= d * b[i]; }
        let s = 0; for (let i = 0; i < N; i++) s += v[i] * v[i]; s = Math.sqrt(s);
        if (s < 1e-6) continue; for (let i = 0; i < N; i++) v[i] /= s; B.push(v);
      }
      ks.push(...B);
    }
    ks.length = KMAX;
    for (let i = 0; i < KMAX; i++) vs.push(unit(N, g));
    return (draws[id] = { ks, vs });
  }
  // h = sum_i v_i k_i^T over the first K pairs (row a = value index, column b = key index).
  function stateOf(N, K, ortho) {
    const { ks, vs } = pairsOf(N, ortho), h = new Float64Array(N * N);
    for (let i = 0; i < K; i++) for (let a = 0; a < N; a++) { const va = vs[i][a]; for (let b = 0; b < N; b++) h[a * N + b] += va * ks[i][b]; }
    return h;
  }
  // How many of the K stored keys get their own value back: from the state, and from the cache.
  function evaluate(N, K, ortho) {
    const id = N + (ortho ? 'o' : 'r') + K;
    if (memo[id]) return memo[id];
    const { ks, vs } = pairsOf(N, ortho), h = stateOf(N, K, ortho), y = new Float64Array(N);
    let okState = 0, okCache = 0;
    for (let j = 0; j < K; j++) {
      const q = ks[j];
      for (let a = 0; a < N; a++) { let s = 0; for (let b = 0; b < N; b++) s += h[a * N + b] * q[b]; y[a] = s; }
      let best = -1, bv = -Infinity, bk = -1, bkv = -Infinity;
      for (let i = 0; i < K; i++) {
        let d = 0, e = 0; for (let a = 0; a < N; a++) { d += vs[i][a] * y[a]; e += ks[i][a] * q[a]; }
        if (d > bv) { bv = d; best = i; }
        if (e > bkv) { bkv = e; bk = i; }
      }
      if (best === j) okState++;
      if (bk === j) okCache++;
    }
    return (memo[id] = { N, K, okState, okCache, acc: okState / K, accCache: okCache / K, h });
  }
  const curve = (N, ortho) => GRID.map(K => [K, evaluate(N, K, ortho).acc]);

  const fromSlider = s => Math.max(1, Math.min(KMAX, Math.round(Math.pow(KMAX, s / 1000))));
  const toSlider = K => Math.round(1000 * Math.log(Math.max(1, K)) / Math.log(KMAX));
  const fmt = n => Math.round(n).toLocaleString('en-US');
  const pct = x => Math.round(100 * x) + '%';

  function readout(S) {
    const K = Math.round(S.K), N = S.N, r = evaluate(N, K, S.ortho);
    return `K = ${K} pair${K > 1 ? 's' : ''}, N = ${N}, ${S.ortho ? 'keys as orthogonal as N dimensions allow' : 'random keys'}. ` +
      `The state (${N} x ${N} = ${fmt(N * N)} numbers, the same at any K) returns the right value for ${r.okState} of ${K} keys (${pct(r.acc)}). ` +
      `The cache (${K} keys + ${K} values = ${fmt(2 * N * K)} numbers, growing with K) returns ${r.okCache} of ${K}. ` +
      `One lookup costs ${fmt(N * N)} multiply-adds from the state at any K, and ${fmt(2 * N * K)} from the cache.`;
  }

  // draw(ctx, W, H, S, glow): S = { N, K (may be fractional while tweening), ortho }; glow(word) in [0, 1].
  function draw(ctx, W, H, S, glow = () => 0) {
    const C = n => PE.css(n), narrow = W < 520, N = S.N, K = Math.max(1, Math.round(S.K)), r = evaluate(N, K, S.ortho);
    const Y = C('--yellow'), Br = C('--brown'), Bl = C('--blue'), G = C('--green'), R = C('--red'), M = C('--muted'), F = C('--fg'), BG = C('--bg');
    ctx.clearRect(0, 0, W, H);
    // Memory panel: the state register (fixed) and the cache (one column per pair, wrapping into rows of 64).
    // Narrow layout starts a line lower so the corner badge stays clear.
    const mp = narrow ? { x: 14, y: 50, w: W - 28, h: 128 } : { x: 18, y: 34, w: Math.min(300, W * .32), h: H - 80 };
    const side = narrow ? 84 : Math.min(140, mp.w * .55), cell = side / N;
    const rx = mp.x, ry = mp.y + 6;
    let mx = 0; for (let i = 0; i < N * N; i++) mx = Math.max(mx, Math.abs(r.h[i]));
    const gs = glow('state'), gw = glow('write'), gr = glow('read');
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
      ctx.fillStyle = PE.mix(BG, Y, PE.clamp(.12 + .88 * Math.abs(r.h[a * N + b]) / (mx || 1), 0, 1));
      ctx.fillRect(rx + b * cell, ry + a * cell, cell - (cell > 4 ? 1 : 0), cell - (cell > 4 ? 1 : 0));
    }
    ctx.save(); ctx.strokeStyle = Y; ctx.lineWidth = 1.5 + 3 * gs; ctx.strokeRect(rx - 1.5, ry - 1.5, side + 2, side + 2); ctx.restore();
    PE.label(ctx, `state h: ${N} x ${N}`, rx, ry - 9, Y, 12);
    if (narrow) PE.label(ctx, `${fmt(N * N)} numbers, fixed`, rx, ry + side + 16, Y, 11);
    else PE.label(ctx, `${fmt(N * N)} numbers, fixed`, rx + side + 10, ry + 14, Y, 11);
    if (gw > .02) {
      const ay = narrow ? ry + side * .3 : ry + side * .55;
      PE.arrow(ctx, rx + side + 40, ay, rx + side + 6, ay, Bl, 2.5, 9, gw);
      if (!narrow) PE.label(ctx, 'write v k^T', rx + side + 44, ay + 4, Bl, 11);
    }
    if (gr > .02 && !narrow) { PE.arrow(ctx, rx + side * .5, ry + side + 4, rx + side * .5, ry + side + 26, G, 2.5, 9, gr); PE.label(ctx, 'read h k', rx + side * .5 + 8, ry + side + 22, G, 11); }
    if (gr > .02 && narrow) PE.arrow(ctx, rx + side + 4, ry + side * .75, rx + side + 26, ry + side * .75, G, 2.5, 9, gr);
    const cx0 = narrow ? rx + side + 34 : rx, cy = narrow ? ry : ry + side + 50, cw = narrow ? W - cx0 - 14 : mp.w;
    const PER = 64, colW = cw / PER, rh = narrow ? 18 : 24, rows = Math.ceil(K / PER), gc = glow('cache');
    ctx.globalAlpha = .92;
    for (let i = 0; i < K; i++) {
      const x = cx0 + (i % PER) * colW, y = cy + Math.floor(i / PER) * (rh + 4), wcol = Math.max(.7, colW - (colW > 3 ? 1 : .4));
      ctx.fillStyle = Br; ctx.fillRect(x, y, wcol, rh / 2 - 1);
      ctx.fillStyle = PE.mix(Br, BG, .4); ctx.fillRect(x, y + rh / 2, wcol, rh / 2 - 1);
    }
    ctx.globalAlpha = 1;
    if (gc > .02) { ctx.save(); ctx.strokeStyle = Br; ctx.lineWidth = 2 + 3 * gc; ctx.globalAlpha = gc; ctx.strokeRect(cx0 - 3, cy - 3, (K < PER ? K : PER) * colW + 6, rows * (rh + 4) + 2); ctx.restore(); }
    PE.label(ctx, `cache: ${K} key${K > 1 ? 's' : ''} + ${K} value${K > 1 ? 's' : ''}`, cx0, cy - 9, Br, narrow ? 11 : 12);
    PE.label(ctx, `${fmt(2 * N * K)} numbers, grows with K`, cx0, cy + rows * (rh + 4) + 12, Br, 11);
    // Chart: share of keys answered correctly against K (log axis).
    const b = narrow ? { x: 46, y: mp.y + mp.h + 34, w: W - 62, h: H - mp.y - mp.h - 82 } : { x: mp.x + mp.w + 66, y: 40, w: W - (mp.x + mp.w + 66) - 24, h: H - 86 };
    const ax = PE.axes(ctx, b, [1, KMAX], [0, 1], { logx: true, xt: [1, 4, 16, 64, 256], yt: [0, .25, .5, .75, 1], yf: v => Math.round(v * 100) + '%',
      xl: 'K, pairs stored (log scale)', yl: narrow ? '' : 'keys answered right', ylo: 44 });
    const X = ax.X, YY = ax.Y;
    // K = N: the most keys that can be mutually orthogonal in N dimensions.
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = M; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(N), b.y); ctx.lineTo(X(N), b.y + b.h); ctx.stroke(); ctx.restore();
    PE.label(ctx, `K = N = ${N}`, X(N) - 5, b.y + b.h - 22, M, 11, 'right');
    // Chance: guessing one of K values.
    const chance = []; for (let k = 1; k <= KMAX; k *= 1.25) chance.push([X(k), YY(1 / k)]);
    chance.push([X(KMAX), YY(1 / KMAX)]);
    PE.polyline(ctx, chance, M, 1.2, [2, 4], .8);
    PE.label(ctx, 'chance, 1/K', X(90), YY(1 / 90) - 6, M, 10, 'center');
    const cur = curve(N, S.ortho), sig = glow('signal'), fade = S.fade == null ? 1 : S.fade;
    // While the key type switches, the old curve fades out under the new one.
    if (fade < 1) PE.polyline(ctx, curve(N, !S.ortho).map(([k, a]) => [X(k), YY(a)]), Y, 2.2, [5, 4], 1 - fade);
    PE.polyline(ctx, cur.map(([k, a]) => [X(k), YY(a)]), Y, 2.6 + 3 * sig, null, .25 + .75 * fade);
    PE.polyline(ctx, [[X(1), YY(1)], [X(KMAX), YY(1)]], Br, 2 + 3 * gc);
    PE.label(ctx, 'cache: every key answered', X(KMAX) - 4, YY(1) - 8, Br, 11, 'right');
    PE.label(ctx, 'state', X(2) + 2, YY(cur[1][1]) + 18, Y, 12);
    // Current K: dots, and the red error gap.
    const x = X(K), ys = YY(r.acc), ct = glow('crosstalk');
    ctx.save(); ctx.strokeStyle = F; ctx.globalAlpha = .3; ctx.beginPath(); ctx.moveTo(x, b.y); ctx.lineTo(x, b.y + b.h); ctx.stroke(); ctx.restore();
    if (r.acc < .97) {
      ctx.save(); ctx.fillStyle = R; ctx.globalAlpha = .1 + .25 * ct; ctx.fillRect(x - 5, YY(1), 10, ys - YY(1)); ctx.restore();
      PE.arrow(ctx, x + 9, YY(1) + 4, x + 9, ys - 4, R, 2 + 2 * ct, 8, .9);
      // Top right, under the cache label: the state's curve is always low there once it has started to fall.
      PE.label(ctx, `${Math.round(100 * (1 - r.acc))}% lost to crosstalk`, b.x + b.w - 4, YY(1) + 20, R, 11, 'right');
    }
    PE.dot(ctx, x, YY(r.accCache), 6, Br);
    PE.dot(ctx, x, ys, 6.5, Y);
  }
  return { NS, KMAX, GRID, draw, readout, evaluate, curve, stateOf, fromSlider, toSlider };
})();
