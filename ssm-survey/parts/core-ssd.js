/* CoreSsd: the T x T matrix M behind three causal sequence layers, built from the same random toy inputs.
   softmax attention (row-normalized exp scores), linear attention (phi(C) phi(B)^T masked, phi = elu + 1, no decay,
   numerator only) and SSD (the same scores times the decay mask L, Mamba-2 Sec 5.1). Dictionary (Mamba-2 Fig 4):
   C plays the query, B the key, x the value. Also the three ways to compute y = M x for SSD (quadratic, recurrent,
   chunked as in Mamba-2 Listing 1), work counts, and pe3 items shared by viz-ssd and atlas stage 8. */
const CoreSsd = (() => {
  const N = 2, P = 2;                 // toy state size and head dimension
  const HS = 1.25, HD = .62, XS = .3; // world heights: matrix columns, decay line, input/output lollipops
  const phi = v => v > 0 ? v + 1 : Math.exp(v);
  const dotv = (a, b) => a.reduce((s, v, k) => s + v * b[k], 0);
  const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
  const matvec = (M, X) => M.map(row => { const y = new Array(P).fill(0); row.forEach((m, i) => { if (m) for (let p = 0; p < P; p++) y[p] += m * X[i][p]; }); return y; });
  const memo = {};

  function data(T, seed = 7) {
    const key = T + ':' + seed; if (memo[key]) return memo[key];
    const r = PE.rng(seed * 131 + T), g = PE.gauss(r);
    const C = [], B = [], X = [], a = [];
    for (let t = 0; t < T; t++) { C.push([g(), g()]); B.push([g(), g()]); X.push([g(), g()]); a.push(.62 + .36 * r()); }
    const forget = Math.round(T * .56); a[forget] = .12;
    const cum = []; a.reduce((s, v, t) => (cum[t] = s + Math.log(v)), 0);
    const Ld = (j, i) => j >= i ? Math.exp(cum[j] - cum[i]) : 0;   // a_j ... a_{i+1}; 1 on the diagonal (empty product)
    const Cf = C.map(v => v.map(phi)), Bf = B.map(v => v.map(phi));
    const L = zeros(T, T), G = zeros(T, T), soft = zeros(T, T), ssd = zeros(T, T);
    for (let j = 0; j < T; j++) {
      const e = []; for (let i = 0; i <= j; i++) e.push(Math.exp(dotv(C[j], B[i]) / Math.sqrt(N)));
      const Z = e.reduce((s, v) => s + v, 0);
      for (let i = 0; i <= j; i++) { L[j][i] = Ld(j, i); G[j][i] = dotv(Cf[j], Bf[i]); soft[j][i] = e[i] / Z; ssd[j][i] = L[j][i] * G[j][i]; }
    }
    const M = { softmax: soft, linattn: G, ssd };
    const Y = { softmax: matvec(soft, X), linattn: matvec(G, X), ssd: matvec(ssd, X) };
    const scale = Math.max(...G.flat());
    const d = { T, C, B, X, a, forget, cum, Cf, Bf, L, G, M, Y, scale };
    d.hSsd = recurrent(d).h; d.lin = linattnRecurrent(d);
    return (memo[key] = d);
  }

  // Mamba-2 Eq. 2 with A_t = a_t I: h_t = a_t h_{t-1} + B_t x_t^T (an N x P state), y_t = C_t^T h_t.
  function recurrent(d) {
    const h = zeros(N, P), Y = [];
    for (let t = 0; t < d.T; t++) {
      for (let n = 0; n < N; n++) for (let p = 0; p < P; p++) h[n][p] = d.a[t] * h[n][p] + d.Bf[t][n] * d.X[t][p];
      Y.push([0, 1].map(p => d.Cf[t][0] * h[0][p] + d.Cf[t][1] * h[1][p]));
    }
    return { Y, h };
  }
  // y = (L o C B^T) x, every entry of M formed explicitly.
  const quadratic = d => matvec(d.M.ssd, d.X);
  // Mamba-2 Listing 1 with chunk length Q (the last chunk may be shorter).
  function chunked(d, Q) {
    const T = d.T, nc = Math.ceil(T / Q), Y = zeros(T, P), cumAt = t => t < 0 ? 0 : d.cum[t];
    const lo = c => c * Q, hi = c => Math.min(T, c * Q + Q) - 1;
    // 1. Diagonal blocks: each chunk's output from its own inputs (zero initial state), in the quadratic form.
    for (let c = 0; c < nc; c++) for (let j = lo(c); j <= hi(c); j++) for (let i = lo(c); i <= j; i++) {
      const w = Math.exp(d.cum[j] - d.cum[i]) * dotv(d.Cf[j], d.Bf[i]);
      for (let p = 0; p < P; p++) Y[j][p] += w * d.X[i][p];
    }
    // 2. Right factors: each chunk's final state from its own inputs, an (N, Q) x (Q, P) product.
    const states = [];
    for (let c = 0; c < nc; c++) {
      const s = zeros(N, P);
      for (let i = lo(c); i <= hi(c); i++) { const w = Math.exp(d.cum[hi(c)] - d.cum[i]); for (let n = 0; n < N; n++) for (let p = 0; p < P; p++) s[n][p] += w * d.Bf[i][n] * d.X[i][p]; }
      states.push(s);
    }
    // 3. Center factors: a scalar recurrence over the nc chunk states gives the true state entering each chunk.
    const enter = [zeros(N, P)];
    for (let c = 0; c < nc; c++) {
      const w = Math.exp(d.cum[hi(c)] - cumAt(lo(c) - 1));
      enter.push(enter[c].map((row, n) => row.map((v, p) => w * v + states[c][n][p])));
    }
    // 4. Left factors: each chunk's output from the state that entered it, a (Q, N) x (N, P) product.
    for (let c = 0; c < nc; c++) for (let j = lo(c); j <= hi(c); j++) {
      const w = Math.exp(d.cum[j] - cumAt(lo(c) - 1));
      for (let p = 0; p < P; p++) Y[j][p] += w * (d.Cf[j][0] * enter[c][0][p] + d.Cf[j][1] * enter[c][1][p]);
    }
    return { Y, h: enter[nc] };
  }
  // Linear attention with its normalizer, Katharopoulos Eq. 9 (quadratic) and Eqs. 10-12 (the recurrent form).
  function linattnQuadratic(d) {
    return d.G.map((row, i) => { const Z = row.reduce((s, v) => s + v, 0); return [0, 1].map(p => row.reduce((s, v, j) => s + v * d.X[j][p], 0) / Z); });
  }
  function linattnRecurrent(d) {
    const S = zeros(N, P), z = [0, 0], Y = [];
    for (let i = 0; i < d.T; i++) {
      for (let n = 0; n < N; n++) { z[n] += d.Bf[i][n]; for (let p = 0; p < P; p++) S[n][p] += d.Bf[i][n] * d.X[i][p]; }
      const den = dotv(d.Cf[i], z); Y.push([0, 1].map(p => (d.Cf[i][0] * S[0][p] + d.Cf[i][1] * S[1][p]) / den));
    }
    return { Y, S, z };
  }
  // Numerical rank of the block rows r0..r1-1, columns c0..c1-1 (Gaussian elimination with full pivoting).
  function rank(M, r0, r1, c0, c1) {
    const A = []; for (let j = r0; j < r1; j++) A.push(M[j].slice(c0, c1));
    const m = A.length, n = A[0].length, tol = 1e-9 * Math.max(1e-300, ...A.flat().map(Math.abs));
    let rk = 0;
    for (let k = 0; k < Math.min(m, n); k++) {
      let bi = -1, bj = -1, bv = tol;
      for (let i = k; i < m; i++) for (let j = k; j < n; j++) if (Math.abs(A[i][j]) > bv) { bv = Math.abs(A[i][j]); bi = i; bj = j; }
      if (bi < 0) break;
      [A[k], A[bi]] = [A[bi], A[k]]; A.forEach(row => { [row[k], row[bj]] = [row[bj], row[k]]; });
      for (let i = k + 1; i < m; i++) { const f = A[i][k] / A[k][k]; for (let j = k; j < n; j++) A[i][j] -= f * A[k][j]; }
      rk++;
    }
    return rk;
  }
  // Multiply-adds per head, counting each matrix product densely (as matrix-multiply hardware runs it).
  // Pieces follow the Sec 6.3 cost list: diagonal blocks, right factors, center scan (T N P / Q), left factors.
  function work(T, Q) {
    const quad = T * T * (N + P), rec = 3 * T * N * P;
    if (!Q) return { quad, rec };
    const diag = T * Q * (N + P), right = T * N * P, pass = Math.ceil(T / Q) * N * P, left = T * N * P;
    return { quad, rec, chunk: diag + right + pass + left, mm: diag + right + left, steps: Math.ceil(T / Q) };
  }

  // A drawable target for (T, mode, Q): heights, color weights (brown quadratic, yellow through the state,
  // red full rank) and overlay strengths. Colors are resolved in items(), so a theme change only needs a redraw.
  function target(T, mode, Q) {
    const d = data(T), M = d.M[mode], sc = mode === 'softmax' ? 1 : d.scale;
    const h = new Float64Array(T * T), w = new Float64Array(T * T * 3);
    for (let j = 0; j < T; j++) for (let i = 0; i <= j; i++) {
      const k = j * T + i, diag = !Q || Math.floor(j / Q) === Math.floor(i / Q);
      h[k] = HS * M[j][i] / sc; w[3 * k + (diag ? 0 : mode === 'softmax' ? 2 : 1)] = 1;
    }
    const ym = Math.max(...d.Y[mode].flat().map(Math.abs));
    const st = mode === 'linattn' ? d.lin.S : d.hSsd, sm = Math.max(...st.flat().map(Math.abs)), zm = Math.max(...d.lin.z);
    return { T, Q, mode, h, w, qa: Q ? 1 : 0, y: d.Y[mode].map(r => r.map(v => 1.6 * XS * v / ym)),
      dec: d.L[T - 1].map(v => HD * (mode === 'linattn' ? 1 : v)), decA: mode === 'softmax' ? 0 : 1,
      mem: mode === 'softmax' ? 0 : 1, zA: mode === 'linattn' ? 1 : 0,
      st: st.map(r => r.map(v => Math.abs(v) / sm)), z: d.lin.z.map(v => v / zm) };
  }
  const lerp = (a, b, u) => a + (b - a) * u;
  function blend(A, B, u) {
    if (u >= 1) return B;
    if (!A || A.T !== B.T) return { ...B, h: B.h.map(v => v * u), y: B.y.map(r => r.map(v => v * u)) };
    const L2 = (x, y) => x.map((r, i) => Array.isArray(r) ? r.map((v, j) => lerp(v, y[i][j], u)) : lerp(r, y[i], u));
    return { T: B.T, Q: B.Q || A.Q, mode: B.mode, h: A.h.map((v, k) => lerp(v, B.h[k], u)), w: A.w.map((v, k) => lerp(v, B.w[k], u)),
      qa: lerp(A.qa, B.qa, u), y: L2(A.y, B.y), dec: L2(A.dec, B.dec), decA: lerp(A.decA, B.decA, u), mem: lerp(A.mem, B.mem, u),
      zA: lerp(A.zA, B.zA, u), st: L2(A.st, B.st), z: L2(A.z, B.z) };
  }

  const rgb = c => { if (c[0] === '#') return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); const m = /\(([^)]+)\)/.exec(c); return m ? m[1].split(',').map(Number) : [136, 136, 136]; };
  const wmix = (ws, cs) => { const s = ws.reduce((a, b) => a + b, 0) || 1; return `rgb(${[0, 1, 2].map(k => Math.round(ws.reduce((a, wv, i) => a + wv * cs[i][k], 0) / s)).join(',')})`; };
  function box(cx, cz, hw, y0, y1, top, side, link, alpha) {
    const P4 = (x, y, z) => [x, y, z], a = cx - hw, b = cx + hw, f = cz - hw, n = cz + hw;
    return [
      { type: 'quad', pts: [P4(a, y1, f), P4(b, y1, f), P4(b, y1, n), P4(a, y1, n)], fill: top, link, alpha },
      { type: 'quad', pts: [P4(a, y0, n), P4(b, y0, n), P4(b, y1, n), P4(a, y1, n)], fill: side, link, alpha },
      { type: 'quad', pts: [P4(a, y0, f), P4(b, y0, f), P4(b, y1, f), P4(a, y1, f)], fill: side, link, alpha },
      { type: 'quad', pts: [P4(a, y0, f), P4(a, y0, n), P4(a, y1, n), P4(a, y1, f)], fill: side, link, alpha },
      { type: 'quad', pts: [P4(b, y0, f), P4(b, y0, n), P4(b, y1, n), P4(b, y1, f)], fill: side, link, alpha }];
  }

  // items(v, S, glow): v from target()/blend(); S.narrow for short labels; glow(word) in [0, 1] from equation links.
  function items(v, S = {}, glow = () => 0) {
    const T = v.T, d = data(T), g = 3 / T, x0 = -1.6, z0 = -1.75, xe = x0 + T * g, ze = z0 + T * g, hw = .37 * g;
    const css = PE.css, bg = css('--bg'), cols = ['--brown', '--yellow', '--red'].map(n => rgb(css(n))), BG = rgb(bg);
    const gD = glow('decay'), gS = glow('scores'), out = [];
    // Floor: the full T x T square (dashed) and the causal staircase.
    out.push({ type: 'line', pts: [[x0, 0, z0], [xe, 0, z0], [xe, 0, ze], [x0, 0, ze], [x0, 0, z0]], color: '--rule', width: 1, dash: [4, 4] });
    const stair = [[x0, 0, z0]]; for (let k = 0; k < T; k++) stair.push([x0 + (k + 1) * g, 0, z0 + k * g], [x0 + (k + 1) * g, 0, z0 + (k + 1) * g]);
    out.push({ type: 'line', pts: [...stair, [x0, 0, ze], [x0, 0, z0]], color: '--muted', width: 1, alpha: .7 });
    // Matrix columns; while an equation term glows the heights show that factor alone.
    for (let j = 0; j < T; j++) for (let i = 0; i <= j; i++) {
      const k = j * T + i;
      let h = v.h[k], ws = [v.w[3 * k], v.w[3 * k + 1], v.w[3 * k + 2]];
      if (gD > 0) { h = lerp(h, HS * d.L[j][i], gD); ws = ws.map((x, q) => lerp(x, q === 1 ? 1 : 0, gD)); }
      if (gS > 0) { h = lerp(h, HS * d.G[j][i] / d.scale, gS); ws = ws.map((x, q) => lerp(x, q === 0 ? 1 : 0, gS)); }
      const top = wmix(ws, cols), side = wmix([.58, .42], [rgb(top), BG]);
      out.push(...box(x0 + (i + .5) * g, z0 + (j + .5) * g, hw, 0, Math.max(.006, h), top, side, 'scores', 1));
    }
    // Chunk overlay: block boundaries inside the triangle, diagonal blocks outlined in brown.
    if (v.qa > .01 && v.Q) {
      const Q = v.Q, nb = Math.ceil(T / Q), at = q => Math.min(T, q * Q);
      for (let b = 1; b < nb; b++) {
        out.push({ type: 'line', pts: [[x0 + at(b) * g, .004, z0 + at(b) * g], [x0 + at(b) * g, .004, ze]], color: '--fg', width: 1, alpha: .45 * v.qa });
        out.push({ type: 'line', pts: [[x0, .004, z0 + at(b) * g], [x0 + at(b) * g, .004, z0 + at(b) * g]], color: '--fg', width: 1, alpha: .45 * v.qa });
      }
      for (let b = 0; b < nb; b++) {
        const s = x0 + at(b) * g, e = x0 + at(b + 1) * g, zs = z0 + at(b) * g, zn = z0 + at(b + 1) * g;
        out.push({ type: 'line', pts: [[s, .006, zs], [e, .006, zs], [e, .006, zn], [s, .006, zn], [s, .006, zs]], color: '--brown', width: 2.2, alpha: v.qa });
      }
      const top = Math.max(...Array.from({ length: Math.min(Q, T) }, (_, j) => v.h[j * T])) + .12;
      if (!S.narrow) out.push({ type: 'label', at: [x0 - .08, .05, z0 + Math.min(Q, T) * g / 2], text: S.narrow ? 'quadratic' : 'diagonal block: quadratic', color: '--brown', alpha: v.qa, align: 'right' });
      if (nb > 1) {
        const soft = v.mode === 'softmax', r0 = at(nb - 1), rk = rank(d.M[v.mode], r0, T, 0, Math.min(Q, r0));
        const txt = Q === 1 ? (soft ? 'no state can carry this' : 'every entry: through the state') : soft ? `rank ${rk} of ${Math.min(Q, r0)}: no shortcut` : `rank ${rk}: through the state`;
        if (!S.narrow) out.push({ type: 'label', at: [x0 - .08, .05, z0 + (r0 + T) * g / 2], text: txt, color: soft ? '--red' : '--yellow', alpha: v.qa, align: 'right' });
      }
    }
    // Inputs x (back edge, one row per channel) and outputs y (right edge), as lollipops.
    for (let p = 0; p < P; p++) {
      const zx = z0 - .2 - .16 * p, xy = xe + .2 + .16 * p, xp = [], yp = [];
      for (let i = 0; i < T; i++) {
        const cx = x0 + (i + .5) * g, hx = XS * d.X[i][p];
        out.push({ type: 'line', pts: [[cx, 0, zx], [cx, hx, zx]], color: '--blue', width: 1.4, link: 'inputs', alpha: .85 }); xp.push([cx, hx, zx]);
        const cz = z0 + (i + .5) * g, hy = v.y[i][p];
        out.push({ type: 'line', pts: [[xy, 0, cz], [xy, hy, cz]], color: '--green', width: 1.4, link: 'output', alpha: .85 }); yp.push([xy, hy, cz]);
      }
      out.push({ type: 'points', pts: xp, color: '--blue', size: S.narrow ? 2.6 : 3.4, link: 'inputs' });
      out.push({ type: 'points', pts: yp, color: '--green', size: S.narrow ? 2.6 : 3.4, link: 'output' });
    }
    out.push({ type: 'label', at: [x0, .42, z0 - .28], text: S.narrow ? 'inputs x' : 'inputs x, by position s', color: '--blue', align: 'left' });
    out.push({ type: 'label', at: [xe + .28, .55, z0 + .1], text: S.narrow ? 'outputs y' : 'outputs y, by position t', color: '--green' });
    // Memory at "now" (the front edge): the KV cache, or the decay line plus the fixed state register.
    const zf = ze + .2, kv = 1 - v.mem, cs = Math.min(.7 * g, .12);
    if (kv > .01) {
      const br = css('--brown'), brV = PE3.mix(br, bg, .38);
      for (let i = 0; i < T; i++) { const cx = x0 + (i + .5) * g;
        out.push(...box(cx, zf, cs / 2 * kv, 0, cs * kv, br, br, 'cache', kv), ...box(cx, zf + cs * 1.25, cs / 2 * kv, 0, cs * kv, brV, brV, 'cache', kv)); }
      out.push({ type: 'label', at: [0, 0, zf + cs * 2], text: S.narrow ? `KV cache: ${T} keys, ${T} values` : `KV cache: a key and a value for all ${T} tokens`, color: '--brown', alpha: kv, dy: 16 });
    }
    const dA = Math.max(v.decA, gD);
    if (dA > .01) {
      const pts = v.dec.map((hh, i) => [x0 + (i + .5) * g, hh, zf]);
      out.push({ type: 'line', pts, color: '--yellow', width: 2, link: 'decay', alpha: dA }, { type: 'points', pts, color: '--yellow', size: S.narrow ? 2.4 : 3, link: 'decay', alpha: dA });
      if (!S.narrow) out.push({ type: 'label', at: [x0 - .05, v.dec[0], zf], text: S.narrow ? 'decay' : 'decay to now', color: '--yellow', align: 'right', alpha: dA, dx: -4 });
    }
    const rA = Math.max(v.mem, glow('state'), glow('norm'));
    if (rA > .01) {
      const yel = css('--yellow'), rc = .13, rp = .16, rx = xe + .22, rz = ze + .24;
      for (let n = 0; n < N; n++) for (let p = 0; p < P; p++) {
        const f = PE3.mix(bg, yel, .25 + .75 * v.st[n][p]);
        out.push(...box(rx + p * rp, rz + n * rp, rc / 2, 0, rc, f, f, 'state', rA));
      }
      const zA = Math.max(v.zA * v.mem, glow('norm'));
      if (zA > .01) for (let n = 0; n < N; n++) { const f = PE3.mix(bg, yel, .15 + .45 * v.z[n]); out.push(...box(rx + P * rp + .05, rz + n * rp, rc / 2, 0, rc * .8, f, f, 'norm', zA)); }
      out.push({ type: 'label', at: [rx + rp / 2, rc + .1, rz + rp], text: v.zA > .5 ? (S.narrow ? 'state S, z' : 'state S and normalizer z') : (S.narrow ? `state ${N} x ${P}` : `state h: ${N} x ${P}, fixed`), color: '--yellow', alpha: rA, dy: -14 });
    }
    return out;
  }
  // Camera fitted to the stage: pe3's field of view is vertical, so a narrow stage pulls the camera back.
  function view(W, H) {
    const narrow = W < 560, r = Math.max(.3, W / Math.max(H, 1));
    return { narrow, yaw: narrow ? .32 : .42, pitch: .72, dist: narrow ? Math.max(7, 5.4 / (.61 * r)) : Math.max(8.2, 7.4 / (.6 * Math.min(r, 1.7))) };
  }

  const fmt = x => Math.round(x).toLocaleString('en-US');
  function readout(S) {
    const d = data(S.T), T = S.T, Q = S.Q, w = work(T, Q), cnt = T * (T + 1) / 2;
    const m = { softmax: `Softmax attention: each row is a softmax over the tokens so far, so each of the ${cnt} entries depends on its own query-key pair through exp. Generating needs the KV cache: ${T} keys and ${T} values.`,
      linattn: `Linear attention: entry (t, s) is phi(c_t) . phi(b_s) with no decay (the decay line is flat at 1). The whole past fits in a ${N} x ${P} state plus a normalizer z.`,
      ssd: `SSD: the same scores times the decay mask L. Entry (t, s) is scaled by a_t ... a_(s+1); the token at position ${d.forget} has a = ${d.a[d.forget]} and wipes most of what came before it.` }[S.mode];
    let c = '';
    if (Q) {
      const nb = Math.ceil(T / Q), r0 = (nb - 1) * Q, rk = nb > 1 ? rank(d.M[S.mode], r0, T, 0, Math.min(Q, r0)) : 0;
      c = S.mode === 'softmax' ? ` Chunks of Q = ${Q}: the bottom-left block has full rank ${rk}, so no state of size ${N} can stand in for it.`
        : ` Chunks of Q = ${Q}: ${nb} diagonal blocks by the quadratic form; every block below them has rank at most N = ${N} (the bottom-left one measures ${rk}), so it goes through the state.`;
    }
    const wq = `Work per head at T = ${T}: quadratic ${fmt(w.quad)} multiply-adds in one parallel step`;
    const wc = Q ? (Q === 1 ? `; chunks of 1 make ${fmt(w.chunk)} in ${w.steps} sequential steps (this is the recurrence)` : `; chunked ${fmt(w.chunk)}, ${Math.round(100 * w.mm / w.chunk)}% of it in matrix products, with ${w.steps} state hand-offs`) : '';
    let chk = '';
    if (S.mode === 'ssd') { const q = quadratic(d), e = Math.max(...[recurrent(d).Y, chunked(d, Q || 4).Y].flatMap(Y => Y.flatMap((r, t) => r.map((x, p) => Math.abs(x - q[t][p]))))); chk = ` Quadratic, recurrent and chunked give the same y here (largest difference ${e.toExponential(0)}).`; }
    return `${m}${c} ${wq}${wc}; recurrent ${fmt(w.rec)} in ${T} sequential steps, none of it matrix products.${chk}`;
  }
  return { N, P, data, recurrent, quadratic, chunked, linattnQuadratic, linattnRecurrent, rank, work, target, blend, items, view, readout };
})();
