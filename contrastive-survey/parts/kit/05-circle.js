/* Losses on small embeddings, shared by the hero and the chapter toys. Pure math (no PE), so node can run them. */
// NT-Xent (SimCLR Eq. 1) on 2D unit vectors given as angles. Points k and k+N are the two views of image k.
// Returns the mean loss and dL/dtheta for every point.
KIT.ntxentAngles = (th, tau) => {
  const M = th.length, N = M / 2, z = th.map(t => [Math.cos(t), Math.sin(t)]);
  const gz = z.map(() => [0, 0]);
  let L = 0;
  for (let a = 0; a < M; a++) {
    const p = a < N ? a + N : a - N;
    let mx = -Infinity; const s = [];
    for (let k = 0; k < M; k++) { s[k] = k === a ? -Infinity : (z[a][0] * z[k][0] + z[a][1] * z[k][1]) / tau; if (s[k] > mx) mx = s[k]; }
    let Z = 0; for (let k = 0; k < M; k++) if (k !== a) Z += Math.exp(s[k] - mx);
    L += -s[p] + mx + Math.log(Z);
    for (let k = 0; k < M; k++) {
      if (k === a) continue;
      const c = (Math.exp(s[k] - mx) / Z - (k === p ? 1 : 0)) / tau;
      gz[a][0] += c * z[k][0]; gz[a][1] += c * z[k][1];
      gz[k][0] += c * z[a][0]; gz[k][1] += c * z[a][1];
    }
  }
  const g = th.map((t, k) => (-Math.sin(t) * gz[k][0] + Math.cos(t) * gz[k][1]) / M);
  return { loss: L / M, g };
};
// One gradient step on the angles, in place. Returns the loss before the step.
KIT.ntxentStep = (th, tau, lr) => { const r = KIT.ntxentAngles(th, tau); for (let k = 0; k < th.length; k++) th[k] -= lr * r.g[k]; return r.loss; };
// Smallest angular gap between neighbouring pair centres, as a fraction of the even spacing 2 pi / N.
KIT.circleStats = th => {
  const N = th.length / 2, wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  let pair = 0; const c = [];
  for (let i = 0; i < N; i++) { const d = wrap(th[i + N] - th[i]); pair += Math.abs(d); c.push(wrap(th[i] + d / 2)); }
  c.sort((a, b) => a - b);
  let gap = Infinity; for (let i = 0; i < N; i++) gap = Math.min(gap, (c[(i + 1) % N] - c[i] + 2 * Math.PI) % (2 * Math.PI) || 2 * Math.PI);
  return { pairGap: pair / N, spread: gap / (2 * Math.PI / N) };
};
