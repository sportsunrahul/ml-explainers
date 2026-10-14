import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const b = await chromium.launch();
for (const f of ['/Users/rahulmishra/paper-explainers/dino/original.html', '/Users/rahulmishra/paper-explainers/dino/dino-explained.html']) {
  const p = await (await b.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'dark' })).newPage();
  await p.goto('file://' + f, { waitUntil: 'networkidle' }); await p.waitForTimeout(2500);
  const out = {};
  for (const fig of ['f-koleo','f-gram','f-koleo','f-gram','f-koleo']) {
    await p.$eval('#' + fig, e => e.scrollIntoView({ block: 'center' })); await p.waitForTimeout(1500);
    out[fig + Object.keys(out).length] = await p.evaluate(() => new Promise(r => { let n = 0, mx = 0, last = performance.now(); const t0 = last; const f = now => { mx = Math.max(mx, now - last); last = now; n++; if (now - t0 < 2000) requestAnimationFrame(f); else r({ fps: Math.round(n / ((now - t0) / 1000)), maxFrame: Math.round(mx) }); }; requestAnimationFrame(f); }));
  }
  console.log(path.basename(f), JSON.stringify(out)); await p.close();
}
await b.close();
