import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const b = await chromium.launch(); const p = await (await b.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'dark', reducedMotion: 'reduce' })).newPage();
await p.goto('file:///Users/rahulmishra/paper-explainers/dino/dino-explained.html', { waitUntil: 'networkidle' });
await p.waitForTimeout(3000);
for (const w of ['koleo', 'nn', 'pts', 'gram', 'gs']) {
  const fig = w === 'gram' || w === 'gs' ? 'f-gram' : 'f-koleo';
  await p.$eval('#' + fig, f => f.scrollIntoView({ block: 'center' })); await p.waitForTimeout(1500);
  const t0 = Date.now();
  await p.$eval(`.eq [data-link="${w}"]`, t => t.focus({ preventScroll: true }));
  await p.waitForTimeout(120);
  const st = await p.evaluate(() => ({ lit: [...document.querySelectorAll('.pe-lit')].map(e => e.id || e.className), active: document.activeElement && document.activeElement.dataset.link }));
  const t1 = Date.now();
  await p.locator('#' + fig).screenshot({ path: `/Users/rahulmishra/paper-explainers/dino/shots/eqs/pulse-${w}.png` });
  console.log(w, JSON.stringify(st), 'eval at', t1 - t0, 'shot done at', Date.now() - t0);
  await p.$eval(`.eq [data-link="${w}"]`, t => t.blur()); await p.waitForTimeout(700);
}
await b.close();
