// node eqshots.mjs   (cwd = ~/paper-explainers/.node): screenshots of every .eq, and the readouts in the original for comparison
import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const D = '/Users/rahulmishra/paper-explainers/dino/';
const b = await chromium.launch();
for (const [w, cs] of [[1440, 'dark'], [1440, 'light'], [375, 'dark']]) {
  const p = await (await b.newContext({ viewport: { width: w, height: 900 }, colorScheme: cs })).newPage();
  await p.goto('file:///Users/rahulmishra/paper-explainers/dino/dino-explained.html', { waitUntil: 'networkidle' });
  await p.waitForTimeout(800);
  const n = await p.$$eval('.eq', e => e.length);
  for (let i = 0; i < n; i++) await p.locator('.eq').nth(i).screenshot({ path: `${D}shots/eqs/eq${i}-${w}-${cs}.png` });
  await p.close();
}
const p = await (await b.newContext({ viewport: { width: 1280, height: 900 }, colorScheme: 'dark' })).newPage();
await p.goto('file://' + D + 'original.html', { waitUntil: 'networkidle' });
for (const id of ['k-read', 'g-read']) { await p.$eval('#' + id, e => e.scrollIntoView({ block: 'center' })); await p.waitForTimeout(1200); await p.locator('#' + id).screenshot({ path: `${D}shots/eqs/orig-${id}.png` }); }
await b.close();
