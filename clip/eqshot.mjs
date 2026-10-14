// node eqshot.mjs <page.html> <outdir>  (cwd = ~/paper-explainers/.node): screenshots of the .eq block, plain and with one term focused.
import path from 'path'; import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2); fs.mkdirSync(out, { recursive: true });
const b = await chromium.launch();
for (const [w, cs] of [[1440, 'dark'], [1440, 'light'], [375, 'dark'], [375, 'light']]) {
  const p = await (await b.newContext({ viewport: { width: w, height: 900 }, colorScheme: cs })).newPage();
  await p.goto('file://' + file, { waitUntil: 'networkidle' }); await p.waitForTimeout(800);
  const eq = p.locator('#eq-loss'); await eq.scrollIntoViewIfNeeded();
  await eq.screenshot({ path: `${out}/eq-${w}-${cs}.png` });
  if (w === 1440) { await p.$eval('#eq-loss [data-link="i2t"]', t => t.focus({ preventScroll: true })); await p.waitForTimeout(300);
    await eq.screenshot({ path: `${out}/eq-${w}-${cs}-focus.png` }); }
  await p.close();
}
await b.close();
