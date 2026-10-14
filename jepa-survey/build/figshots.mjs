// node figshots.mjs <html> <outdir>: screenshot every figure.viz in its initial state at 1280 dark, plus 375 dark.
import path from 'path'; import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2); fs.mkdirSync(out, { recursive: true });
const b = await chromium.launch();
for (const [name, vp] of [['d', { width: 1280, height: 900 }], ['m', { width: 375, height: 812 }]]) {
  const p = await (await b.newContext({ viewport: vp, colorScheme: 'dark' })).newPage();
  p.on('pageerror', e => console.log('ERR', e.message)); await p.goto('file://' + file); await p.waitForTimeout(1500);
  const ids = ['hero-stage', ...await p.$$eval('figure.viz', es => es.map(e => e.id))];
  for (const id of ids) { const loc = id === 'hero-stage' ? p.locator('.hero-stage') : p.locator('#' + id); await loc.scrollIntoViewIfNeeded(); await p.waitForTimeout(700); await loc.screenshot({ path: `${out}/${name}-${id.replace(/\W+/g, '_')}.png` }); }
  await p.close();
}
await b.close(); console.log('done');
