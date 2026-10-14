// node hero-shot.mjs <html> <outdir> [ms]  (cwd ~/paper-explainers/.node): hero stage at ms after load, three views
import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out, ms = '8200'] = process.argv.slice(2);
const b = await chromium.launch();
for (const [name, vp, cs] of [['dark', { width: 1280, height: 900 }, 'dark'], ['light', { width: 1440, height: 900 }, 'light'], ['mobile', { width: 375, height: 812 }, 'dark']]) {
  const p = await (await b.newContext({ viewport: vp, colorScheme: cs })).newPage();
  p.on('pageerror', e => console.log('ERR', e.message));
  await p.goto('file://' + file); await p.waitForTimeout(+ms);
  await p.locator('.hero-stage').screenshot({ path: `${out}/hero-${name}-${ms}.png` });
}
await b.close();
