// node shoot.mjs <html> <outdir> [prefix]   (cwd ~/paper-explainers/.node)
import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out, pre = 'cur'] = process.argv.slice(2);
const b = await chromium.launch();
for (const [name, vp, cs] of [['desk-dark', { width: 1440, height: 900 }, 'dark'], ['desk-light', { width: 1440, height: 900 }, 'light'], ['mob-dark', { width: 375, height: 812 }, 'dark']]) {
  const p = await (await b.newContext({ viewport: vp, colorScheme: cs })).newPage();
  await p.goto("file://" + file); await p.waitForTimeout(6500);
  await p.screenshot({ path: `${out}/${pre}-${name}-top.png` });
  await p.evaluate(() => document.getElementById('ch-shared-space').scrollIntoView()); await p.waitForTimeout(800);
  await p.screenshot({ path: `${out}/${pre}-${name}-ch1.png` });
  await p.evaluate(() => document.getElementById('viz-clipflow').scrollIntoView()); await p.waitForTimeout(800);
  await p.screenshot({ path: `${out}/${pre}-${name}-flow.png` });
}
await b.close();
