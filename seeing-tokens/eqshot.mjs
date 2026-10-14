import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const out = process.argv[3]; const b = await chromium.launch();
for (const [w, cs] of [[1280, 'dark'], [1280, 'light'], [375, 'dark']]) {
  const p = await (await b.newContext({ viewport: { width: w, height: 900 }, colorScheme: cs })).newPage();
  await p.goto('file://' + process.argv[2], { waitUntil: 'networkidle' }); await p.waitForTimeout(800);
  for (const id of ['eq-vae', 'eq-vq']) await p.locator('#' + id).screenshot({ path: `${out}/${id}-${w}-${cs}.png` });
  await p.close();
}
await b.close();
