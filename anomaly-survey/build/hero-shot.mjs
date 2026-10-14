// Screenshots the hero's final frame (dark, light, mobile) after the animation ends.
import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2);
const b = await chromium.launch();
for (const [name, vp, cs] of [['dark', { width: 1280, height: 900 }, 'dark'], ['light', { width: 1440, height: 900 }, 'light'], ['mobile', { width: 375, height: 812 }, 'dark']]) {
  const p = await (await b.newContext({ viewport: vp, colorScheme: cs })).newPage();
  p.on('pageerror', e => console.log('ERR', e.message));
  await p.goto('file://' + file); await p.waitForTimeout(2600);
  await p.locator('.hero-stage').screenshot({ path: `${out}/hero-${name}-mid.png` });
  await p.waitForTimeout(4600);
  await p.locator('header.hero').screenshot({ path: `${out}/hero-${name}.png` });
}
await b.close();
