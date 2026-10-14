import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2);
const b = await chromium.launch();
for (const [name, vp, cs, rm] of [['dark', { width: 1280, height: 900 }, 'dark', 'no-preference'], ['light', { width: 1280, height: 900 }, 'light', 'reduce'], ['mobile', { width: 375, height: 812 }, 'dark', 'reduce']]) {
  const p = await (await b.newContext({ viewport: vp, colorScheme: cs, reducedMotion: rm })).newPage();
  p.on('pageerror', e => console.log('ERR', e.message)); p.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', m.text()); });
  await p.goto('file://' + file);
  if (rm !== 'reduce') { await p.waitForTimeout(2500); await p.locator('#hero-stage').screenshot({ path: `${out}/hero-${name}-mid.png` }); await p.waitForTimeout(4500); }
  else await p.waitForTimeout(600);
  await p.locator('#hero-stage').screenshot({ path: `${out}/hero-${name}-end.png` });
}
await b.close();
