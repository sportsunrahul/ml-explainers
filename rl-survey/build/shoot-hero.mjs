// node shoot-hero.mjs <html> <outdir>   (cwd = ~/paper-explainers/.node): hero final frame at 1280 dark/light, 375 dark, and reduced motion.
import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2);
const b = await chromium.launch();
for (const [name, vp, cs, rm] of [['dark', 1280, 'dark', 'no-preference'], ['light', 1280, 'light', 'no-preference'], ['mobile', 375, 'dark', 'no-preference'], ['reduced', 1280, 'dark', 'reduce'], ['mid', 1280, 'dark', 'no-preference']]) {
  const p = await (await b.newContext({ viewport: { width: vp, height: 900 }, colorScheme: cs, reducedMotion: rm })).newPage();
  p.on('pageerror', e => console.log('ERR', name, e.message)); p.on('console', m => { if (m.type() === 'error') console.log('CONSOLE', name, m.text()); });
  await p.goto('file://' + file); await p.waitForTimeout(name === 'mid' ? 2600 : rm === 'reduce' ? 400 : 7600);
  await p.locator('.hero-stage').screenshot({ path: `${out}/hero-${name}.png` });
}
await b.close();
