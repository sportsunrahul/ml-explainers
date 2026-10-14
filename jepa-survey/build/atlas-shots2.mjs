// Targeted states: later steps and views the smoke test does not screenshot.
import path from 'path'; import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file, out] = process.argv.slice(2); fs.mkdirSync(out, { recursive: true });
const b = await chromium.launch();
const shots = [['ijepa', async p => { await p.click('article[data-chapter="ijepa"] [data-step="3"]'); }, 'ijepa-loss'],
  ['ijepa', async p => { await p.click('article[data-chapter="ijepa"] [data-step="2"]'); }, 'ijepa-predict'],
  ['loss', async p => { await p.click('article[data-chapter="loss"] [data-step="2"]'); await p.fill('#loss-noise', '0.8'); await p.$eval('#loss-noise', e => e.dispatchEvent(new Event('input'))); }, 'loss-jepa'],
  ['loss', async p => { await p.fill('#loss-noise', '0.8'); await p.$eval('#loss-noise', e => e.dispatchEvent(new Event('input'))); }, 'loss-pixels-noisy'],
  ['family', async p => {}, 'family-months'], ['family', async p => { await p.click('[data-view="domains"]'); }, 'family-domains'],
  ['video', async p => {}, 'video-short'], ['collapse', async p => { await p.click('article[data-chapter="collapse"] [data-act="run"]'); await p.waitForTimeout(2500); }, 'collapse-trained'],
  ['collapse', async p => { await p.click('article[data-chapter="collapse"] [data-key="sg"][data-val="0"]'); await p.click('article[data-chapter="collapse"] [data-act="run"]'); await p.waitForTimeout(2500); }, 'collapse-nosg']];
for (const [vpName, vp] of [['d', { width: 1440, height: 900 }], ['m', { width: 390, height: 844 }]]) {
  for (const [ch, act, name] of shots) {
    const p = await (await b.newContext({ viewport: vp, colorScheme: 'dark' })).newPage(); const errs = []; p.on('pageerror', e => errs.push(e.message));
    await p.goto('file://' + file + '#' + ch); await p.waitForTimeout(500); await act(p); await p.waitForTimeout(1500);
    await p.screenshot({ path: `${out}/${vpName}-${name}.png` }); if (errs.length) console.log(name, errs); await p.close();
  }
}
await b.close(); console.log('done');
