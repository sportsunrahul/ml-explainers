import { chromium } from 'playwright';
const [,, html, out] = process.argv;
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1280, height: 900 }, colorScheme: 'dark', reducedMotion: 'reduce' });
await p.goto('file://' + html); await p.waitForTimeout(400); await p.click('#btn-sam2-click');
for (const f of [14, 18]) { await p.$eval('#sam2-frame', (e, v) => { e.value = v; e.dispatchEvent(new Event('input')); }, f); await p.waitForTimeout(200); await p.locator('#viz-sam2').screenshot({ path: `${out}/sam2-scrub-${f}.png` }); }
await b.close();
