import { chromium } from 'playwright';
const [,, html, out, w = '1800'] = process.argv;
const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: +w, height: 1000 } });
await p.goto('file://' + html); await p.waitForTimeout(500); await p.screenshot({ path: out, fullPage: true }); await b.close();
