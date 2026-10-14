import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const b = await chromium.launch(); const p = await (await b.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'dark' })).newPage();
await p.goto('file:///Users/rahulmishra/paper-explainers/dino/dino-explained.html', { waitUntil: 'networkidle' });
await p.evaluate(() => { document.documentElement.style.scrollBehavior = 'auto'; document.getElementById('choose').scrollIntoView(); scrollBy(0, -260); });
await p.waitForTimeout(800);
await p.screenshot({ path: '/Users/rahulmishra/paper-explainers/dino/shots/eqs/choose-1440.png' });
console.log(await p.evaluate(() => ['#compare .prose', '#choose h2', 'footer p'].map(q => q + ' ' + Math.round(document.querySelector(q).getBoundingClientRect().left))));
await b.close();
