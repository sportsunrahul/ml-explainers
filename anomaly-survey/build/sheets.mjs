// Renders build/contact/sheet-*.html to PNG.
import path from 'path'; import fs from 'fs'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const dir = process.argv[2]; const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1960, height: 900 } });
for (const f of fs.readdirSync(dir).filter(f => f.startsWith('sheet-') && f.endsWith('.html'))) {
  await p.goto('file://' + path.join(dir, f)); await p.waitForTimeout(200);
  await p.screenshot({ path: path.join(dir, f.replace('.html', '.png')), fullPage: true }); }
await b.close(); console.log('ok');
