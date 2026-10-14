// node find-err.mjs <html> : load in reduced motion at 1280 and 375, click every button, print page errors with stacks.
import path from 'path'; import { createRequire } from 'module';
const require = createRequire(path.join(process.cwd(), 'x.js')); const { chromium } = require('playwright');
const [file] = process.argv.slice(2);
const b = await chromium.launch();
for (const width of [1280, 375]) {
  const p = await (await b.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' })).newPage();
  p.on('pageerror', e => console.log('ERR', width, e.message, '\n', (e.stack || '').split('\n').slice(0, 6).join('\n')));
  await p.goto('file://' + file); await p.waitForTimeout(500);
  const ids = await p.$$eval('button', bs => bs.filter(b => b.offsetParent !== null).map(b => b.id).filter(Boolean));
  for (const id of ids) { await p.click('#' + id).catch(e => console.log('clickfail', id)); await p.waitForTimeout(30); }
  await p.waitForTimeout(500);
}
await b.close();
