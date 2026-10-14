"""python3 sheet.py <slug>: contact sheet of the key top/ shots -> <slug>/shots/review.png"""
import sys, os
from PIL import Image, ImageDraw
slug = sys.argv[1]; D = f'/Users/rahulmishra/paper-explainers/{slug}/shots/top/'
names = ['hero-mid-1500', 'hero-mid-3200', 'hero-mid-4800', 'hero-1440-dark', 'hero-1440-light', 'hero-reduced',
         'hero-375-dark', 'hero-375-light', 'menu-375-dark', 'top-1440-light', 'mid-1440-dark', 'mid-375-light']
fs = [n for n in names if os.path.exists(D + n + '.png')]
W = 520
ims = [Image.open(D + n + '.png') for n in fs]
th = [(W, int(i.height * W / i.width)) if i.width > 600 else (int(i.width * min(1, 700 / i.height)), int(i.height * min(1, 700 / i.height))) for i in ims]
rows = [list(range(k, min(k + 3, len(fs)))) for k in range(0, len(fs), 3)]
H = sum(max(th[i][1] for i in r) + 22 for r in rows)
S = Image.new('RGB', (3 * W + 20, H), (60, 60, 60)); d = ImageDraw.Draw(S); y = 0
for r in rows:
    for c, i in enumerate(r):
        d.text((c * (W + 10) + 4, y + 4), fs[i], fill=(255, 255, 0)); S.paste(ims[i].resize(th[i]), (c * (W + 10), y + 18))
    y += max(th[i][1] for i in r) + 22
S.save(f'/Users/rahulmishra/paper-explainers/{slug}/shots/review.png'); print(slug, S.size)
