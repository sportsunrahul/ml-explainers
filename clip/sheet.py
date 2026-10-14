import sys, glob, os
from PIL import Image, ImageDraw
out, cols, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
files = sys.argv[4:]
ims = [Image.open(f).convert('RGB') for f in files]
cw = width // cols
tiles = []
for f, im in zip(files, ims):
    h = int(im.height * cw / im.width)
    t = Image.new('RGB', (cw, h + 22), (60, 60, 60))
    t.paste(im.resize((cw, h)), (0, 22))
    ImageDraw.Draw(t).text((6, 4), os.path.basename(f), fill=(255, 255, 0))
    tiles.append(t)
rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
H = sum(max(t.height for t in r) for r in rows)
sheet = Image.new('RGB', (cw * cols, H), (40, 40, 40))
y = 0
for r in rows:
    for k, t in enumerate(r):
        sheet.paste(t, (k * cw, y))
    y += max(t.height for t in r)
sheet.save(out)
print(out, sheet.size)
