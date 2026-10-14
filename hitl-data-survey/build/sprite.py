# Pack exported 32x32 tiles into one PNG sprite (pure python + zlib) and a JSON index; out: build/data/<name>.png/.json
import json, struct, sys, zlib, numpy as np
src, name, cols = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16
d = np.load(src); img, idx = d['img'], d['idx']; n = len(idx); rows = (n + cols - 1) // cols
S = np.zeros((rows * 32, cols * 32, 3), np.uint8)
for k in range(n): r, c = divmod(k, cols); S[r*32:(r+1)*32, c*32:(c+1)*32] = img[k]
raw = b''.join(b'\x00' + S[y].tobytes() for y in range(S.shape[0]))
def chunk(t, b): return struct.pack('>I', len(b)) + t + b + struct.pack('>I', zlib.crc32(t + b) & 0xffffffff)
png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', S.shape[1], S.shape[0], 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
P = '/'.join(src.split('/')[:-1])
open(f'build/data/{name}.png', 'wb').write(png); json.dump({'cols': cols, 'idx': idx.tolist()}, open(f'build/data/{name}.json', 'w'))
print(name, n, 'tiles', len(png) // 1024, 'KB')
