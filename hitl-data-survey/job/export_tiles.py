# Export CIFAR-10 train images by index: python export_tiles.py <out.npz> <idx,idx,...>
import pickle, sys, numpy as np
d = '/root/work/paper-explainer/data/cifar-10-batches-py'
xs = []
for i in range(1, 6):
    b = pickle.load(open(f'{d}/data_batch_{i}', 'rb'), encoding='bytes'); xs.append(b[b'data'])
X = np.concatenate(xs).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
idx = np.array([int(v) for v in sys.argv[2].split(',')])
np.savez(sys.argv[1], idx=idx, img=X[idx])
print('exported', len(idx))
