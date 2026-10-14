# Package run A (CE, 40% symmetric noise) for the page: ridge histograms, per-epoch series, GMM loss samples, claims.
import json, numpy as np, sys
P = sys.argv[1]
h = json.load(open(f'{P}/out/ce_sym40.json')); L = np.load(f'{P}/out/labels.npz'); d = np.load(f'{P}/out/ce_sym40.npz')
yt, ys = L['y_true'], L['y_sym40']; noisy = ys != yt
losses = d['losses'].astype(np.float32); preds = d['preds']; E = losses.shape[0]
acc = [x['test_acc'] for x in h]
lo, hi, nb = -4.0, 1.2, 52
edges = np.linspace(lo, hi, nb + 1)
slices = list(range(1, 61)) + list(range(70, E + 1, 10))
ridge = []
for e in slices:
    lg = np.log10(np.maximum(losses[e - 1], 1e-4))
    hc, _ = np.histogram(lg[~noisy], edges); hn, _ = np.histogram(lg[noisy], edges)
    ridge.append({'e': e, 'c': hc.tolist(), 'n': hn.tolist()})
series = []
for e in range(1, E + 1):
    p, l = preds[e - 1], losses[e - 1]
    keep = np.argsort(l)[:int(0.6 * len(l))]
    series.append({'e': e, 'test': round(acc[e - 1], 4), 'fitWrong': round(float((p[noisy] == ys[noisy]).mean()), 4),
                   'asTrue': round(float((p[noisy] == yt[noisy]).mean()), 4), 'prec60': round(float((~noisy[keep]).mean()), 4)})
rng = np.random.default_rng(5); sub = rng.choice(len(ys), 1500, replace=False)
gmm = {str(e): [round(float(v), 3) for v in losses[e - 1][sub]] for e in [3, 5, 8, 10, 12, 15, 20, 30]}
out = {'noiseFrac': round(float(noisy.mean()), 4), 'nTrain': int(len(ys)), 'edges': [lo, hi, nb], 'ridge': ridge, 'series': series,
       'gmm': {'idx': sub.tolist(), 'noisy': noisy[sub].astype(int).tolist(), 'loss': gmm}}
json.dump(out, open(f'{P}/build/data/runA.json', 'w'), separators=(',', ':'))
i = int(np.argmax(acc)); s = {x['e']: x for x in series}
claims = [
 dict(id='run-a-noise-frac', claim='fraction of CIFAR-10 training labels flipped in our symmetric-noise run', value=f'{100*noisy.mean():.2f}%'),
 dict(id='run-a-peak-acc', claim='peak test accuracy of plain cross-entropy training in run A', value=f'{100*acc[i]:.2f}%'),
 dict(id='run-a-peak-epoch', claim='epoch of the peak test accuracy in run A', value=f'epoch {i+1}'),
 dict(id='run-a-last10', claim='mean test accuracy over the last 10 epochs in run A', value=f'{100*np.mean(acc[-10:]):.2f}%'),
 dict(id='run-a-e10-astrue', claim='share of mislabeled training images predicted as their true class at epoch 10', value=f'{100*s[10]["asTrue"]:.1f}%'),
 dict(id='run-a-e10-fitwrong', claim='share of mislabeled training images predicted as their wrong given label at epoch 10', value=f'{100*s[10]["fitWrong"]:.1f}%'),
 dict(id='run-a-e30-fitwrong', claim='share of mislabeled training images predicted as their wrong given label at epoch 30', value=f'{100*s[30]["fitWrong"]:.1f}%'),
 dict(id='run-a-e10-prec60', claim='share of clean labels among the 60% smallest-loss training images at epoch 10', value=f'{100*s[10]["prec60"]:.1f}%'),
 dict(id='run-a-e200-prec60', claim='share of clean labels among the 60% smallest-loss training images at epoch 200', value=f'{100*s[200]["prec60"]:.1f}%'),
]
for c in claims: c.update(paper='our-run-a', version='job/run_noise.py', loc='out/ce_sym40.json, out/ce_sym40.npz (build/prep_runA.py)', source='our-run')
json.dump(claims, open(f'{P}/notes/claims/our-run-a.json', 'w'), indent=1)
print(len(ridge), 'ridge slices;', len(json.dumps(out)) // 1024, 'KB'); [print(c['id'], c['value']) for c in claims]
