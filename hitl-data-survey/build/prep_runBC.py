# Package run B (Co-teaching) and run C (4-fold CV probabilities, asymmetric noise) for the page, plus our-run claims.
import json, numpy as np, sys
P = sys.argv[1]
L = np.load(f'{P}/out/labels.npz'); yt, ys, ya = L['y_true'], L['y_sym40'], L['y_asym40']
NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
hB = json.load(open(f'{P}/out/coteach_sym40.json')); dB = np.load(f'{P}/out/coteach_sym40.npz')
l1, l2 = dB['losses'].astype(np.float32), dB['losses2'].astype(np.float32); noisyS = ys != yt
dA = np.load(f'{P}/out/ce_sym40.npz'); lA = dA['losses'].astype(np.float32)
seriesB = [{'e': x['epoch'], 'R': round(1 - x['forget_rate'], 4), 'p1': round(x['label_precision_1'], 4), 'p2': round(x['label_precision_2'], 4),
            'a1': round(x['test_acc_1'], 4), 'a2': round(x['test_acc_2'], 4)} for x in hB]
rng = np.random.default_rng(21)
clean_i, noisy_i = np.where(~noisyS)[0], np.where(noisyS)[0]
batch = np.concatenate([rng.choice(clean_i, 19, replace=False), rng.choice(noisy_i, 13, replace=False)]); rng.shuffle(batch)
EP = [2, 5, 10, 20, 50, 200]
batchB = {'idx': batch.tolist(), 'noisy': noisyS[batch].astype(int).tolist(), 'given': ys[batch].tolist(), 'true': yt[batch].tolist(),
          'epochs': EP, 'f': {str(e): [round(float(v), 3) for v in l1[e - 1][batch]] for e in EP},
          'g': {str(e): [round(float(v), 3) for v in l2[e - 1][batch]] for e in EP},
          'ce': {str(e): [round(float(v), 3) for v in lA[e - 1][batch]] for e in EP}}
json.dump({'series': seriesB, 'batch': batchB}, open(f'{P}/build/data/runB.json', 'w'), separators=(',', ':'))

p = np.load(f'{P}/out/cv_asym40.npz')['pred_probs'].astype(np.float64); n, m = p.shape; err = ya != yt
t = np.array([p[ya == j, j].mean() for j in range(m)]); above = p >= t[None, :]
C = np.zeros((m, m), int); flagged = np.zeros(n, bool); guess = np.full(n, -1)
for k in range(n):
    js = np.where(above[k])[0]
    if len(js) == 0: continue
    j = js[np.argmax(p[k, js])]; C[ya[k], j] += 1; guess[k] = j; flagged[k] = j != ya[k]
cnt = np.bincount(ya, minlength=m); Qn = C / C.sum(1, keepdims=True) * cnt[:, None]; Q = Qn / Qn.sum()
Qtrue = np.zeros((m, m)); np.add.at(Qtrue, (ya, yt), 1); Qtrue /= n
selfc = p[np.arange(n), ya]; pm = p.copy(); pm[np.arange(n), ya] = -1; margin = selfc - pm.max(1)
eps = 1e-12; CE = -np.log(selfc + eps); H = -(p * np.log(p + eps)).sum(1); Phi = CE - H
am = p.argmax(1) != ya
order_flag = np.where(flagged)[0][np.argsort(margin[flagged])]
CELLS = [(3, 5), (1, 9), (7, 4), (9, 1)]   # cat->dog, automobile->truck, horse->deer, truck->automobile (given -> likely true)
rc = np.random.default_rng(8); cells = []
for i, j in CELLS:
    s_ = np.where((ya == i) & (guess == j))[0]
    pick = rc.choice(s_, 8, replace=False)
    cells.append({'given': i, 'guess': j, 'n': int(len(s_)), 'prec': round(float(err[s_].mean()), 4),
                  'tiles': [{'i': int(k), 'true': int(yt[k]), 'err': int(err[k])} for k in pick]})
top = [t_['i'] for c in cells for t_ in c['tiles']]
prec_at = {k: round(float(err[order_flag[:k]].mean()), 4) for k in [100, 1000, 5000]}
orders = {'random': np.random.default_rng(0).permutation(n), 'margin': np.argsort(margin), 'phi': np.argsort(-Phi)}
grid = list(range(0, 20001, 500))
curves = {k: [0.0] + [round(float(np.cumsum(err[o])[b - 1] / err.sum()), 4) for b in grid[1:]] for k, o in orders.items()}
# per-class sample of probabilities for the threshold figure: 60 per given class
rs = np.random.default_rng(4); samp = {}
for j in range(m):
    ii = rs.choice(np.where(ya == j)[0], 60, replace=False)
    samp[j] = {'self': [round(float(v), 3) for v in p[ii, j]], 'err': err[ii].astype(int).tolist()}
runC = {'names': NAMES, 't': [round(float(v), 4) for v in t], 'C': C.tolist(), 'Q': np.round(Q, 5).tolist(), 'Qtrue': np.round(Qtrue, 5).tolist(),
        'counts': cnt.tolist(), 'n': int(n), 'nErr': int(err.sum()), 'estErr': int(round(n * (1 - np.trace(Q)))),
        'flags': {'cl': {'n': int(flagged.sum()), 'prec': round(float((err & flagged).sum() / flagged.sum()), 4), 'rec': round(float((err & flagged).sum() / err.sum()), 4)},
                  'argmax': {'n': int(am.sum()), 'prec': round(float((err & am).sum() / am.sum()), 4), 'rec': round(float((err & am).sum() / err.sum()), 4)}},
        'precAt': prec_at,
        'cells': cells,
        'review': {'grid': grid, 'curves': curves}, 'sample': samp,
        'oosAcc': round(float((p.argmax(1) == yt).mean()), 4)}
json.dump(runC, open(f'{P}/build/data/runC.json', 'w'), separators=(',', ':'))
open(f'{P}/build/data/tiles2_idx.txt', 'w').write(','.join(map(str, list(top) + list(batch))))

a1 = np.array([x['test_acc_1'] for x in hB]); a2 = np.array([x['test_acc_2'] for x in hB]); pr = np.array([x['label_precision_1'] for x in hB])
pc = lambda v: f'{100 * v:.1f}%'
cl = []
cl += [dict(id='run-b-last10', claim='Co-teaching (run B) mean test accuracy over the last 10 epochs, network f', value=f'{100*a1[-10:].mean():.2f}%'),
       dict(id='run-b-peak', claim='Co-teaching (run B) peak test accuracy, network f', value=f'{100*a1.max():.2f}%'),
       dict(id='run-b-prec-e1', claim='label precision of the kept set at epoch 1 in run B', value=pc(pr[0])),
       dict(id='run-b-prec-e10', claim='label precision of the kept set at epoch 10 in run B', value=pc(pr[9])),
       dict(id='run-b-prec-e200', claim='label precision of the kept set at epoch 200 in run B', value=pc(pr[199]))]
cl += [dict(id='run-c-noise-frac', claim='fraction of CIFAR-10 training labels flipped in our asymmetric-noise run', value=f'{100*err.mean():.2f}%'),
       dict(id='run-c-n-err', claim='number of flipped labels in run C', value=f'{int(err.sum()):,}'),
       dict(id='run-c-est-err', claim='number of label errors confident learning estimates in run C', value=f'{runC["estErr"]:,}'),
       dict(id='run-c-oos-acc', claim='accuracy of the 4-fold out-of-sample predictions against the true labels in run C', value=pc(runC['oosAcc'])),
       dict(id='run-c-cl-prec', claim='precision of the confident-joint flags in run C', value=pc(runC['flags']['cl']['prec'])),
       dict(id='run-c-cl-rec', claim='recall of the confident-joint flags in run C', value=pc(runC['flags']['cl']['rec'])),
       dict(id='run-c-am-prec', claim='precision of argmax-disagreement flags in run C', value=pc(runC['flags']['argmax']['prec'])),
       dict(id='run-c-am-rec', claim='recall of argmax-disagreement flags in run C', value=pc(runC['flags']['argmax']['rec'])),
       *[dict(id=f'run-c-cell-{NAMES[c["given"]]}-{NAMES[c["guess"]]}', claim=f'share of real errors among images labeled {NAMES[c["given"]]} that the confident joint puts under {NAMES[c["guess"]]}, run C', value=pc(c['prec'])) for c in cells],
       dict(id='run-c-prec-at-100', claim='share of real errors among the 100 lowest-margin flags in run C', value=pc(prec_at[100])),
       dict(id='run-c-review-5000-margin', claim='share of errors fixed after 5,000 reviews in margin order, run C', value=pc(curves['margin'][10])),
       dict(id='run-c-review-5000-random', claim='share of errors fixed after 5,000 reviews in random order, run C', value=pc(curves['random'][10])),
       dict(id='run-c-review-5000-phi', claim='share of errors fixed after 5,000 reviews in Phi order, run C', value=pc(curves['phi'][10]))]
for c in cl:
    run = 'b' if c['id'].startswith('run-b') else 'c'
    c.update(paper=f'our-run-{run}', version='job/run_noise.py', source='our-run',
             loc='out/coteach_sym40.json, out/coteach_sym40.npz (build/prep_runBC.py)' if run == 'b' else 'out/cv_asym40.npz, out/labels.npz (build/prep_runBC.py)')
json.dump(cl, open(f'{P}/notes/claims/our-run-bc.json', 'w'), indent=1)
for c in cl: print(c['id'], c['value'])
print('precAt', prec_at)
