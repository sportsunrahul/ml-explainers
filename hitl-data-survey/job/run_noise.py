# CIFAR-10 label-noise runs for the explainer figures. Runs on the pod GPU.
#   A "ce":      standard cross-entropy, 40% symmetric noise; per-example train loss every epoch
#   B "coteach": Co-teaching (Han et al. 2018) on the same noisy labels
#   C "cv":      4-fold cross-validated predicted probabilities, asymmetric noise (input for confident learning)
# Model, optimizer and schedules follow github.com/bhanML/Co-teaching (model.py, main.py, loss.py).
import json, os, pickle, sys, tarfile, time, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('OUT_DIR', os.path.join(ROOT, 'out')); os.makedirs(OUT, exist_ok=True)
DATA = '/root/work/paper-explainer/data'
dev = 'cuda'
PROG = os.path.join(ROOT, 'job', 'progress.json')

def progress(**kw):
    json.dump(kw, open(PROG, 'w'))

def load_cifar():
    d = os.path.join(DATA, 'cifar-10-batches-py')
    if not os.path.isdir(d):
        os.makedirs(DATA, exist_ok=True)
        tgz = os.path.join(DATA, 'cifar-10-python.tar.gz')
        urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', tgz)
        tarfile.open(tgz).extractall(DATA)
    def rd(f):
        b = pickle.load(open(os.path.join(d, f), 'rb'), encoding='bytes')
        return b[b'data'].reshape(-1, 3, 32, 32), np.array(b[b'labels'])
    xs, ys = zip(*[rd('data_batch_%d' % i) for i in range(1, 6)])
    xt, yt = rd('test_batch')
    return np.concatenate(xs), np.concatenate(ys), xt, yt

class CNN(nn.Module):  # Co-teaching model.py, top_bn=False
    def __init__(self, n_out=10, p=0.25):
        super().__init__()
        ch = [(3, 128, 1), (128, 128, 1), (128, 128, 1), (128, 256, 1), (256, 256, 1), (256, 256, 1), (256, 512, 0), (512, 256, 0), (256, 128, 0)]
        self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, 1, pd) for a, b, pd in ch])
        self.bns = nn.ModuleList([nn.BatchNorm2d(b) for _, b, _ in ch])
        self.fc = nn.Linear(128, n_out); self.p = p
    def forward(self, h):
        for i, (c, bn) in enumerate(zip(self.convs, self.bns)):
            h = F.leaky_relu(bn(c(h)), 0.01)
            if i in (2, 5):
                h = F.dropout2d(F.max_pool2d(h, 2, 2), p=self.p, training=self.training)
        h = F.avg_pool2d(h, h.shape[2]).flatten(1)
        return self.fc(h)

def plans(n_epoch=200, decay_start=80, lr=1e-3):
    a = [lr] * n_epoch; b = [0.9] * n_epoch
    for i in range(decay_start, n_epoch):
        a[i] = (n_epoch - i) / (n_epoch - decay_start) * lr; b[i] = 0.1
    return a, b

def set_opt(opt, lr, b1):
    for g in opt.param_groups: g['lr'] = lr; g['betas'] = (b1, 0.999)

def augment(x):  # random crop (pad 4) + horizontal flip, per sample
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (B,), device=x.device); ox = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    cols = (ox[:, None] + ar)[:, None, None, :].expand(B, 3, 32, 32)
    xp = xp.gather(2, (oy[:, None] + ar)[:, None, :, None].expand(B, 3, 32, 40)).gather(3, cols)
    flip = torch.rand(B, device=x.device) < 0.5
    xp[flip] = xp[flip].flip(3)
    return xp

@torch.no_grad()
def evaluate(model, X, Y=None, bs=2000):
    model.eval(); L, P, Pr = [], [], []
    for i in range(0, len(X), bs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lo = model(X[i:i + bs]).float()
        Pr.append(F.softmax(lo, 1)); P.append(lo.argmax(1))
        if Y is not None: L.append(F.cross_entropy(lo, Y[i:i + bs], reduction='none'))
    model.train()
    return (torch.cat(L) if L else None), torch.cat(P), torch.cat(Pr)

def sym_noise(y, rate, rng, C=10):  # Co-teaching 'symmetric': flip to another class uniformly with prob rate
    y2 = y.copy(); flip = rng.random(len(y)) < rate
    off = rng.integers(1, C, size=flip.sum())
    y2[flip] = (y[flip] + off) % C
    return y2

ASYM = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3}  # truck->automobile, bird->airplane, deer->horse, cat<->dog (Patrini et al. 2017)
def asym_noise(y, rate, rng):
    y2 = y.copy()
    for s, t in ASYM.items():
        idx = np.where(y == s)[0]; m = rng.random(len(idx)) < rate
        y2[idx[m]] = t
    return y2

def main():
    torch.backends.cudnn.benchmark = True
    xtr, ytr, xte, yte = load_cifar()
    X = torch.tensor(xtr, device=dev).float().div(255).contiguous(memory_format=torch.channels_last)
    Xt = torch.tensor(xte, device=dev).float().div(255).contiguous(memory_format=torch.channels_last)
    Yt = torch.tensor(yte, device=dev)
    rng = np.random.default_rng(0)
    y_sym = sym_noise(ytr, 0.4, rng)
    y_asym = asym_noise(ytr, 0.4, np.random.default_rng(1))
    np.savez(os.path.join(OUT, 'labels.npz'), y_true=ytr, y_sym40=y_sym, y_asym40=y_asym, y_test=yte)
    print('sym actual noise', (y_sym != ytr).mean(), 'asym actual noise', (y_asym != ytr).mean(), flush=True)
    N = len(ytr); E = int(os.environ.get('EPOCHS', 200)); bs = 128; nb = N // bs
    runs = sys.argv[1:] or ['ce', 'coteach', 'cv']
    total = E * (('ce' in runs) + ('coteach' in runs)) + 4 * int(os.environ.get('EPOCHS_CV', 60)) * ('cv' in runs); done = 0
    a_plan, b_plan = plans(E, min(80, E // 2))

    if 'ce' in runs:
        torch.manual_seed(1)
        Y = torch.tensor(y_sym, device=dev)
        m = CNN().to(dev).to(memory_format=torch.channels_last); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        losses = np.zeros((E, N), np.float16); preds = np.zeros((E, N), np.uint8); hist = []
        for ep in range(E):
            set_opt(opt, a_plan[ep], b_plan[ep]); t0 = time.time()
            perm = torch.randperm(N, device=dev)
            for b in range(nb):
                idx = perm[b * bs:(b + 1) * bs]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(X[idx]), Y[idx])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            L, P, _ = evaluate(m, X, Y); _, Pt, _ = evaluate(m, Xt)
            losses[ep] = L.cpu().numpy(); preds[ep] = P.cpu().numpy().astype(np.uint8)
            hist.append({'epoch': ep + 1, 'test_acc': (Pt == Yt).float().mean().item(), 'sec': time.time() - t0})
            done += 1; progress(phase='ce', done=done, total=total, finished=False, last=hist[-1])
            print('ce', hist[-1], flush=True)
        np.savez_compressed(os.path.join(OUT, 'ce_sym40.npz'), losses=losses, preds=preds)
        json.dump(hist, open(os.path.join(OUT, 'ce_sym40.json'), 'w'))

    if 'coteach' in runs:
        torch.manual_seed(2)
        Y = torch.tensor(y_sym, device=dev); clean = torch.tensor(y_sym == ytr, device=dev)
        rate = np.ones(E) * 0.4; rate[:min(10, E)] = np.linspace(0, 0.4, 10)[:min(10, E)]
        m1 = CNN().to(dev).to(memory_format=torch.channels_last); m2 = CNN().to(dev).to(memory_format=torch.channels_last)
        o1 = torch.optim.Adam(m1.parameters(), lr=1e-3); o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
        losses = np.zeros((E, N), np.float16); losses2 = np.zeros((E, N), np.float16); hist = []
        for ep in range(E):
            set_opt(o1, a_plan[ep], b_plan[ep]); set_opt(o2, a_plan[ep], b_plan[ep]); t0 = time.time()
            perm = torch.randperm(N, device=dev); pr1 = pr2 = 0.0
            keep = int((1 - rate[ep]) * bs)
            for b in range(nb):
                idx = perm[b * bs:(b + 1) * bs]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    l1 = F.cross_entropy(m1(X[idx]), Y[idx], reduction='none')
                    l2 = F.cross_entropy(m2(X[idx]), Y[idx], reduction='none')
                s1 = torch.argsort(l1.detach().float())[:keep]; s2 = torch.argsort(l2.detach().float())[:keep]
                pr1 += clean[idx[s1]].float().mean().item(); pr2 += clean[idx[s2]].float().mean().item()
                loss1 = l1[s2].sum() / keep; loss2 = l2[s1].sum() / keep  # each net learns from its peer's small-loss picks
                o1.zero_grad(set_to_none=True); o2.zero_grad(set_to_none=True)
                (loss1 + loss2).backward(); o1.step(); o2.step()
            L, _, _ = evaluate(m1, X, Y); L2, _, _ = evaluate(m2, X, Y); _, P1, _ = evaluate(m1, Xt); _, P2, _ = evaluate(m2, Xt)
            losses[ep] = L.cpu().numpy(); losses2[ep] = L2.cpu().numpy()
            hist.append({'epoch': ep + 1, 'forget_rate': float(rate[ep]), 'label_precision_1': pr1 / nb, 'label_precision_2': pr2 / nb,
                         'test_acc_1': (P1 == Yt).float().mean().item(), 'test_acc_2': (P2 == Yt).float().mean().item(), 'sec': time.time() - t0})
            done += 1; progress(phase='coteach', done=done, total=total, finished=False, last=hist[-1])
            print('coteach', hist[-1], flush=True)
        np.savez_compressed(os.path.join(OUT, 'coteach_sym40.npz'), losses=losses, losses2=losses2)
        json.dump(hist, open(os.path.join(OUT, 'coteach_sym40.json'), 'w'))

    if 'cv' in runs:
        # out-of-sample probabilities: each fold's model never saw the labels it predicts (Northcutt et al., Sec. on cross-validation)
        Y = torch.tensor(y_asym, device=dev); K = 4; Ecv = int(os.environ.get('EPOCHS_CV', 60))
        folds = np.random.default_rng(3).permutation(N) % K
        probs = np.zeros((N, 10), np.float32); a2, b2 = plans(Ecv, Ecv // 2); hist = []
        for k in range(K):
            torch.manual_seed(10 + k)
            tr = torch.tensor(np.where(folds != k)[0], device=dev); ho = np.where(folds == k)[0]
            m = CNN().to(dev).to(memory_format=torch.channels_last); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for ep in range(Ecv):
                set_opt(opt, a2[ep], b2[ep])
                perm = tr[torch.randperm(len(tr), device=dev)]
                for b in range(len(tr) // bs):
                    idx = perm[b * bs:(b + 1) * bs]
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = F.cross_entropy(m(augment(X[idx])), Y[idx])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                done += 1; progress(phase='cv', fold=k, done=done, total=total, finished=False)
            _, _, pr = evaluate(m, X[torch.tensor(ho, device=dev)])
            probs[ho] = pr.cpu().numpy()
            _, Pt, _ = evaluate(m, Xt)
            hist.append({'fold': k, 'test_acc': (Pt == Yt).float().mean().item(),
                         'heldout_acc_vs_true': float((probs[ho].argmax(1) == ytr[ho]).mean()),
                         'heldout_acc_vs_given': float((probs[ho].argmax(1) == y_asym[ho]).mean())})
            print('cv', hist[-1], flush=True)
        np.savez_compressed(os.path.join(OUT, 'cv_asym40.npz'), pred_probs=probs, folds=folds)
        json.dump(hist, open(os.path.join(OUT, 'cv_asym40.json'), 'w'))

    progress(phase='done', done=done, total=total, finished=True)
    print('RUN_DONE', flush=True)

if __name__ == '__main__':
    main()
