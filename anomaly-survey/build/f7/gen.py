# Generates the data-bearing parts of ch-speed and the closing sections from notes/claims/*.json, so every number on
# them is copied from a claim (never typed). Run: python3 -I build/f7/gen.py   (from the project root)
# Templates: build/f7/ch-speed.tpl.html, build/f7/ch-closing.tpl.html. Outputs: parts/ch-speed.html, parts/ch-closing.html.
import glob, html, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'notes', 'claims', '*.json'))):
    for c in json.load(open(f)):
        C[c['id']] = c


def claim(i):
    if i not in C:
        sys.exit('missing claim: ' + i)
    if C[i].get('uncertain'):
        sys.exit('uncertain claim used: ' + i)
    return C[i]


def v(i):
    return claim(i)['value']


def num(i, scale=1.0):
    s = v(i).replace(',', '').replace('%', '').replace(' ms', '').strip()
    return float(s) * scale


def span(i, suffix=''):
    return '<span data-claim="%s">%s</span>%s' % (i, html.escape(v(i)), suffix)


# ---------- ch-speed ----------
GPUS = [('a6000', 'RTX A6000'), ('a5000', 'RTX A5000'), ('v100', 'Tesla V100'), ('3080', 'RTX 3080'), ('2080ti', 'RTX 2080 Ti')]
DESK = [('s', 'EfficientAD-S'), ('m', 'EfficientAD-M'), ('gcad', 'GCAD'), ('simplenet', 'SimpleNet'), ('fastflow', 'FastFlow'),
        ('dsr', 'DSR'), ('patchcore', 'PatchCore'), ('ast', 'AST'), ('st', 'S-T'), ('patchcore-ens', 'PatchCore-Ens')]
EAD = 'EfficientAD 2303.14535v3, Tables 1, 2 and 17 (float16, one image at a time; baselines re-run by the EfficientAD authors)'

points = []
for g, gname in GPUS:
    for k, name in DESK:
        points.append(dict(view='desktop', sub=g, key=k, name=name, device=gname, lat=f'efficientad-t17-{k}-{g}', latScale=1,
                           acc={'mvtec': f'efficientad-t2-{k}-mvtec', 'mean': f'efficientad-t1-{k}-auroc'}, accScale=1,
                           proto='one-class', src=EAD))

RC = '2605.24251v2 Table 4 (latency) and Table 1 (accuracy, continual protocol)'
UQ = '2407.02968v1 Table 5 (Jetson Xavier NX, 15-class model; "mean AUROC", image or pixel level not stated)'
J = [
    ('patchcore', 'PatchCore* (ResNet-18)', 'Jetson Orin Nano', 'rcad-orin-patchcore-ms', 'rcad-mvtec-img-patchcore', 100, 'continual', RC),
    ('efficientad', 'EfficientAD*', 'Jetson Orin Nano', 'rcad-orin-efficientad-ms', 'rcad-mvtec-img-efficientad', 100, 'continual', RC),
    ('dinosaur', 'DINOSaur (DINOv3 ViT-S/16)', 'Jetson Orin Nano', 'rcad-orin-dinosaur-ms', 'rcad-mvtec-img-dinosaur', 100, 'continual', RC),
    ('dne', 'DNE', 'Jetson Orin Nano', 'rcad-orin-dne-ms', 'rcad-mvtec-img-dne', 100, 'continual', RC),
    ('ucad', 'UCAD', 'Jetson Orin Nano', 'rcad-orin-ucad-ms', 'rcad-mvtec-img-ucad', 100, 'continual', RC),
    ('edgezsad', 'EdgeZSAD (TensorRT FP16)', 'Jetson Orin Nano Super', 'edgezsad-jetson-ms', 'edgezsad-mvtec-img', 1, 'zero-shot',
     'EdgeZSAD 2606.16119v1 Tables 4 and 5 (device-rescored)'),
    ('stfpm-fp32', 'STFPM FP32', 'Jetson Xavier NX', 'unifiedq-xnx-stfpm-fp32-ms', 'unifiedq-xnx-stfpm-fp32-auroc', 100, 'multi-class', UQ),
    ('stfpm-fp16', 'STFPM, student FP16', 'Jetson Xavier NX', 'unifiedq-xnx-stfpm-fp16-ms', 'unifiedq-xnx-stfpm-fp16-auroc', 100, 'multi-class', UQ),
    ('stfpm-int8', 'STFPM, student INT8', 'Jetson Xavier NX', 'unifiedq-xnx-stfpm-int8-ms', 'unifiedq-xnx-stfpm-int8-auroc', 100, 'multi-class', UQ),
    ('rd-fp32', 'RD FP32', 'Jetson Xavier NX', 'unifiedq-xnx-rd-fp32-ms', 'unifiedq-xnx-rd-fp32-auroc', 100, 'multi-class', UQ),
    ('rd-fp16', 'RD, student FP16', 'Jetson Xavier NX', 'unifiedq-xnx-rd-fp16-ms', 'unifiedq-xnx-rd-fp16-auroc', 100, 'multi-class', UQ),
    ('rd-int8', 'RD, student INT8', 'Jetson Xavier NX', 'unifiedq-xnx-rd-int8-ms', 'unifiedq-xnx-rd-int8-auroc', 100, 'multi-class', UQ),
    ('us-fp32', 'Uninformed Students FP32', 'Jetson Xavier NX', 'unifiedq-xnx-us-fp32-ms', 'unifiedq-xnx-us-fp32-auroc', 100, 'multi-class', UQ),
    ('us-fp16', 'Uninformed Students, student FP16', 'Jetson Xavier NX', 'unifiedq-xnx-us-fp16-ms', 'unifiedq-xnx-us-fp16-auroc', 100, 'multi-class', UQ),
]
for k, name, dev, lat, acc, sc, proto, src in J:
    points.append(dict(view='jetson', sub='all', key=k, name=name, device=dev, lat=lat, latScale=1, acc={'mvtec': acc}, accScale=sc, proto=proto, src=src))

PL = '2603.20288v1 Tables I and II (Intel Core i5-9600K)'
CPU = [
    ('pc', 'PatchCore', 'Core i5-9600K', 'pclite-patchcore-ms', 1, 'pclite-patchcore-mvtec-img', 100, 'one-class', PL),
    ('pc-edge', 'PatchCore-Edge', 'Core i5-9600K', 'pclite-patchcore-edge-ms', 1, 'pclite-patchcore-edge-mvtec-img', 100, 'one-class', PL),
    ('pc-lite', 'PatchCore-Lite', 'Core i5-9600K', 'pclite-patchcore-lite-ms', 1, 'pclite-patchcore-lite-mvtec-img', 100, 'one-class', PL),
    ('padim', 'PaDiM', 'Core i5-9600K', 'pclite-padim-ms', 1, 'pclite-padim-mvtec-img', 100, 'one-class', PL),
    ('padim-edge', 'PaDiM-Edge', 'Core i5-9600K', 'pclite-padim-edge-ms', 1, 'pclite-padim-edge-mvtec-img', 100, 'one-class', PL),
    ('padim-lite', 'PaDiM-Lite', 'Core i5-9600K', 'pclite-padim-lite-ms', 1, 'pclite-padim-lite-mvtec-img', 100, 'one-class', PL),
    ('stfpm', 'STFPM', 'Core i5-9600K', 'pclite-stfpm-ms', 1, 'pclite-stfpm-mvtec-img', 100, 'one-class', PL),
    ('stfpm-edge', 'STFPM-Edge', 'Core i5-9600K', 'pclite-stfpm-edge-ms', 1, 'pclite-stfpm-edge-mvtec-img', 100, 'one-class', PL),
    ('paste', 'PaSTe', 'Core i5-9600K', 'pclite-paste-ms', 1, 'pclite-paste-mvtec-img', 100, 'one-class', PL),
    ('padim-paper', 'PaDiM-WR50-Rd550', 'Core i7-4710HQ', 'padim-time-wr50-rd550', 1000, 'padim-mvtec-img-auroc-wr50-rd550', 1, 'one-class',
     'PaDiM 2011.08785v1 Table VII (time) and Table V (accuracy)'),
    ('spade-paper', 'SPADE', 'Core i7-4710HQ', 'padim-time-spade', 1000, 'padim-t5-spade-img-auroc', 1, 'one-class',
     'PaDiM 2011.08785v1 Table VII (time, PaDiM re-run) and Table V (accuracy as printed in PaDiM)'),
    ('pi-pc', 'PatchCore* (ResNet-18)', 'Raspberry Pi 5', 'rcad-pi5-patchcore-ms', 1, 'rcad-mvtec-img-patchcore', 100, 'continual', RC),
    ('pi-ead', 'EfficientAD*', 'Raspberry Pi 5', 'rcad-pi5-efficientad-ms', 1, 'rcad-mvtec-img-efficientad', 100, 'continual', RC),
    ('pi-dino', 'DINOSaur', 'Raspberry Pi 5', 'rcad-pi5-dinosaur-ms', 1, 'rcad-mvtec-img-dinosaur', 100, 'continual', RC),
]
for k, name, dev, lat, ls, acc, sc, proto, src in CPU:
    lab = name.replace(' (ResNet-18)', '') + (' Pi' if dev == 'Raspberry Pi 5' else '')
    points.append(dict(view='cpu', sub='all', key=k, name=name, label=lab, device=dev, lat=lat, latScale=ls, acc={'mvtec': acc}, accScale=sc, proto=proto, src=src))

RUG = [
    dict(view='jetson', name='KairosAD (no on-device accuracy; supervised)', device='Jetson NX', lat='kairosad-jetson-nx-ms',
         src='KairosAD 2505.24334v1 Sec. 4.4'),
    dict(view='jetson', name='KairosAD (no on-device accuracy; supervised)', device='Jetson AGX', lat='kairosad-jetson-agx-ms',
         src='KairosAD 2505.24334v1 Sec. 4.4'),
    dict(view='jetson', name='CLIP ViT-L/14 backbone alone', device='Jetson Orin Nano Super', lat='edgezsad-clipl-jetson-ms',
         src='EdgeZSAD 2606.16119v1 Table 4 (backbone-only probe)'),
    dict(view='jetson', name='DINOv2 ViT-L/14 backbone alone', device='Jetson Orin Nano Super', lat='edgezsad-dinov2l-jetson-ms',
         src='EdgeZSAD 2606.16119v1 Table 4 (backbone-only probe)'),
]

for p in points:
    p['latV'] = v(p['lat']); p['latMs'] = num(p['lat'], p['latScale'])
    p['accV'] = {m: v(i) for m, i in p['acc'].items()}; p['accPct'] = {m: num(i, p['accScale']) for m, i in p['acc'].items()}
for r in RUG:
    r['latV'] = v(r['lat']); r['latMs'] = num(r['lat'])


def lat_cell(p):
    return span(p['lat'], ' s' if p['latScale'] == 1000 else '')


rows = []
for view, sub in [('desktop', g) for g, _ in GPUS] + [('jetson', 'all'), ('cpu', 'all')]:
    body = []
    for p in [q for q in points if q['view'] == view and q['sub'] == sub]:
        accs = ' / '.join(span(i) for m, i in p['acc'].items())
        body.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class="src">%s</td></tr>' % (
            html.escape(p['name']), html.escape(p['device']), lat_cell(p), accs, p['proto'], html.escape(p['src'])))
    for r in [q for q in RUG if q['view'] == view]:
        body.append('<tr><td>%s</td><td>%s</td><td>%s</td><td>not reported for this run</td><td>latency only</td><td class="src">%s</td></tr>' % (
            html.escape(r['name']), html.escape(r['device']), span(r['lat']), html.escape(r['src'])))
    rows.append('<tbody data-view="%s" data-sub="%s"%s>\n%s\n</tbody>' % (view, sub, '' if (view, sub) == ('desktop', 'a6000') else ' hidden', '\n'.join(body)))

speed_json = json.dumps({'points': [{k: p[k] for k in ('view', 'sub', 'key', 'name', 'label', 'device', 'latMs', 'latV', 'accPct', 'accV', 'proto', 'src', 'latScale') if k in p} for p in points],
                         'rug': [{k: r[k] for k in ('view', 'name', 'device', 'latMs', 'latV', 'src')} for r in RUG]}, separators=(',', ':'))

tpl = open(os.path.join(ROOT, 'build', 'f7', 'ch-speed.tpl.html')).read()
out = tpl.replace('{{SPEED_ROWS}}', '\n'.join(rows)).replace('{{SPEED_JSON}}', speed_json.replace('</', '<\\/'))
# {{c:id}} -> claim span, {{v:id}} -> bare value (only inside data-claim'd context or captions that cite it)
import re
out = re.sub(r'\{\{c:([a-z0-9.\-]+)\}\}', lambda m: span(m.group(1)), out)
open(os.path.join(ROOT, 'parts', 'ch-speed.html'), 'w').write(out)

# ---------- scoreboard ----------
SB = [
    # name, year claim, setting, mvtec img, pix, pro, visa img, pix, pro ; each cell: claim id or ('r', id, note) for a re-run, or None
    ('SPADE', 'spade-published', 'one-class', 'spade-mvtec-img-auroc', 'spade-mvtec-pix-auroc', 'spade-mvtec-pro', None, None, None, 'spade'),
    ('PaDiM (WR50-Rd550)', 'padim-published', 'one-class', 'padim-mvtec-img-auroc-wr50-rd550', 'padim-mvtec-pix-auroc-wr50-rd550', 'padim-mvtec-pro-wr50-rd550',
     ('r', 'visa-padim-r50-visa-img-auroc', 'VisA paper re-run, ResNet-50, its 90/10 split'), ('r', 'visa-padim-r50-visa-pix-auroc', 'VisA paper re-run, ResNet-50'), None, 'padim'),
    ('PatchCore-25%', 'patchcore-published', 'one-class', 'patchcore-mvtec-img-auroc-25pct', 'patchcore-mvtec-pix-auroc-25pct', 'patchcore-mvtec-pro-25pct',
     ('r', 'visa-patchcore-wrn50-visa-img-auroc', 'VisA paper re-run, WRN-50, its 90/10 split'), ('r', 'visa-patchcore-wrn50-visa-pix-auroc', 'VisA paper re-run, WRN-50'),
     ('r', 'efficientad-t12-patchcore-visa-aupro', 'EfficientAD re-run'), 'patchcore'),
    ('FastFlow (CaiT)', 'fastflow-published', 'one-class', 'fastflow-mvtec-img-auroc-cait', 'fastflow-mvtec-pix-auroc-cait', None,
     ('r', 'efficientad-t2-fastflow-visa', 'EfficientAD re-run (anomalib)'), None, ('r', 'efficientad-t12-fastflow-visa-aupro', 'EfficientAD re-run (anomalib)'), 'fastflow'),
    ('CutPaste (3-way)', 'cutpaste-published', 'one-class, synthetic defects', 'cutpaste-mvtec-img-auroc-3way', 'cutpaste-mvtec-pix-auroc', None, None, None, None, 'cutpaste'),
    ('DRAEM', 'draem-published', 'one-class, synthetic defects', 'draem-mvtec-img-auroc', 'draem-mvtec-pix-auroc', None, None, None, None, 'draem'),
    ('EfficientAD-S', 'efficientad-published', 'one-class', 'efficientad-t2-s-mvtec', None, 'efficientad-t12-s-mvtec-aupro', 'efficientad-t2-s-visa', None, 'efficientad-t12-s-visa-aupro', 'efficientad'),
    ('EfficientAD-M', 'efficientad-published', 'one-class', 'efficientad-t2-m-mvtec', None, 'efficientad-t12-m-mvtec-aupro', 'efficientad-t2-m-visa', None, 'efficientad-t12-m-visa-aupro', 'efficientad'),
    ('UniAD', 'uniad-published', 'multi-class', 'uniad-img-uniad-unified', 'uniad-pix-uniad-unified', None, None, None, None, 'uniad'),
    ('Dinomaly', 'dinomaly-published', 'multi-class', 'dinomaly-mvtec-img-auroc', 'dinomaly-mvtec-pix-auroc', 'dinomaly-mvtec-aupro',
     'dinomaly-visa-img-auroc', 'dinomaly-visa-pix-auroc', 'dinomaly-visa-aupro', 'dinov2'),
    ('WinCLIP', 'winclip-published', 'zero-shot', 'winclip-mvtec-0shot-img-auroc', 'winclip-mvtec-0shot-pix-auroc', 'winclip-mvtec-0shot-pro',
     'winclip-visa-0shot-img-auroc', 'winclip-visa-0shot-pix-auroc', 'winclip-visa-0shot-pro', 'winclip'),
    ('AnomalyCLIP', 'anomalyclip-v1-published', 'zero-shot', 'anomalyclip-mvtec-img-auroc', 'anomalyclip-mvtec-pix-auroc', 'anomalyclip-mvtec-pro',
     'anomalyclip-visa-img-auroc', 'anomalyclip-visa-pix-auroc', 'anomalyclip-visa-pro', 'anomalyclip'),
    ('EdgeZSAD', 'edgezsad-published', 'zero-shot', 'edgezsad-mvtec-img', 'edgezsad-mvtec-pix', 'edgezsad-mvtec-pro', 'edgezsad-visa-img', 'edgezsad-visa-pix', 'edgezsad-visa-pro', 'speed'),
    ('WinCLIP+', 'winclip-published', '1-shot', 'winclip-plus-mvtec-1shot-img-auroc', 'winclip-plus-mvtec-1shot-pix-auroc', 'winclip-plus-mvtec-1shot-pro',
     'winclip-plus-visa-1shot-img-auroc', 'winclip-plus-visa-1shot-pix-auroc', 'winclip-plus-visa-1shot-pro', 'winclip'),
    ('AnomalyDINO-S (448)', 'anomalydino-published', '1-shot', 'anomalydino-mvtec-img-auroc-1shot-448', 'anomalydino-mvtec-pix-auroc-1shot-448', 'anomalydino-mvtec-pro-1shot-448',
     'anomalydino-visa-img-auroc-1shot-448', 'anomalydino-visa-pix-auroc-1shot-448', 'anomalydino-visa-pro-1shot-448', 'dinov2'),
]


def cell(x):
    if x is None:
        return '<td class="empty" data-sort="">not reported</td>'
    if isinstance(x, tuple):
        _, i, note = x
        return '<td class="rerun" data-sort="%s" title="%s">%s<sup>r</sup></td>' % (num(i), html.escape(note + ': ' + claim(i)['loc']), span(i))
    return '<td data-sort="%s" title="%s">%s</td>' % (num(x), html.escape(claim(x)['loc'] + (('; ' + claim(x)['protocol']) if claim(x).get('protocol') else '')), span(x))


sb_rows = []
for r in SB:
    name, yr, setting = r[0], r[1], r[2]
    cells = ''.join(cell(x) for x in r[3:9])
    sb_rows.append('<tr><th scope="row"><a href="#ch-%s">%s</a></th><td data-sort="%s">%s</td><td>%s</td>%s</tr>' % (
        r[9], html.escape(name), v(yr)[:4], v(yr)[:4], setting, cells))

# ---------- tree ----------
LANES = ['Benchmarks and tools', 'Feature memory', 'Normalizing flows', 'Synthetic defects', 'Student-teacher', 'Reconstruction and unified',
         'Vision-language', 'Edge deployment']
FAMILY_LANE = {'benchmark': 0, 'survey': 0, 'tools': 0, 'precursor/reconstruction': 5, 'precursor': 1, 'feature embedding': 1,
               'feature embedding precursor': 1, 'flow': 2, 'synthetic': 3, 'student-teacher': 4, 'unified': 5, 'zero/few-shot': 6,
               'zero-shot': 6, 'few-shot': 6, 'few-shot generalist': 6, 'few-shot unified': 6}
SPINE = [
    ('mvtec-ad', 'MVTec AD', '2019-06', 'mvtecad-venue', 0, 'Fifteen categories of objects and textures with pixel masks; train on good images only.', 'mvtecad-images-total', 'images', 'https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html', 'ch-benchmark'),
    ('visa', 'VisA', None, 'visa-published', 0, 'Twelve objects, often several per image and with complex structure; harder than MVTec AD.', 'visa-images-total', 'images', 'https://arxiv.org/abs/2207.14315v1', 'ch-benchmark'),
    ('anomalib', 'Anomalib', None, 'anomalib-published', 0, 'One open library that re-implements the main methods under one protocol.', 'anomalib-mvtec-img-patchcore', 'PatchCore image AUROC, reproduced', 'https://arxiv.org/abs/2202.08341v1', 'ch-speed'),
    ('ader', 'ADer', None, 'ader-published', 0, 'One library and protocol for multi-class methods.', 'ader-mvtec-img-auroc-patchcore-100e', 'PatchCore image AUROC, multi-class', 'https://arxiv.org/abs/2406.03262v6', 'ch-uniad'),
    ('mvtec-ad2', 'MVTec AD 2', None, 'mad2-published', 0, 'Harder scenes and lighting, and a stricter PRO limit.', 'mad2-best', 'best AU-PRO at FPR 0.30', 'https://arxiv.org/abs/2503.21622v1', 'ch-benchmark'),
    ('spade', 'SPADE', None, 'spade-published', 1, 'Pretrained features and nearest neighbors, first per image, then per pixel.', 'spade-mvtec-pix-auroc', 'MVTec AD pixel AUROC', 'https://arxiv.org/abs/2005.02357v3', 'ch-spade'),
    ('padim', 'PaDiM', None, 'padim-published', 1, 'One Gaussian per patch position, scored with Mahalanobis distance.', 'padim-mvtec-img-auroc-wr50-rd550', 'MVTec AD image AUROC', 'https://arxiv.org/abs/2011.08785v1', 'ch-padim'),
    ('patchcore', 'PatchCore', None, 'patchcore-published', 1, 'Keep every normal patch feature, thin them with a greedy coreset, score by nearest neighbor.', 'patchcore-mvtec-img-auroc-25pct', 'MVTec AD image AUROC', 'https://arxiv.org/abs/2106.08265v2', 'ch-patchcore'),
    ('anomalydino', 'AnomalyDINO', None, 'anomalydino-published', 1, 'PatchCore-style patch nearest neighbors on DINOv2 features, from a handful of good images.', 'anomalydino-mvtec-img-auroc-1shot-448', 'MVTec AD image AUROC, 1-shot', 'https://arxiv.org/abs/2405.14529v3', 'ch-dinov2'),
    ('fastflow', 'FastFlow', None, 'fastflow-published', 2, 'A 2D normalizing flow maps normal features to a Gaussian; low likelihood flags a defect.', 'fastflow-mvtec-img-auroc-cait', 'MVTec AD image AUROC', 'https://arxiv.org/abs/2111.07677v2', 'ch-fastflow'),
    ('cutpaste', 'CutPaste', None, 'cutpaste-published', 3, 'Cut and paste patches to make fake defects, and learn to tell them apart.', 'cutpaste-mvtec-img-auroc-3way', 'MVTec AD image AUROC', 'https://arxiv.org/abs/2104.04015v1', 'ch-synthetic'),
    ('draem', 'DRAEM', None, 'draem-published', 3, 'Paint Perlin-noise defects, reconstruct the clean image, and segment the difference.', 'draem-mvtec-img-auroc', 'MVTec AD image AUROC', 'https://arxiv.org/abs/2108.07610v2', 'ch-synthetic'),
    ('efficientad', 'EfficientAD', None, 'efficientad-published', 4, 'A tiny student imitates a teacher on good images; an autoencoder adds a check for logical defects.', 'efficientad-t1-s-latency', 'ms per image on an RTX A6000 (EfficientAD-S)', 'https://arxiv.org/abs/2303.14535v3', 'ch-efficientad'),
    ('uniad', 'UniAD', None, 'uniad-published', 5, 'One transformer reconstructs features for every class; masking stops it from copying defects.', 'uniad-img-uniad-unified', 'MVTec AD image AUROC, multi-class', 'https://arxiv.org/abs/2206.03687v3', 'ch-uniad'),
    ('dinomaly', 'Dinomaly', None, 'dinomaly-published', 5, 'A DINOv2 encoder and a deliberately noisy, loose decoder: one model for all classes.', 'dinomaly-mvtec-img-auroc', 'MVTec AD image AUROC, multi-class', 'https://arxiv.org/abs/2405.14325v5', 'ch-dinov2'),
    ('winclip', 'WinCLIP', None, 'winclip-published', 6, 'Score image windows against "damaged" and "flawless" prompts with CLIP; no training images.', 'winclip-mvtec-0shot-img-auroc', 'MVTec AD image AUROC, zero-shot', 'https://arxiv.org/abs/2303.14814v1', 'ch-winclip'),
    ('anomalyclip', 'AnomalyCLIP', None, 'anomalyclip-v1-published', 6, 'Learn object-agnostic "normal" and "damaged" prompts once, and transfer them to new products.', 'anomalyclip-mvtec-img-auroc', 'MVTec AD image AUROC, zero-shot', 'https://arxiv.org/abs/2310.18961v12', 'ch-anomalyclip'),
    ('unifiedq', 'Unified AD on Xavier NX', None, 'unifiedq-published', 7, 'Quantizes multi-class student-teacher models with TensorRT for a Jetson.', 'unifiedq-xnx-stfpm-fp16-ms', 'ms, STFPM with an FP16 student on Jetson Xavier NX', 'https://arxiv.org/abs/2407.02968v1', 'ch-speed'),
    ('paste', 'PaSTe', None, 'paste-published', 7, 'Student and teacher share their first layers, so a light backbone fits a tiny device.', 'paste-train-ram-reduction', '% less training RAM than STFPM', 'https://arxiv.org/abs/2410.11591v1', 'ch-speed'),
    ('kairosad', 'KairosAD', None, 'kairosad-published', 7, 'A MobileSAM encoder for supervised detection on embedded boards.', 'kairosad-jetson-nx-ms', 'ms on a Jetson NX', 'https://arxiv.org/abs/2505.24334v1', 'ch-speed'),
    ('pclite', 'PatchCore-Lite', None, 'pclite-published', 7, 'Product-quantized memory bank and diagonal-covariance PaDiM for edge CPUs.', 'pclite-padim-lite-ms', 'ms, PaDiM-Lite on a Core i5 CPU', 'https://arxiv.org/abs/2603.20288v1', 'ch-speed'),
    ('rcad', 'Continual AD on the edge', None, 'rcad-published-v1', 7, 'Benchmarks continual methods on a Jetson; a frozen DINOv3 coreset method beats them and runs faster than PatchCore.', 'rcad-orin-dinosaur-ms', 'ms on a Jetson Orin Nano', 'https://arxiv.org/abs/2605.24251v2', 'ch-speed'),
    ('edgezsad', 'EdgeZSAD', None, 'edgezsad-published', 7, 'A compact zero-shot detector exported with TensorRT for a Jetson.', 'edgezsad-jetson-ms', 'ms on a Jetson Orin Nano Super', 'https://arxiv.org/abs/2606.16119v1', 'ch-speed'),
]
EDGES = [('spade', 'padim'), ('padim', 'patchcore'), ('patchcore', 'anomalydino'), ('cutpaste', 'draem'), ('uniad', 'dinomaly'),
         ('winclip', 'anomalyclip'), ('mvtec-ad', 'visa'), ('visa', 'mvtec-ad2')]

nodes = []
for key, name, date, dclaim, lane, line, kc, klabel, url, ch in SPINE:
    d = date or v(dclaim)
    nodes.append(dict(key=key, name=name, date=d, dateClaim=dclaim, dateV=v(dclaim), lane=lane, spine=True, line=line,
                      kc=kc, kv=v(kc), klabel=klabel, url=url, ch=ch))

DISPLAY = {'hvq-trans': 'HVQ-Trans', 'diad': 'DiAD', 'mambaad': 'MambaAD', 'april-gan': 'APRIL-GAN', 'anomalygpt': 'AnomalyGPT',
           'musc': 'MuSc', 'inctrl': 'InCTRL', 'promptad': 'PromptAD', 'adaclip': 'AdaCLIP', 'univad': 'UniVAD', 'mvtec-3d': 'MVTec 3D-AD',
           'real-iad': 'Real-IAD', 'im-iad': 'IM-IAD', 'survey-liu': 'Survey (Liu et al.)', 'ast': 'AST'}
branch_count = 0
for bf in ['branches-a.json', 'branches-b.json']:
    p = os.path.join(ROOT, 'notes', bf)
    if not os.path.exists(p):
        continue
    for b in json.load(open(p)):
        fam = (b.get('family') or '').lower()
        lane = FAMILY_LANE.get(fam)
        if lane is None:
            lane = 6 if ('shot' in fam or 'clip' in fam) else 0
        k = b.get('key') or {}
        kid = k.get('claim_id')
        if not kid or kid not in C or C[kid].get('uncertain'):
            kid = None
        dclaim = None
        for cand in [b['name'] + '-published', b['name'] + '-published-v1', (kid.split('-')[0] + '-published') if kid else None]:
            if cand and cand in C and not C[cand].get('uncertain'):
                dclaim = cand
                break
        if any(n['key'] == b['name'] for n in nodes):
            continue
        nodes.append(dict(key=b['name'], name=b.get('display') or DISPLAY.get(b['name'], b['name']), date=b['date'], dateClaim=dclaim, dateV=v(dclaim) if dclaim else b['date'],
                          lane=lane, spine=False, line=b.get('one_line', ''), kc=kid, kv=v(kid) if kid else None, klabel=k.get('label', ''),
                          url='https://arxiv.org/abs/' + b['arxiv'], ch=b.get('parent_chapter') or ''))
        branch_count += 1

tree_json = json.dumps({'lanes': LANES, 'nodes': nodes, 'edges': EDGES}, separators=(',', ':'))
tree_list = []
for lane_i, lane in enumerate(LANES):
    items = sorted([n for n in nodes if n['lane'] == lane_i], key=lambda n: n['date'])
    li = []
    for n in items:
        num_part = (' %s %s.' % (span(n['kc']), html.escape(n['klabel']))) if n['kc'] else ''
        date_part = span(n['dateClaim']) if n['dateClaim'] else html.escape(n['date'])
        li.append('<li><a href="%s">%s</a> (%s)%s: %s%s</li>' % (n['url'], html.escape(n['name']), date_part, '' if n['spine'] else ', branch',
                                                                 html.escape(n['line']), num_part))
    tree_list.append('<h4>%s</h4><ul>%s</ul>' % (lane, ''.join(li)))

tpl = open(os.path.join(ROOT, 'build', 'f7', 'ch-closing.tpl.html')).read()
out = (tpl.replace('{{SCOREBOARD_ROWS}}', '\n'.join(sb_rows)).replace('{{TREE_JSON}}', tree_json.replace('</', '<\\/'))
          .replace('{{TREE_LIST}}', '\n'.join(tree_list)).replace('{{BRANCH_COUNT}}', str(branch_count)).replace('{{NODE_COUNT}}', str(len(nodes))))
out = re.sub(r'\{\{c:([a-z0-9.\-]+)\}\}', lambda m: span(m.group(1)), out)
left = re.findall(r'\{\{[^}]*\}\}', out)
if left:
    sys.exit('unfilled template markers: %s' % left[:5])
open(os.path.join(ROOT, 'parts', 'ch-closing.html'), 'w').write(out)
print('speed points', len(points), 'rug', len(RUG), '| scoreboard rows', len(SB), '| tree nodes', len(nodes), 'branches', branch_count)
