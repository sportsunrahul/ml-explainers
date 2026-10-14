# Monthly counts and a title-keyword split from the arXiv scan (heuristic), plus claims for every count shown.
import json, re, collections
out, claims = {}, []
KW = {'active-learning': [('uncertainty / Bayesian', r'uncertain|bayes|entropy|dropout|bald'), ('diversity / core-set', r'divers|core-?set|coverage|cluster|represent'),
                          ('detection / segmentation', r'detect|segment'), ('medical', r'medic|clinic|patholog|radiol|mri|ct\b'), ('language / LLM', r'language|text|nlp|llm|token')],
      'label-noise': [('sample selection', r'select|small-loss|co-teach|clean sample|filter'), ('robust loss / correction', r'robust loss|loss correct|transition|loss function'),
                      ('semi-supervised', r'semi-supervised|pseudo'), ('detecting label errors', r'label error|mislabel|detect.*noisy|identif'), ('medical', r'medic|clinic|patholog|radiol')]}
for name in ['active-learning', 'label-noise']:
    d = json.load(open(f'notes/scan/{name}.json')); recs = d['records']
    months = collections.Counter(r['published'][:7] for r in recs)
    keys = [f'{y}-{m:02d}' for y in range(2015, 2027) for m in range(1, 13) if f'{y}-{m:02d}' <= '2026-10']
    yrs = collections.Counter(r['published'][:4] for r in recs)
    kw = [(lab, sum(1 for r in recs if re.search(rx, r['title'], re.I))) for lab, rx in KW[name]]
    out[name] = {'n': len(recs), 'months': keys, 'counts': [months.get(k, 0) for k in keys], 'before2015': sum(v for k, v in months.items() if k < '2015'), 'kw': kw,
                 'y2020': yrs['2020'], 'y2025': yrs['2025'], 'y2026': yrs['2026'], 'query': d['query'], 'fetched': d['fetched']}
    loc = f"arXiv API query fetched {d['fetched']}: {d['query']} (notes/scan/{name}.json)"
    claims += [dict(id=f'scan-{name}-n', claim=f'records returned by the {name} query', value=f'{len(recs):,}'),
               dict(id=f'scan-{name}-2020', claim=f'{name} records submitted in 2020', value=str(yrs['2020'])),
               dict(id=f'scan-{name}-2025', claim=f'{name} records submitted in 2025', value=str(yrs['2025'])),
               dict(id=f'scan-{name}-2026', claim=f'{name} records submitted in 2026 up to the scan date', value=str(yrs['2026']))]
    for c in claims[-4:]: c.update(paper='arxiv-scan', version=d['fetched'], loc=loc, source='scan')
a = set(r['id'].split('v')[0] for r in json.load(open('notes/scan/active-learning.json'))['records']); b = set(r['id'].split('v')[0] for r in json.load(open('notes/scan/label-noise.json'))['records'])
out['overlap'] = len(a & b)
claims.append(dict(id='scan-overlap', claim='records in both scans', value=str(len(a & b)), paper='arxiv-scan', version='2026-10-09', loc='intersection of notes/scan/*.json by arXiv id', source='scan'))
json.dump(out, open('build/data/scan.json', 'w'), separators=(',', ':')); json.dump(claims, open('notes/claims/scan.json', 'w'), indent=1)
for c in claims: print(c['id'], c['value'])
for n in ['active-learning', 'label-noise']: print(n, out[n]['kw'])
