# Applies fact-check patches [{id, field, old, new, reason}] to notes/claims/*.json, only where the current value equals `old`.
import json, glob, sys
patches = json.load(open(sys.argv[1]))
files = {f: json.load(open(f)) for f in sorted(glob.glob('notes/claims/*.json'))}
where = {c['id']: (f, c) for f, cs in files.items() for c in cs}
done, skip = 0, []
for p in patches:
    hit = where.get(p['id'])
    if not hit: skip.append((p['id'], 'no such id')); continue
    f, c = hit; cur = c.get(p['field'])
    old = p.get('old')
    if old not in (None, '') and cur != old and not (cur is None and old in (False, 'false', 'absent', None)):
        skip.append((p['id'], f"field {p['field']} is {cur!r}, patch expects {old!r}")); continue
    c[p['field']] = p['new']; c.setdefault('factcheck', []).append(p['reason'][:300]); done += 1
for f, cs in files.items(): json.dump(cs, open(f, 'w'), indent=1)
print(done, 'applied;', len(skip), 'skipped')
for s in skip: print('  SKIP', *s)
