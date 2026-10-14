# Pull every number out of atlas/chapters.html prose and check it against notes/claims.json values.
import re,json,html,sys
P='/Users/rahulmishra/paper-explainers/sam-survey/'
vals={}
for c in json.load(open(P+'notes/claims.json')):
    vals.setdefault(c['value'],[]).append(c['id'])
t=re.sub(r'<[^>]+>',' ',open(P+'atlas/chapters.html').read()); t=html.unescape(t)
nums=sorted(set(re.findall(r'\d+(?:\.\d+)?(?:x\d+)?%?',t)))
for n in nums:
    hit=[v for v in vals if n==v or re.fullmatch(r'[^\d]*%s[^\d]*'%re.escape(n),v)]
    print(('OK   ' if hit else 'MISS ')+n, (vals[hit[0]][:2] if hit else ''))
