# Every [data-claim] span must contain its claim value verbatim; no uncertain claim may be used as fact.
import re,json,html,sys
P='/Users/rahulmishra/paper-explainers/sam-survey/'
C={c['id']:c for c in json.load(open(P+'notes/claims.json'))}
src=open(P+(sys.argv[1] if len(sys.argv)>1 else 'explainer.src.html')).read()
bad=0
for m in re.finditer(r'<(span|td|b|i)[^>]*data-claim="([^"]+)"[^>]*>(.*?)</\1>',src,re.S):
    i,txt=m.group(2),html.unescape(re.sub(r'<[^>]+>','',m.group(3)))
    c=C.get(i)
    if not c: print('MISSING',i); bad+=1; continue
    v=html.unescape(c['value'])
    norm=lambda s: s.replace('×','x').replace('&amp;','&')
    if norm(v) not in norm(txt): print('VALUE',i,'| page:',txt[:60],'| claim:',v); bad+=1
    if c.get('uncertain'): print('UNCERTAIN USED',i,txt[:50]); bad+=1
print('spans checked',len(re.findall(r'data-claim=',src)),'problems',bad)
