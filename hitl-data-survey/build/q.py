import json,glob,sys,re
pat=re.compile(sys.argv[2],re.I); paper=sys.argv[1]
for f in glob.glob('notes/claims/*.json'):
    for c in json.load(open(f)):
        if (paper=='*' or c.get('paper')==paper) and (pat.search(c['id']) or pat.search(c['claim'])):
            print(('U ' if c.get('uncertain') else '  ')+c['id'],'|',c['value'][:70],'|',c['claim'][:70])
