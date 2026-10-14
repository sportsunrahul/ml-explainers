import sys, json, re, collections
root=sys.argv[1]
d=json.load(open(root+'/notes/arxiv-scan-2026-10-09.json'))
P=d['papers']
txt=lambda p:(p['title']+' '+p['abstract']).lower()
VIS=lambda p:'cs.CV' in p['cats'] or re.search(r'\bimage|\bvisual|\bvision|\bpixel|mvtec|visa\b',txt(p))
IND=lambda p:re.search(r'mvtec|\bvisa\b|industr|defect|manufactur|surface|inspection',txt(p))
VID=lambda p:re.search(r'\bvideo|surveillance',txt(p)) and not re.search(r'mvtec|\bvisa\b',txt(p))
S=[p for p in P if VIS(p) and IND(p) and not VID(p)]
FAM=[
 ('vlm','vision-language / zero-shot',r'\bclip\b|vision-language|vision language|zero-shot|zero shot|\bvlm|multimodal large|\bmllm|language model|prompt'),
 ('fewshot','few-shot',r'few-shot|few shot'),
 ('multiclass','multi-class / unified',r'multi-class|multiclass|unified'),
 ('synthetic','synthetic anomalies',r'synthe|cutpaste|pseudo-anomal|pseudo anomal|simulat'),
 ('memory','memory bank / kNN',r'memory bank|coreset|nearest neighbo|patchcore|padim'),
 ('flow','normalizing flow',r'normali[sz]ing flow'),
 ('st','student-teacher / distillation',r'distillation|student|teacher'),
 ('recon','reconstruction',r'reconstruct|autoencoder|auto-encoder'),
 ('diffusion','diffusion',r'diffusion'),
 ('3d','3D / point cloud / depth',r'point cloud|\b3d\b|rgb-d|depth'),
 ('logical','logical anomalies',r'logical'),
 ('edge','edge / real-time / latency',r'jetson|edge device|edge comput|embedded|real-time|real time|latency|lightweight|fps\b'),
]
years=[str(y) for y in range(2018,2027)]
byyear=collections.Counter(p['date'][:4] for p in S)
fy={}
for k,lab,rx in FAM:
    fy[k]=collections.Counter(p['date'][:4] for p in S if re.search(rx,txt(p)))
loc_base='arXiv API scan 2026-10-09 (notes/arxiv-scan-2026-10-09.json): 4 queries; industrial visual subset computed by build/mapclaims.py (rule in state.json arxiv_scan.subset_rule)'
claims=[{'id':'map-scan-date','claim':'arXiv scan date','value':'2026-10-09','paper':'arXiv scan','loc':loc_base},
 {'id':'map-unique','claim':'unique records across the 4 queries','value':f"{len(P):,}",'paper':'arXiv scan','loc':loc_base},
 {'id':'map-subset','claim':'records in the industrial visual subset','value':str(len(S)),'paper':'arXiv scan','loc':loc_base}]
for q in d['queries']:
    if q['start']==0: claims.append({'id':'map-q-'+q['name'],'claim':'totalResults for query '+q['query'],'value':str(q['totalResults']),'paper':'arXiv scan','loc':q['url']})
for y in years:
    claims.append({'id':f'map-year-{y}','claim':f'industrial visual subset papers first posted in {y}'+(' (to 2026-10-09)' if y=='2026' else ''),'value':str(byyear[y]),'paper':'arXiv scan','loc':loc_base+'; by v1 date'})
for k,lab,rx in FAM:
    claims.append({'id':f'map-fam-{k}','claim':f'industrial visual subset papers whose title or abstract matches the {lab} keywords (heuristic, overlapping)','value':str(sum(fy[k].values())),'paper':'arXiv scan','loc':loc_base+'; regex '+rx})
json.dump(claims,open(root+'/notes/claims/map.json','w'),indent=1)
data={'scanned':d['scanned'],'subset':len(S),'years':years,'total':[byyear[y] for y in years],
      'families':[{'key':k,'label':lab,'regex':rx,'counts':[fy[k][y] for y in years]} for k,lab,rx in FAM],
      'note':'heuristic keyword match on title+abstract; families overlap; 2026 is partial (to Oct 9)'}
json.dump(data,open(root+'/notes/map-data.json','w'),indent=1)
print(len(claims),'claims'); print(json.dumps(data['total'])); [print(f['key'],f['counts']) for f in data['families']]
