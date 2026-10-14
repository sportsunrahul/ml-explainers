import json, sys, time, urllib.parse, urllib.request, re
out = sys.argv[1]
base = "https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=1"
queries = {
  "title_contrastive": 'ti:contrastive',
  "abs_contrastive_learning": 'abs:"contrastive learning"',
  "title_contrastive_graph": 'ti:contrastive AND ti:graph',
  "title_contrastive_sentence_text": 'ti:contrastive AND (ti:sentence OR ti:text OR ti:language)',
  "title_contrastive_image_visual": 'ti:contrastive AND (ti:image OR ti:visual OR ti:vision)',
  "title_contrastive_medical": 'ti:contrastive AND (ti:medical OR ti:clinical OR ti:radiology)',
  "title_contrastive_video_audio": 'ti:contrastive AND (ti:video OR ti:audio OR ti:speech)',
  "title_contrastive_recommendation": 'ti:contrastive AND (ti:recommendation OR ti:recommender)',
  "title_contrastive_anomaly": 'ti:contrastive AND ti:anomaly',
  "title_contrastive_supervised": 'ti:contrastive AND ti:supervised',
}
rows = []
for name, q in queries.items():
  for y in range(2017, 2027):
    full = f'({q}) AND submittedDate:[{y}01010000 TO {y}12312359]'
    url = base.format(q=urllib.parse.quote(full))
    txt = urllib.request.urlopen(url, timeout=60).read().decode()
    m = re.search(r"<opensearch:totalResults[^>]*>(\d+)<", txt)
    assert m, txt[:300]
    rows.append({"query": name, "year": y, "count": int(m.group(1)), "url": url})
    print(name, y, m.group(1), flush=True)
    time.sleep(3)
json.dump({"date": "2026-10-13", "note": "2026 partial through 2026-10-13; title-keyword split is a heuristic", "rows": rows}, open(out, "w"), indent=1)
