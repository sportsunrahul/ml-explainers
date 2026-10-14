set -u
fetch() { n=$1; id=$2; mkdir -p $n; cd $n
  [ -s abs.html ] || curl -sL -m 60 -o abs.html "https://arxiv.org/abs/$id"
  [ -s eprint ] || curl -sL -m 180 -o eprint "https://arxiv.org/e-print/$id"
  [ -s paper.pdf ] || curl -sL -m 180 -o paper.pdf "https://arxiv.org/pdf/$id"
  mkdir -p src; (file eprint | grep -q gzip && tar -xzf eprint -C src 2>/dev/null) || (file eprint | grep -q gzip && gunzip -c eprint > src/main.tex) || true
  echo "$n $(ls src | wc -l) files, pdf $(du -h paper.pdf|cut -f1)"; cd ..; sleep 2; }
fetch vit 2010.11929v2; fetch clip 2103.00020v1; fetch align 2102.05918v2; fetch siglip 2303.15343v4
fetch blip 2201.12086v2; fetch flamingo 2204.14198v2; fetch blip2 2301.12597v3; fetch llava 2304.08485v2
fetch llava15 2310.03744v2; fetch qwenvl 2308.12966v3; fetch florence2 2311.06242v1; fetch qwen2vl 2409.12191v2
fetch idefics2 2405.02246v1; fetch molmo 2409.17146v2; fetch chameleon 2405.09818v2; fetch qwen3vl 2511.21631v2
