set -u
cd "$(dirname "$0")/sources"
fetch() { n=$1; id=$2; mkdir -p $n; cd $n
  for t in 1 2 3; do [ -s abs.html ] && grep -q citation_title abs.html && break; curl -sL -m 60 -o abs.html "https://arxiv.org/abs/$id"; sleep 3; done
  for t in 1 2 3; do [ -s eprint ] && ! grep -qi '<html' eprint 2>/dev/null && break; curl -sL -m 240 -o eprint "https://arxiv.org/e-print/$id"; sleep 3; done
  for t in 1 2 3; do [ -s paper.pdf ] && head -c 4 paper.pdf | grep -q '%PDF' && break; curl -sL -m 240 -o paper.pdf "https://arxiv.org/pdf/$id"; sleep 3; done
  mkdir -p src; if [ -z "$(ls src)" ]; then (file eprint | grep -q 'gzip' && (tar -xzf eprint -C src 2>/dev/null || gunzip -c eprint > src/main.tex)) || (file eprint | grep -q 'tar archive' && tar -xf eprint -C src) || true; fi
  echo "$n $id src=$(ls src | wc -l | tr -d ' ') tex=$(find src -name '*.tex' | wc -l | tr -d ' ') pdf=$(du -h paper.pdf|cut -f1)"; cd ..; sleep 3; }
readme() { n=$1; url=$2; mkdir -p $n; [ -s $n/README.md ] || curl -sL -m 60 -o $n/README.md "$url"; echo "$n README $(wc -c < $n/README.md)"; }
fetch vild 2104.13921v3; fetch owlvit 2205.06230v2; fetch owlv2 2306.09683v3; fetch glip 2112.03857v2
fetch gdino 2303.05499v5; fetch yoloworld 2401.17270v3
fetch rf100vl 2505.20612v4; fetch fsod-vlm 2312.14494v4; fetch ovdeval 2308.13177v2; fetch d3 2307.12813v2
fetch trex2 2403.14610v1; fetch regionclip 2112.09106v1; fetch detic 2201.02605v3; fetch ovrcnn 2011.10678v2
fetch mdetr 2104.12763v2; fetch glipv2 2206.05836v2; fetch mmgdino 2401.02361v2; fetch gdino15 2405.10300v2
fetch ovdino 2407.07844v2; fetch dinox 2411.14347v3; fetch llmdet 2501.18954v1; fetch yoloe 2503.07465v2
readme vild https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/detection/projects/vild/README.md
readme owlvit https://raw.githubusercontent.com/google-research/scenic/main/scenic/projects/owl_vit/README.md
readme glip https://raw.githubusercontent.com/microsoft/GLIP/main/README.md
readme gdino https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/README.md
readme yoloworld https://raw.githubusercontent.com/AILab-CVC/YOLO-World/master/README.md
readme regionclip https://raw.githubusercontent.com/microsoft/RegionCLIP/main/README.md
readme detic https://raw.githubusercontent.com/facebookresearch/Detic/main/README.md
readme trex2 https://raw.githubusercontent.com/IDEA-Research/T-Rex/main/README.md
readme rf100vl https://raw.githubusercontent.com/roboflow/rf100-vl/main/README.md
readme mmgdino https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/mm_grounding_dino/README.md
echo FETCH_DONE
