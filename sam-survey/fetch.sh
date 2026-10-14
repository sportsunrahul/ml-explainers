# name id(with version) tier
set -u
P=~/paper-explainers/sam-survey
while read name id tier; do
  [ -z "$name" ] && continue; d=$P/sources/$name; mkdir -p $d/latex; echo "$id $tier" > $d/ID
  [ -s $d/abs.html ] || curl -sS -m 60 -L "https://arxiv.org/abs/$id" -o $d/abs.html
  [ -s $d/paper.pdf ] || curl -sS -m 180 -L "https://arxiv.org/pdf/$id" -o $d/paper.pdf
  if [ ! -s $d/eprint ]; then curl -sS -m 180 -L "https://arxiv.org/e-print/$id" -o $d/eprint; fi
  t=$(file -b $d/eprint)
  case "$t" in
    *gzip*) (cd $d/latex && (tar xzf ../eprint 2>/dev/null || gunzip -c ../eprint > main.tex));;
    *tar*) (cd $d/latex && tar xf ../eprint);;
    *PDF*) echo "pdf-only" > $d/latex/PDF_ONLY;;
  esac
  echo "$name $id $tier :: $(file -b $d/paper.pdf | cut -c1-20) :: $t :: $(ls $d/latex | wc -l) files"
  sleep 2
done <<'LIST'
sam 2304.02643v1 spine
sam2 2408.00714v2 spine
sam3 2511.16719v2 spine
sam3d 2511.16624v2 spine
fcn 1411.4038v2 prehistory
unet 1505.04597v1 prehistory
maskrcnn 1703.06870v3 prehistory
maskformer 2107.06278v2 prehistory
mask2former 2112.01527v3 prehistory
dios 1603.04042v1 prehistory
ritm 2102.06583v1 prehistory
simpleclick 2210.11006v3 prehistory
focalclick 2204.02574v2 branch
stm 1904.00607v2 prehistory
xmem 2207.07115v2 prehistory
cutie 2310.12982v2 branch
mae 2111.06377v3 prehistory
vitdet 2203.16527v2 prehistory
hiera 2306.00989v1 prehistory
mobilesam 2306.14289v2 branch
mobilesamv2 2312.09579v1 branch
efficientsam 2312.00863v1 branch
efficientvitsam 2402.05008v2 branch
efficientsam3 2511.15833v1 branch
edgetam 2501.07256v1 branch
hqsam 2306.01567v2 branch
medsam 2304.12306v3 branch
medsam2 2504.03600v1 branch
groundedsam 2401.14159v1 branch
groundingdino 2303.05499v5 branch
seem 2304.06718v4 branch
trackanything 2304.11968v2 branch
sam2long 2410.16268v3 branch
samurai 2411.11922v2 branch
industrialsam 2307.12674v1 inspection
samdefect 2311.10245v1 inspection
saa 2305.10724v1 inspection
kairosad 2505.24334v1 inspection
boxes2pixels 2604.11162v1 inspection
notperfect 2304.05750v3 inspection
samadapter 2304.09148v3 inspection
concealed 2304.06022v4 inspection
LIST
for r in segment-anything sam2 sam3 sam-3d-objects; do mkdir -p $P/sources/repos; curl -sS -m 60 -L "https://raw.githubusercontent.com/facebookresearch/$r/main/README.md" -o $P/sources/repos/$r.README.md; echo "repo $r $(wc -c < $P/sources/repos/$r.README.md)"; done
echo FETCH DONE
