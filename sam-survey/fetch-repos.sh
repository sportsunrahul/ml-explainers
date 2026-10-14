P=~/paper-explainers/sam-survey
for r in segment-anything sam2 sam3 sam-3d-objects; do mkdir -p $P/sources/repos; curl -sS -m 60 -L "https://raw.githubusercontent.com/facebookresearch/$r/main/README.md" -o $P/sources/repos/$r.README.md; echo "repo $r $(wc -c < $P/sources/repos/$r.README.md)"; done

