#!/bin/bash
# usage: fetch.sh papers.tsv sources_dir
while IFS=$'\t' read -r name id tier; do
  d="$2/$name"; mkdir -p "$d/src"
  [ -s "$d/abs.html" ] || { curl -sSL -o "$d/abs.html" "https://arxiv.org/abs/$id"; sleep 3; }
  [ -s "$d/eprint.bin" ] || { curl -sSL -o "$d/eprint.bin" "https://arxiv.org/e-print/$id"; sleep 3; }
  [ -s "$d/paper.pdf" ] || { curl -sSL -o "$d/paper.pdf" "https://arxiv.org/pdf/$id"; sleep 3; }
  if [ -z "$(ls -A "$d/src")" ]; then
    if tar -tzf "$d/eprint.bin" >/dev/null 2>&1; then tar -xzf "$d/eprint.bin" -C "$d/src"
    elif gzip -t "$d/eprint.bin" 2>/dev/null; then gunzip -c "$d/eprint.bin" > "$d/src/main.tex"
    else echo "$name eprint-not-archive $(file -b "$d/eprint.bin" | cut -c1-40)"; fi
  fi
  echo "$name $(wc -c < "$d/abs.html") $(wc -c < "$d/eprint.bin") $(wc -c < "$d/paper.pdf") tex=$(find "$d/src" -name '*.tex' | wc -l)"
done < "$1"
echo FETCH-DONE
