#!/bin/bash
# usage: fetch1.sh name id tier
set -u
P=~/paper-explainers/sam-survey
name=$1; id=$2; tier=$3

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
