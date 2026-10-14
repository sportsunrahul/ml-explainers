# OpenCLIP scaling laws: Reproducible scaling laws for contrastive language-image learning

## Identity
- arXiv 2212.07143, version read: v2 (v1 Wed 14 Dec 2022; v2 Sat 13 Jul 2024). Mehdi Cherti ... Jenia Jitsev (9 authors, LAION et al.).
- Venue: CVPR 2023 (abs.html comments: "CVPR 2023. Version with minor extension").
- Repo: https://github.com/LAION-AI/scaling-laws-openclip

## Key idea
Train CLIP (OpenCLIP, InfoNCE) on public LAION-400M/2B across model size, data size and samples seen, and fit power
laws to downstream error. The training distribution changes the scaling exponent: OpenAI CLIP (WIT) and OpenCLIP
(LAION) scale differently with identical architectures.

## Defining equation
Power law fit on the Pareto frontier: `E = \beta C^{\alpha}` (main.tex Sec. 4.1; E = error, C = compute).

## Key numbers
- ViT-H/14 on LAION-2B, 34B samples seen: zero-shot ImageNet 78.0 (Table 1; text "78%" Sec. 4.1). Same table: CLIP L/14 WIT-400M 75.5, OpenCLIP L/14 LAION-2B 75.2. 224 px.
- Scaling exponent zero-shot ImageNet: alpha_openCLIP = -0.11, alpha_CLIP = -0.16 (Sec. 4.1).
