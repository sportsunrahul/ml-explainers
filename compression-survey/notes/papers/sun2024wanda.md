# sun2024wanda -- Wanda (BRANCH)

## Identity
- Title: "A Simple and Effective Pruning Approach for Large Language Models"
- arXiv 2306.11695, read v3. v1 2023-06-20; v3 (read) 2024-05-06 (abs.html).
- Authors: Mingjie Sun (first) ... J. Zico Kolter (last); with Zhuang Liu, Anna Bair. Venue: ICLR 2024 (abs.html).
- Source: `src/main.tex`, `src/appendix.tex`. Macros: `\mbf` = `\mathbf`; `\method` = Wanda.

## Key idea (two sentences)
Score each weight by its magnitude times the L2 norm of its input feature over calibration tokens, so a small
weight on a large-activation channel survives (the same activation-awareness as AWQ). Compare scores within each
output row and drop the lowest s%; there is no weight update, and the score needs one forward pass.

## Defining equation (`eq:prune`, PDF Eq. 1, Sec. 3), verbatim
```
\mbf{S}_{ij}=|\mbf{W}_{ij}|\cdot \|\mbf{X}_{j}\|_{2}
```
Gloss: W is (C_out, C_in); X is (N x L, C_in) tokens; ||X_j||_2 is the L2 norm of input channel j across all N x L
tokens. Comparison group (Eq. 2): all weights of output i, G_ij = {W_uv | u = i}.

## Setup facts
- Calibration: the same 128 sequences from C4 as SparseGPT (Sec. 4).
- Score cost (Table 4, seconds): LLaMA-7B Wanda 0.54 vs SparseGPT 203.1; 65B 5.6 vs 1353.4.

## Key row
- WikiText PPL, LLaMA-7B, unstructured 50% (Table 3): Dense 5.68, Magnitude 17.29, SparseGPT 7.22, Wanda 7.26.
  Same table LLaMA-65B 50%: dense 3.56, magnitude 5.90, SparseGPT 4.57, Wanda 4.57.
- 2:4 speed (Table 5; LLaMA-65B, batch 1, A6000, CUTLASS matmul, ms): q/k/v/o_proj 3.49 -> 2.14 1.63x;
  up/gate_proj 9.82 -> 6.10 1.61x; down_proj 9.92 -> 6.45 1.54x. End-to-end LLaMA-7B 1.24x (312ms -> 251ms) (Sec. 4).
- Vision: ConvNeXt-B and DeiT-B on ImageNet-1K, one-shot, 4096 calibration images (App.): the Wanda metric beats
  magnitude (App. Fig. 4) but only as plots, no table; per-output grouping did NOT help image classifiers
  (layer-wise was slightly better, App. Fig. 3).

## Outline feedback
- One line for ch-gptq-awq: "Wanda prunes with |W| x ||X|| and no weight update: LLaMA-7B 50% PPL 7.26 vs
  magnitude 17.29 (Table 3)". Mention it as the pruning twin of AWQ's activation-aware saliency.
- If the page mentions Wanda on vision models, say the per-output trick is LLM-specific (App. Fig. 3).
