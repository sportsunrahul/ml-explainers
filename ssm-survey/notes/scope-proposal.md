# Checkpoint 0 proposal (2026-10-09)

## Spine (full read + extraction)
| paper | arXiv | first posted | role |
|---|---|---|---|
| HiPPO | 2008.07669v2 | 2020-08-17 | memory as online polynomial projection; the A matrix |
| S4 | 2111.00396v3 | 2021-10-31 | NPLR structure, convolution view, LRA |
| Mamba | 2312.00752v2 | 2023-12-01 | selection (input-dependent Delta, B, C), scan, block |
| Mamba-2 (SSD) | 2405.21060v1 | 2024-05-31 | semiseparable matrices, duality with linear attention, chunked algorithm |
| Vision Mamba (Vim) | 2401.09417v3 | 2024-01-17 | bidirectional scan over patches |
| VMamba | 2401.10166v4 | 2024-01-18 | SS2D cross-scan |
| Jamba | 2403.19887v2 | 2024-03-28 | attention:Mamba ratio, MoE, KV-cache savings |
| Linear attention (Transformers are RNNs) | 2006.16236v3 | 2020-06-29 | needed for SSD; not in the transformers page |
| LRU (Resurrecting RNNs) | 2303.06349v1 | 2023-03-11 | linear recurrences, why RNNs were dropped, diagonal complex eigenvalues |
| Zoology (MQAR) | 2312.04927v1 | 2023-12-08 | claim-card paper: recall gap (lighter read) |
| Repeat After Me | 2402.01032v2 | 2024-02-01 | claim-card paper: copying gap (lighter read) |
| Empirical Study of Mamba LMs (NVIDIA) | 2406.07887v1 | 2024-06-12 | claim-card paper: controlled 8B Mamba vs Transformer vs hybrid |

## Branches (abstract + one table, one line on the family tree)
LSSL 2110.13985v1; DSS 2203.14343v3; S4D 2206.11893v2; S5 2208.04933v3; How to Train Your HiPPO 2206.12037v2;
H3 2212.14052v3; Hyena 2302.10866v3; GLA 2312.06635v6; Based 2402.18668v2; Griffin 2402.19427v1;
Illusion of State 2404.08819v3; MambaOut 2405.07992v3; MLLA 2405.16605v2; Jamba-1.5 2408.12570v1;
Gated DeltaNet 2412.06464v3; Nemotron-H 2504.03624 (v4); Mamba-3 2603.15569 (v1, ICLR 2026);
Martin and Cundy parallel scan 1709.04057v2.
To verify when the arXiv limit clears: RetNet, RWKV, DeltaNet, xLSTM, minGRU, Samba, Zamba, Hymba, MambaVision, Falcon-H1.
Off-arXiv: Blelloch 1990 prefix-sum report (cite as a tech report).

## Map chapter
Computed from notes/arxiv-scan-2026-10-09.json: ti:Mamba; abs selective/structured SSM/SSD;
ti "state space model(s)" restricted to ML/CV/NLP/audio/signal categories. Monthly counts + title-keyword domain split
(heuristic). Hand-added: linear-attention / gated-RNN families the scan cannot see (RetNet, RWKV, GLA, DeltaNet, Griffin, xLSTM, LRU).

## Chapters (13, ~25 min if narrated)
0 The bill attention sends (hero: KV cache grows, SSM state does not)
1 Linear recurrences, and why RNNs were dropped
2 Continuous-time state spaces and discretization
3 HiPPO: memory as compression of the past
4 S4: structured A and the convolution view (delta: DSS/S4D, diagonal is enough)
5 Why time-invariance is not enough (selective copying, induction heads)
6 Mamba: input-dependent selection
7 The parallel scan and the hardware-aware kernel (+ Mamba block walkthrough)
8 Mamba-2: linear attention from scratch, then state-space duality
9 Vision: Vim, VMamba, and MambaOut's counterpoint
10 Where SSMs still lose: recall, copying, retrieval (claim cards)
11 Hybrids: Jamba, NVIDIA 8B hybrid, Nemotron-H
12 Map, family tree, what came next (Gated DeltaNet, Mamba-3), summary

## Trim path (~10 chapters, ~18 min)
Merge 5 into 6 (motivation becomes the opening of Mamba); fold 7 into 6 as a scan stepper; merge 11 and 12.

## 3D candidates
- Hero: the transformers page's KV-cache cubes (orange keys, green values) growing per token, beside a fixed SSM state block.
- Ch 2/6: a trajectory of an N=3 state driven by the input; in ch 6, Delta/B/C bend it per token (selection).
- Ch 3: HiPPO coefficients over time as a surface (time x polynomial degree), with the reconstruction below.
- Ch 8: the SSD matrix M = L o (C B^T) as a surface, so the chunk/block structure and the decay read.
