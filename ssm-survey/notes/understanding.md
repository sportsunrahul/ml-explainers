# Understanding: state-space models as an alternative to attention (2026-10-09)

Detail, equations with numbers, configs and locations are in `notes/papers/<paper>.md`; every number is in
`notes/claims/*.json` (merged to `notes/claims.json`). This file is the synthesis: the problem, the one idea per
paper as a change from its predecessor, and the limits. Corrections from extraction are in `notes/outline-feedback.md`.

## Core problem
Attention reads every past token for every new token: prompt cost grows with L^2 per head per layer, and the KV
cache grows with L (Llama 3 8B: 128 KiB per token at 16 bits, from its Table 3 config). A recurrent model keeps a
fixed-size state and pays a constant cost per token, but classic nonlinear RNNs train sequentially and their
signal and gradients vanish or explode over long spans (LRU Sec 1, citing Bengio 1994 and Pascanu 2013).
Transformers won (2017 on) because every position trains in parallel.

## Lens per paper
- Method/architecture lens (trace tensors, ask what is trainable): HiPPO, LSSL, S4, S4D, LRU, Mamba, Mamba-2,
  linear attention, Vim, VMamba, Jamba, Jamba-1.5, Nemotron-H, Griffin.
- Analysis lens (claim, measurement, falsifier): Zoology, Repeat After Me, NVIDIA empirical study, Illusion of State,
  Based, MambaOut, MLLA. LRU is method-led but its evidence is an ablation ladder.

## The line of ideas (each a delta from the one before)
1. Linear recurrence (LRU 2303.06349v1; Martin and Cundy 1709.04057v2). Drop the nonlinearity inside the
   recurrence (keep it between layers): h_t = lambda h_{t-1} + B x_t unrolls to a weighted sum and can be computed
   with a parallel scan. LRU: diagonal complex lambda, exponential parameterization, ring init, gamma normalization;
   the ablation ladder reaches S4-level LRA scores (LRU Tables 1-3; LRU reports no average). The gain
   1/(1-|lambda|^2) (LRU Eq. 6) motivates normalization.
2. Continuous-time view (LSSL 2110.13985v1; Mamba Eq. 1-4). h'(t) = A h(t) + B x(t), y = C h; discretize with step
   Delta. Three views of the same layer: continuous, recurrent, convolutional (LSSL Fig. 1). HiPPO/LSSL/S4 use the
   bilinear rule; Mamba uses zero-order hold (Abar = exp(Delta A)). Delta sets how far one step moves the state.
3. HiPPO (2008.07669v2). Choose A, B so the state c(t) holds the optimal projection of the whole input history onto
   Legendre polynomials (LegS: measure scaled to [0, t]); the optimal update is a linear ODE c' = -(1/t) A c +
   (1/t) B f with a fixed lower-triangular A (Theorem 2). LegS is time-varying (1/t); only LegT and LagT are LTI.
   Use bilinear discretization (Euler fails at large N). Notation: input f, state c. pMNIST 98.3 is a HiPPO-RNN.
4. S4 (2111.00396v3). Keep HiPPO's A, drop the 1/t, bring back Delta: an LTI SSM whose output is one long
   convolution with kernel K = (CB, CAB, ..., CA^(L-1) B) (S4 Eq. 5): train with FFT convolution, generate with
   the recurrence. Computing K for HiPPO's non-normal A is the hard part: write A as normal plus low rank (DPLR),
   evaluate the kernel's generating function at roots of unity with a Cauchy kernel and the Woodbury identity;
   O~(N+L) in theory, O(NL) time in code. LRA updated v3 row: 86.09 average, Path-X 96.35 (prose still quotes v1).
   S4D (2206.11893v2): a diagonal A works if initialized from HiPPO; random diagonal fails.
5. Why LTI is not enough (Mamba Sec 3.1; H3 2212.14052v3). A fixed kernel weighs position, not content: it cannot
   skip filler (selective copying) or retrieve a value by its key (induction heads, associative recall). H3 Table 2:
   S4D 35.6 vs attention 100.0 on induction head.
6. Mamba (2312.00752v2). Selection: Delta, B, C become functions of the current token (Algorithm 2; Delta via
   softplus, broadcast across channels). Large Delta writes the token and forgets the past; small Delta ignores it.
   Theorem 1: with N=1, A=-1, B=1 this is the gated RNN h_t = (1-g_t) h_{t-1} + g_t x_t. Cost: the convolution
   trick is gone. Fix: a hardware-aware selective scan (kernel fusion, parallel scan, recomputation; the expanded
   (B, L, D, N) state lives only in SRAM). Block: in_proj with E=2, causal conv (width 4, README), SiLU, SSM, gate,
   out_proj; N=16. Up to 5x generation throughput (peak, large batch, Fig. 8). Table 10: larger N helps only when B
   and C are selective. The 130M-2.8B sizes are in the paper; their layer counts and widths are README-only.
7. Linear attention (2006.16236v3) and Mamba-2 (2405.21060v1). Replace exp(q.k) by phi(q).phi(k): (Q K^T) V =
   Q (K^T V), a recurrence with a matrix state (S_i = S_{i-1} + phi(k_i) v_i^T). Mamba-2: with A a scalar times
   the identity, a selective SSM is y = M x with M = L o (C B^T), L the 1-semiseparable mask of cumulative decays
   (Sec 5.1): masked linear attention with a decay. State-space duality: the same M can be computed quadratically
   (attention-like) or linearly (scan); the chunked SSD algorithm mixes both and runs on matrix-multiply units, so
   N grows to around 64-128. Theorem 5.2: efficient autoregressive masked attention needs a semiseparable mask.
8. Vision (Vim 2401.09417v3; VMamba 2401.10166v4). Images have no single order: Vim scans patch sequences forward
   and backward (class token in the middle); VMamba's SS2D scans four routes. Vim-S 80.3 top-1 (v3); Vim-Ti vs
   DeiT-Ti at 1248 px backbone: 2.8x faster, 86.8% less memory. VMamba final models use d_state = 1. MambaOut
   (2405.07992v3): ImageNet classification (196 tokens) is not long enough to need the SSM ("long" when L > 6D);
   detection and segmentation still benefit. MLLA (2405.16605v2): vision Mamba ~ linear attention + input gate +
   forget gate + skip; the forget gate and the block design explain most of the gain.

## Limitations, and who fixed what
- Recall and retrieval (Zoology 2312.04927v1; Repeat After Me 2402.01032v2; NVIDIA 2406.07887v1; Based 2402.18668v2):
  a fixed state must compress what attention keeps verbatim. Zoology: most of the perplexity gap of gated
  convolutions to attention is associative recall (82% stated; its table averages 81.6); its d >= N result covers
  gated convolutions, its fixed-state result RetNet. Repeat After Me: transformers copy strings exponentially long
  in their heads; fixed-state models need state bits that grow with the string; a 410M transformer beats a 2.8B Mamba on phonebook
  lookup from 70 entries on (Fig 1c shows a gap earlier). NVIDIA (8B, same data): pure SSMs lag on MMLU 5-shot and fail Phonebook past about 500 (1.1T tokens)
  to 1000 (3.5T) tokens. Not absolute: Mamba-2 at N=256 beats attention on MQAR at small width.
- Expressivity (Illusion of State 2404.08819v3): SSMs and transformers are both in TC0; neither tracks state
  (permutation composition) exactly at fixed depth; Mamba does better at approximate tracking.
- Fix in practice: hybrids. NVIDIA's 8B hybrid (4 attention layers of 56) beats the Transformer on 12 standard
  tasks; lowest loss near 8% attention at 130M. Mamba-2 Tables 2-3: about 10% attention layers best. Jamba
  (2403.19887v2): 1 attention : 7 Mamba, MoE every other layer, 52B total / 12B active, 256K context, KV cache 8x
  smaller than Mixtral (Table 1). Jamba-1.5 kept Mamba-1; Nemotron-H uses Mamba-2. Griffin mixes linear recurrences
  with local attention.
- What came next (a pair, per Mamba-3's intro): Gated DeltaNet (2412.06464v3; gating + delta rule) and Mamba-3
  (2603.15569v1; exponential-trapezoidal discretization, complex states, MIMO).
