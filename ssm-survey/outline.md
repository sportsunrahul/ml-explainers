# State-space models: the alternative to attention (outline)

Reader: basic deep learning only. Has the transformers page (attention, KV cache), not linear attention.
Frame: every chapter answers "what does this cost compared with attention, and what does it give up?"
Budget: 13 chapters + one unnumbered interlude; long page (no narration, cue ids kept) + atlas page.
Notation for the whole page (Mamba's): input x, hidden state h (size N), output y, matrices A, B, C,
step size Delta. HiPPO writes the input as f and the state as c; LSSL and S4 write u and x. The page translates once, in ch 3.

Color key (one meaning each, whole page): blue = input tokens x; yellow = the state h (the memory);
green = output y; red = cost (compute, memory, error); brown = attention and its KV cache (the baseline).

Story: attention's bill -> RNNs were cheap but untrainable -> make the recurrence linear -> make it
continuous and principled (HiPPO) -> make it fast (S4) -> it cannot choose what to remember -> selection
(Mamba) -> make selection fast (scan) -> selection is a kind of attention (SSD) -> images -> where it
still loses -> mix both (hybrids) -> the field now.

## Act I: the bill and the old fix

### 0. ch-bill: The bill attention sends
- Intuition: attention re-reads the whole past for every new token, so compute grows with L^2 and the
  KV cache grows with L; an SSM carries a fixed-size summary, so each token costs the same.
- Hero (3D, pe3): the transformers page's KV-cache block (layers x tokens, cubes) grows one column per
  generated token; beside it a fixed block for the SSM state that only changes color. Ends static. Replay.
- Visual `viz-bill`: a context-length slider (1K to 1M). Three readouts: attention score FLOPs (L^2),
  KV-cache bytes (Llama-3-8B-sized config, matching the transformers page: 128 KiB/token), SSM state bytes
  for a Mamba-2.8B-sized config (computed from its config: layers, d_model, E, N). Computed, not toy.
- Prediction: "at what length does the cache outgrow the model weights?"
- Sources: transformers page config; Mamba config table (model sizes). Claim: Mamba 5x generation
  throughput (Mamba abstract / Fig. 8).

### 1. ch-recurrence: Linear recurrences, and why RNNs were dropped
- Intuition: an RNN is the cheapest possible memory (fixed state, constant cost per token), but its
  nonlinear recurrence must run step by step and its gradient shrinks or explodes as a^T. Remove the
  nonlinearity from the recurrence and the whole sequence becomes a weighted sum you can compute in parallel.
- Visual `viz-recur`: scalar recurrence h_t = a h_{t-1} + b x_t. Slider a in [-1.2, 1.2]: the impulse
  response a^k decays, rings or explodes; the gradient-through-T readout shows a^T. A toggle "nonlinear
  (tanh)" vs "linear" shows the unrolled sum only exists for the linear one. Toy (exact arithmetic).
- Equation build: h_t = sum_k a^{t-k} b x_k.
- LRU delta (claim card style box): a linear diagonal recurrence with complex eigenvalues, exponential
  parameterization and normalization matches S4 on LRA (LRU Table). Sourced numbers.
- Why dropped: sequential training depth L, vanishing/exploding gradients; Transformers train every
  position in parallel. Cite the LRU intro framing, not folklore.

## Act II: principled memory

### 2. ch-continuous: Continuous-time state spaces and discretization
- Intuition: describe the memory as a continuous system h'(t) = A h(t) + B x(t); a sampled sequence is
  just this system read at steps of size Delta. Delta is the knob: small Delta = slow, long memory per
  step; large Delta = the state jumps to the new input.
- Visual `viz-state3d` (3D, pe3): an N=3 state trajectory driven by a 1-D input; A's eigenvalues set decay
  and rotation; a Delta slider re-samples the same continuous curve with fewer or more points (resolution
  invariance). Computed exactly (matrix exponential, ZOH).
- Equation build: Mamba Eq. 1 (continuous), Eq. 4 (ZOH), Eq. 2 (recurrence).

### 3. ch-hippo: HiPPO, memory as compression of the past
- Intuition: choose the state so it holds the best polynomial fit of the whole history; the update that
  keeps that fit optimal as time moves is a fixed linear ODE with a specific A. That A is why SSMs remember.
- Visual `viz-hippo` (3D surface + 2D below): HiPPO-LegS coefficients c_n(t) over time (axes: time,
  degree n, value) for a chosen signal; below, reconstruction of the history from c(t) vs the true signal;
  N slider (4 to 64) changes error. Computed from HiPPO-LegS (paper's A, B, discretization).
- Equation build: HiPPO-LegS A and B (HiPPO Eq. / Theorem number to extract).
- Sourced: pMNIST accuracy; timescale robustness claim.

### 4. ch-s4: S4, structured A and the convolution view
- Intuition: when A, B, C do not change over time, unrolling the recurrence gives one long convolution
  kernel K = (CB, CAB, CA^2B, ...), so training is one FFT convolution and generation is the recurrence.
  The hard part is computing K for HiPPO's A cheaply; S4 writes A as normal plus low-rank.
- Visual `viz-views`: the same toy SSM computed three ways (continuous, recurrent, convolutional) with
  identical outputs; the kernel plot; a cost readout (naive O(N^2 L) vs S4's near O(N + L)).
- Architecture walkthrough `viz-s4flow`: S4 layer forward (H channels, N state, kernel gen, FFT conv,
  D skip, GELU, output mixing) with shapes from the S4 config, then backward (A via NPLR params, B, C, Delta
  trainable).
- Delta: DSS and S4D show a diagonal A is enough (one paragraph, branch).
- Sourced: LRA table incl. Path-X (first above chance), speed vs Transformer.

## Act III: selection

### 5. ch-lti: Why time-invariance is not enough
- Intuition: a fixed kernel weighs position, not content; it cannot skip filler or look up a key it saw
  earlier. Selective copying and induction heads expose this.
- Visual `viz-selcopy`: colored tokens scattered among filler; an LTI kernel smears them, a content-aware
  gate picks them out. Prediction prompt before the reveal. Toy.
- Sourced: Mamba Table 1 (selective copying) and induction-head extrapolation (Table 2 / Fig.).

### 6. ch-mamba: Mamba, input-dependent selection
- Intuition: let each token set its own Delta, B and C. Large Delta: reset and store this token; small
  Delta: ignore it. Theorem 1: with N=1 this is exactly a gated RNN.
- Visual `viz-select3d` (3D, reuses viz-state3d): per-token Delta bends the trajectory; tokens marked
  important are written, filler is skipped. Toy.
- Equation build: Algorithm 2 (S6) lines; Theorem 1 gate equation.
- Architecture walkthrough `viz-mambaflow`: Mamba block forward (in_proj x2 expand E=2, conv1d k=4, SiLU,
  x_proj -> Delta, B, C, selective scan, gate, out_proj) with shapes from Mamba-130M..2.8B configs; backward
  (A_log, D, dt bias init, everything trainable).
- Sourced: scaling/perplexity, throughput, DNA/audio highlights.

### 7. ch-scan: The parallel scan and the hardware-aware kernel
- Intuition: selection breaks the convolution trick, but the recurrence is still linear, so an
  associative scan computes all states in log-depth; Mamba keeps the big (B, L, D, N) state in fast SRAM and
  recomputes it in the backward pass.
- Visual `viz-scan`: a scan-tree stepper over 8 or 16 tokens with the operator (a1,b1) o (a2,b2) =
  (a1 a2, a2 b1 + b2); work and depth readouts. Second panel: HBM vs SRAM traffic. Toy + sourced speed.

### Interlude (unnumbered, not narrated): one generated token, end to end
- Stepper: a token through a Mamba-2.8B-sized LM in recurrent mode vs a Transformer with a KV cache;
  per-layer state shapes and bytes, computed from configs; link back to each chapter's equation.

### 8. ch-ssd: Mamba-2, linear attention, and state-space duality
- Intuition: drop softmax and attention becomes (Q K^T) V = Q (K^T V), an RNN with a matrix state (linear
  attention, built from scratch here). Mamba-2 shows a selective SSM with scalar A is the same thing with a
  decay mask: y = (L o C B^T) x. One matrix, two algorithms: quadratic (attention-like) or linear (scan),
  and a chunked mix that uses tensor cores.
- Visual `viz-ssdmat` (3D surface): the T x T matrix M; buttons: softmax attention / linear attention /
  SSD decay; chunk overlay showing diagonal blocks (quadratic) and off-diagonal blocks (low rank, via
  states). Computed from random toy inputs.
- Equation build: linear attention (Katharopoulos Eq. numbers), SSD matrix form (Mamba-2 Eq.).
- Architecture walkthrough `viz-mamba2flow`: Mamba-2 block (parallel projections of X, B, C, Delta; heads;
  N 64-256; norm before out_proj).
- Sourced: speedup vs Mamba scan, state sizes, MQAR.

## Act IV: beyond text, and honesty

### 9. ch-vision: Vision Mamba, VMamba, and MambaOut's counterpoint
- Intuition: an image has no single order; flatten patches into sequences and scan more than one way
  (Vim: forward + backward; VMamba: four cross-scan routes) so every patch can see every other.
- Visual `viz-scan2d`: a patch grid; choose Vim (2 routes) or VMamba (4 routes); click a patch to see how
  much each other patch can influence it after decay. Toy decay; sourced accuracy/memory numbers.
- Architecture walkthrough `viz-vimflow`: Vim forward (patch 16, cls token in the middle, D, depth) with
  shapes from Vim configs.
- MambaOut box: ImageNet classification does not need the SSM; detection/segmentation (long sequences) still
  benefit. MLLA one line: Mamba in vision behaves like linear attention with a forget gate.

### 10. ch-recall: Where SSMs still lose (claim cards)
- Intuition: a fixed state is lossy compression. Exact recall of many key-value pairs, copying long
  strings, and looking up a phone number seen 10K tokens ago need memory that grows with the context.
- Visual `viz-phonebook`: store K key-value pairs in a state of size N (outer-product write, linear
  attention style) vs a cache that keeps everything; retrieval accuracy as K passes N. Computed toy.
- Claim cards: Zoology (MQAR, share of perplexity gap from recall); Repeat After Me (copying, phonebook);
  NVIDIA 8B study (MMLU and phonebook gaps for pure SSMs at matched data); Illusion of State (one line).

### 11. ch-hybrid: Hybrids, a little attention goes a long way
- Intuition: keep a few attention layers for lookup and let SSM layers do the rest; most of the KV cache
  disappears and recall comes back.
- Visual `viz-stack`: a layer-stack builder; attention fraction slider; KV-cache memory at a chosen
  context computed from the Jamba config, compared with Jamba's reported numbers (sourced points).
- Architecture walkthrough `viz-jambaflow`: Jamba block (8 layers, a:m = 1:7, MoE every other layer,
  16 experts top-2), shapes from Jamba config.
- Sourced: Jamba KV-cache and throughput table; NVIDIA hybrid vs Transformer; Nemotron-H layer mix.

### 12. ch-map: The field now
- Map figure `viz-map`: monthly counts from notes/arxiv-scan-2026-10-09.json, deep-learning queries shown
  apart from the title-SSM query; title-keyword domain split labelled heuristic; every count a data-claim
  with the query as loc.
- Family tree `viz-tree`: spine + branches by year and lineage (HiPPO -> LSSL -> S4 -> DSS/S4D/S5 -> H3 ->
  Mamba -> Mamba-2 -> Mamba-3; linear attention -> RetNet/GLA/DeltaNet -> Gated DeltaNet; LRU/Griffin;
  vision; hybrids).
- What came next: Gated DeltaNet, Mamba-3 (claims from abstracts).

Then: summary section, "where to go deeper" (paper, section, equation per item), sources.

## Atlas page (9 stages, ~10 min; each figure written once as a scene, mounted in both pages)
1 bill (hero cubes) / 2 recurrence / 3 continuous (state3d) / 4 HiPPO surface / 5 S4 views /
6 selection + scan / 7 SSD matrix / 8 vision scans / 9 recall vs hybrids (phonebook + stack).

## Revisions after extraction (2026-10-09; details in notes/outline-feedback.md)
- ch-bill: Mamba 2.8B state = 11.9 MiB (SSM 16 + last 3 conv inputs), about 95 tokens of Llama 3 8B cache; its
  config is README-only (code-only); "5x" is peak throughput at large batch.
- ch-recurrence: a real eigenvalue only decays, flips or explodes; add a phase slider for complex lambda and the
  gain readout 1/(1-|lambda|^2) (LRU Eq. 6); a^T shown as the impulse response; RNN history cited through LRU Sec 1.
- ch-continuous: the constant-A example must be LTI (not HiPPO-LegS, which has a 1/t factor); one sentence on
  bilinear (HiPPO, LSSL, S4) vs zero-order hold (Mamba).
- ch-hippo: computed with the bilinear update (Euler fails); pMNIST 98.3 is a HiPPO-RNN, not a linear SSM; first
  stacked linear-SSM numbers from LSSL Table 1. HiPPO notation is f (input) and c (state).
- ch-s4: the three views are LSSL's (Fig. 1); LRA shown as one labelled row (updated v3 row: 86.09 average,
  Path-X 96.35); cost O~(N+L) in theory, O(NL) time in the code; "nonlinearity", not GELU; diagonal works only
  with HiPPO-derived inits (S4D).
- ch-lti: H3 Table 2 (S4D 35.6 vs 100.0 on induction head) and Mamba Table 11 accuracies.
- ch-mamba: Mamba Table 10 (state size helps only when B and C are selective); backward pass names only what the
  paper or README states.
- ch-scan: operator from Martin and Cundy; speedup "20-40x" (text) vs "40x" (Fig. 8 caption); memory comparable to
  FlashAttention, not lower; Griffin found the scan slower than a sequential loop on TPU-v3.
- ch-ssd: notation box (N means length in linear attention, state size in Mamba-2); N "around 64 or 128";
  Theorem 5.2; linear attention's 317x / 4,462x baselines have no KV cache.
- ch-vision: Vim-S is 80.3 in v3 (80.5 elsewhere); 2.8x / 86.8% is Vim-Ti vs DeiT-Ti backbone at 1248 px;
  VMamba's final d_state = 1; MambaOut's "long when L > 6D" rule as a live readout; MLLA's two factors.
- ch-recall: Mamba-2 at N=256 beats attention on MQAR at small model width, so the chapter must not say SSMs
  always lose; it loses when state is small relative to what must be recalled.
- ch-hybrid: Jamba's KV cache back-solved from its Table 1 (no width or head counts published); its "Llama" row is
  Llama-2 7B; Nemotron-H's 8% counts FFN layers (by sequence mixers 4 of 28 and 10 of 64); Griffin is a local-
  attention hybrid; Mamba-2 Tables 2-3 (about 10% attention layers is best).
