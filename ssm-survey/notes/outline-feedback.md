# Outline feedback from extraction agents (apply when building each chapter)

## s4 (2026-10-09)
- S4 v3 prose keeps v1 numbers (80.48 avg, 88% Path-X, "within 0.8 ppl"); Tables 4/8 show 86.09, 96.35, 20.95. Table 10 has both rows. Pick one row and say which (original pairs with S4 Table 11; updated with S4D Table 12).
- S4 App C.1 calls Eq (2) "LegT"; it is LegS. Algorithm 1 vs Lemma C.2 opposite root-of-unity signs. S4D Eq (5) drops sign on n<k. S4D Sec 5.3 says only S4D-Inv beats chance on Path-X but Table 5 gives S4D-LegS 91.95.
- viz-s4flow: do not say GELU (S4 v3 never names it); say "nonlinearity", or GLU for the updated-LRA config.
- Cost readout: theory O~(N+L) (Theorem 3); code runs O(NL) time, O(N+L) memory (App A); naive O(N^2 L).
- Delta init log-uniform (0.001, 0.1), B formula, C init come from S4D; Delta trained is in S4 App D.2.
- Diagonal A works only with HiPPO-derived inits; random diagonal fails; S4D-Lin fails Path-X.
- ch 2: S4 discretizes bilinear, Mamba ZOH. Kernel formula is S4 Eq (5).
- ch 3: HiPPO B formula is in S4D Eq (4), not S4 v3.
- Speech 98.32 is SC10 (10-class subset); S4 App D.5 asks to cite S4D Table 11 (35 classes).
- ~25 claim values are ASCII math spellings (O~(N+L) etc.): page must use the same spelling or not wrap them.

## hippo + lssl (2026-10-09)
- Notation: HiPPO writes input f, state c, and x for a past time (not u/x). LSSL/S4 use u/x.
- LegS is time-varying (1/t factor); only LegT/LagT are LTI. LSSL/S4 keep LegS's A, drop 1/t, bring Delta back. Ch 2 must not use LegS as its constant-A example; LegS has no Delta knob (Prop. 3).
- Discretization: HiPPO and LSSL use bilinear; Mamba ZOH. viz-hippo must use bilinear (forward Euler at N=64: rmse 0.30 vs 0.013 bilinear).
- Sign trap: Theorem 2 Eq (3) writes -(1/t)A with positive A; App B.3 GBT recurrence drops the minus. With Theorem 2's A: c_{k+1} = (I + A/(2(k+1)))^-1 [(I - A/(2k)) c_k + B f_k / k]. Start-index convention in hippo.md E8.
- pMNIST 98.3 is a HiPPO-RNN (gated nonlinear RNN with fixed linear memory), not a linear SSM. First stacked-linear numbers: LSSL Table 1 (98.76; fixed-A 98.60).
- Three views come from LSSL Fig 1, Eq (6)-(7): cite LSSL in ch 4. LSSL Eq (6) misses a C; LSSL App B.2 uses positive A without minus.
- Bridge: LSSL Lemma 3.1 (gate = backward Euler) and HiPPO Sec 2.5 connect ch 1 to ch 6 (Mamba Theorem 1).

## hybrid: jamba, jamba15, nemotronh (2026-10-09)
- Jamba paper gives no d_model/heads/KV heads; KV back-solved from Table 1 (4GB @256K, 4 attention layers) = 16 KiB/token total = 4 KiB/token per attention layer = 1024 K/V channels/layer; exact only if GB means GiB; consistent with the "8x smaller" claim. Claims with our_reading:true.
- Jamba Table 1 "Llama" row is Llama-2 7B (6.7B), not 70B.
- viz-stack/viz-jambaflow: no config in sources; use back-solved per-layer KV or add HF config.json as code-only.
- Nemotron-H "8%" counts FFN layers; by sequence mixers its attention share is 4/28 (8B), 10/64 (56B), higher than Jamba 1:7. The slider must say what it divides by.
- Jamba-1.5 kept Mamba-1 (beat Mamba-2 inside the hybrid, Fig 1); Nemotron-H uses Mamba-2.
- Jamba-1.5 Table 1 LLaMA-3.1 405B row (252GB) is 2x what Llama 3 config gives (126 GiB): caveat or omit.
- Jamba has no throughput table: two "3x" ratios in text; Fig 3 values are read-offs.
- Jamba paper: 1:3 and 1:7 tie, 1:7 chosen for compute efficiency (Jamba-1.5 calls it "optimal"): quote Jamba.

## mamba + h3 + parscan (2026-10-09)
- 130M-2.8B config table is README only (code-only); paper Table 12 stops at 1.3B and gives Transformer layer counts.
- N=16 for released checkpoints is our reading. State bytes (our reading): 11.875 MiB at 2 bytes (SSM 16 + conv d_conv-1 = 3) = 95 tokens of Llama-3-8B KV. APPLIED in ch-bill.
- ch 6: A_log, D skip, dt_rank default, vocab are in neither paper nor README; fetch mamba_simple.py (code-only) or drop from the backward walkthrough. conv k=4 code-only.
- ch 5: "Table 2" is the induction-heads plot; exact accuracies in Table 11. H3 Table 2: S4D 35.6 vs 100.0 on Induction Head (cleanest LTI-fails-recall number).
- ch 0: "5x" is peak throughput at large batch (Fig 8: 1814 vs 364 tokens/s, prompt 2048); body text says "4-5x". APPLIED.
- ch 7: scan speedup "20-40x" in text, "40x" in Fig 8 caption. Memory comparable to a FlashAttention Transformer, not lower (Table 15).
- ch 7: Mamba never writes the scan operator; Martin & Cundy write (Lambda, x) . h = Lambda h + x; the pair form is our derivation. Their Table 1: parallel kernel 0.06x on 16-step sequences (pays off only when long).
- ch 6: add Table 10 (N 1->16 helps only with selective B, C: 9.73 -> 8.71).

## lru + griffin (2026-10-09)
- LRU reports no LRA average: show a mean only labelled computed. LRU uses colour sCIFAR; LRU's S4 row 91.1 is S4's sequential-CIFAR (91.13), not S4's LRA Image (88.65).
- LRU does not state the scan cost (only combine operator, "~8 times faster", steps/sec): take cost from parscan or S5.
- a^T readout: LRU gives no gradient formula. Show a^T as the forward impulse response; cite Bengio 1994 and Pascanu 2013 through LRU Sec 1.
- viz-recur: a real a cannot ring (decays, flips sign, explodes). Add a phase slider (Fig 4, complex lambda) and gain readout 1/(1-|lambda|^2) (Eq 6), which motivates gamma.
- "Dense A" (Table 2) = "RNN-Lin" (Table 1), not a separate step.
- LRU typos: Eq 4 sum should be j=0..k; Sec 3.4 lambda drops i; use App A form exp(-exp(nu_log) + i exp(theta_log)).
- ch 7: on TPU-v3 Griffin found the associative scan slower than a sequential linear scan.
- ch 11: Griffin is local-attention hybrid (2 recurrent : 1 local MQA, window 1024); keep apart from global-attention hybrids.
- Griffin token ratio: "over 6 times fewer" (2 places) and "roughly 7 times fewer" (2 places): quote each with its location.

## ssd: mamba2 + linattn (2026-10-09)
- M = L o (C B^T) has no equation number (Sec 5.1): cite Sec 5.1 with Eq (16) and Eq (3). Complexity table: "Sec 6.3 table". Linear attention Eqs (4), (6) numbered in PDF only.
- Notation box: linear attention N = sequence length; Mamba-2 N = state size, T = length. Linear attention state S_i/Z_i, later s_i/z_i.
- N default "around 64 or 128"; 256 only in MQAR/ablations. Conv width and released-model N not in paper.
- Mamba-2 block: Fig 6 vs Sec 8.1 disagree whether B, C pass through the conv: say so or avoid.
- Add Theorem 5.2 line: efficient autoregressive masked attention needs a semiseparable mask.
- Linear attention 317x / 4,462x baselines have no KV cache; vs KV-cache baseline CIFAR-10 gap is "more than 50x".
- ch 10: Mamba-2 N=256 beats attention on MQAR at small model dim (Fig 8): do NOT say SSMs always lose on recall.
- ch 11: Mamba-2 Table 2: 6 attention of 48 layers (~10%) best ppl; Table 3: +6 attention at 2.7B improves Pile ppl 6.09 -> 5.95.
- ch 12: Sec 10.4 describes RetNet as SSD with constant A_t.

## vision: vim, vmamba, mambaout, mlla (2026-10-09)
- Vim-S top-1: v3 Table 1 = 80.3 (README/VMamba/MambaOut cite 80.5). Use 80.3. VMamba-T ADE20K mIoU 47.9 main vs 48.0 App Table 8. lr claims spelled "1e-3".
- viz-vimflow: Vim-B D/E not in paper/README; use Vim-Ti (D=192, E=384) and Vim-S (D=384, E=768), L=24, N=16.
- "2.8x faster, 86.8% less memory" = Vim-Ti vs DeiT-Ti, backbone only, at 1248^2 (6084 tokens); with FPN head 2.8x and 73.2%. DeiT OOM at 1248.
- Class token middle: best in ablation (76.1 vs 75.2 head).
- VMamba final models use d_state = 1 (vs Mamba 16): fits ch 10.
- VMamba routes: row order, column order, both reversed, each its own SSM; merge not stated in text.
- VMamba "versions": paper never says v1-v4; Sec 4.3 speed-up steps (Table 5): 82.17% @426 img/s -> 82.60% @1686 img/s. MambaOut calls it "VMambaV9".
- Detection: Vim Cascade Mask R-CNN vs VMamba/MambaOut Mask R-CNN 1x: do not compare.
- MambaOut ties LocalVMamba-T 82.7; its "1.4/1.1 AP behind" is vs old VMamba-T row. Rule "long when L > 6D" (thresholds 2304, 4608 vs 196 ImageNet tokens, ~4K COCO/ADE20K): live readout candidate.
- MLLA: Mamba = single-head linear attention + input gate (Delta) + forget gate (A-bar) + skip (D), no normalization. Two credited factors: forget gate (77.6 -> 78.4, throughput 1152 -> 743 img/s) and block design (80.9). Position encodings (LePE 81.6, CPE 81.7) can replace the forget gate. Renamed MILA in v2.

## recall: zoology, repeat, nvidia, illusion, based (2026-10-09)
- Zoology's "d >= N" result covers gated convolutions (H3, Hyena, RWKV, BaseConv); Zoology never trains Mamba; its fixed-state recall result is about RetNet (App F Fig 6, Cor H.24). Do NOT write "Zoology shows SSMs need d >= N".
- "Phone number seen 10K tokens ago" has no source. Pure SSMs fail Phonebook past ~500 tokens (NVIDIA, 1.1T tokens) and past ~1000 (3.5T). Repeat After Me: gap opens at 70+ entries.
- Illusion of State is state tracking (TC0), not recall; the same figure shows Mamba beating transformers at approximate state tracking. Separate one-line card.
- ch 11: NVIDIA Fig 4 (lowest loss near 8% attention at 130M); 4 attention layers of 56 in the hybrid. "8x" speedup is predicted, not measured, at batch 32. Mamba-2-scan-vs-Mamba-scan 8x is a different claim.
- No paper gives a capacity formula for KV pairs in a state of size N. Closest: Repeat After Me Cor 2.8 (copying L tokens needs ~ L log2 D bits); Based Thm 3.1 / Cor F.13 and Zoology Cor H.24 (any fixed-state model needs Omega(N) bits for recall); Based App E.1 state-size formulas.
- Self-disagreements: Zoology 17 models vs 14 runs; 82% not in a table (own table averages 81.6). NVIDIA MMLU gap "nearly 15" vs 17; RULER lead 13 (prose) vs 11.86 (table). Based 10.36 not reproducible from Table 1; arxiv_main.tex is v1 text, main.tex matches v2 PDF.

## branches (2026-10-09)
- notes/family-tree.json: 40 nodes; four parent links are our reading (hyena, retnet, hymba, falconh1); xlstm and minrnn have no parent.
- Based abstract page says 6.22 points over Mamba; v2 PDF/LaTeX say 10.36; arxiv_main.tex 9.03. Headline 24x; 6.22 uncertain.
- Mamba-3 half-state claim: only Fig 3 (440M, d_state 16-128), worded three ways; 1.5B Table 3: SISO 56.4, GDN 55.8, MIMO 57.6.
- MambaOut-Tiny ties LocalVMamba-T at 82.7: "surpasses all" holds only at S and B.
- Mamba-3 intro pairs Mamba-2 and GDN as the production-hybrid layers: "what came next" is a pair.
