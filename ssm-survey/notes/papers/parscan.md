# Martin and Cundy, parallel linear recurrence (1709.04057v2) -- branch read

## 1. Identity
- Title: "Parallelizing Linear Recurrent Neural Nets Over Sequence Length".
- arXiv 1709.04057, read v2. v1 2017-09-12; v2 2018-02-22 (abs.html submission history).
- Authors: Eric Martin and Chris Cundy (UC Berkeley).
- Venue: abs Comments "Published as a conference paper at ICLR 2018".
- Code: github.com/eamartin/parallelizing_linear_rnns (Sec 2.3). No README in `sources/parscan/`.
- Notation: x_t is the (already transformed) input added at step t, h_t the state, Lambda_t / lambda_t the
  per-step decay. In Mamba terms (our reading): lambda_t = A_bar_t, x_t = B_bar_t x_t, so h_t = A_bar_t h_{t-1} + B_bar_t x_t.

## 2. Core problem
Nonlinear RNNs (LSTM, GRU) must run step by step over the sequence, so training is slow on long sequences and
needs large minibatches for GPU use (Sec 1).

## 3. What prior methods did and why that hurt
Minibatching turns matrix-vector into matrix-matrix work but costs memory, latency and generalization (Sec 1).
Linear-recurrence RNNs (QRNN, SRU, strongly typed RNNs) were still evaluated serially over time (Sec 1, Sec 4.1.2).

## 4. The one key idea
A first-order linear recurrence h_t = Lambda_t h_{t-1} + x_t is a scan, so Blelloch's parallel scan computes all
h_t in O(T/p + lg p) steps when the operators meet his three conditions. Keep Lambda_t diagonal so combining two
steps costs O(n), not O(n^3).

## 5. Mechanism (Sec 2, Algorithm 1 "Parallel linear recurrence on p processors")
1. Split the T steps into p contiguous chunks.
2. In parallel per chunk: P_i = product of the chunk's Lambdas; R_i = chunk's recurrence run from h = 0.
3. Serial scan over the p chunk summaries: C_i = P_i C_{i-1} + R_i, with C_{-1} = h_0.
4. In parallel per chunk: rerun the scan from the chunk's true start state C_{i-1}.
5. Backward (Sec 2.2) is the same kind of linear recurrence, run in reverse, with lambda_{t+1} as the decay;
   so both passes parallelize over length.
6. Implementation (Sec 2.3): CUDA kernel, each warp is a processor (p up to 3200); 32 lanes cover 32 vector
   elements; peak only for sequences of "at least several thousand steps" on "at least a 32 element vector".

## 6. Equations
Scan definition (Sec 1, unnumbered; macro `\scan` = `\text{SCAN}`):
```tex
\scan(\oplus, [a_1, a_2, ..., a_n], b) = [(a_1 \oplus b), (a_2 \oplus a_1 \oplus b), ..., (a_n \oplus a_{n-1} ... \oplus a_1 \oplus b)].
```
Blelloch's form and conditions (Sec 1): `h_t = (\Lambda_t \otimes h_{t-1}) \oplus x_t` parallelizes if (1) oplus is
associative, (2) otimes is semiassociative (`a \otimes (b \otimes c) = (a \odot b) \otimes c`), (3) otimes distributes
over oplus. With vector add, matrix-vector and matrix-matrix products this gives `h_t = \Lambda_t h_{t-1} + x_t`.
Algorithm 1 operator (line 2): `$(\Lambda, x) \bullet h = \Lambda h + x$`; line 9 comment:
`compute $C_i = P_i C_{i-1} + R_i$ with $C_{-1}=h_0$`.
Pair composition for the page, (a1, b1) o (a2, b2) = (a1 a2, a2 b1 + b2): not written in this form in the paper;
it is what applying step 1 then step 2 of the bullet operator gives (our reading; matches P_i, R_i in Alg. 1).
Costs (Sec 2.1), verbatim:
```tex
C_\text{sscan} \in \mathcal{O}((C_\otimes + C_\oplus)T)
C_\text{pscan} \in \mathcal{O}(2(C_\odot + C_\otimes + C_\oplus)(T/p + \lg p))
```
Diagonal case: `C_\text{pscan}\in \mathcal{O}(6n(T/p + \lg p))`, `C_\text{sscan} \in \mathcal{O}(2nT)`, speedup factor
written as `pT/3(T+\lg p)`; "Assuming p << T, then C_pscan <= C_sscan when p >= 3". Dense case has an O(n^3)
matrix-matrix cost that "can counter-act any parallel speedups". Gloss (our reading): work grows by about 3x (6n vs 2n),
depth drops from T to T/p + lg p.
Backprop (Sec 2.2): `\nabla_{h_t}L = \lambda_{t+1} \odot \nabla_{h_{t+1}} L + \frac{\partial L}{\partial h_t}`,
`\nabla_{\lambda_t}L = h_{t-1} \odot \nabla_{h_t}L`.

## 7. Setup
- Kernel benchmark (Table 1): minibatch 1, m = 4, 32, 128 features, lengths 16 to 65,536; parallel vs serial
  CUDA kernel, measured at kernel level.
- Model benchmark (Table 2): two stacked layers, 256 hidden units, input size 4, bT = 65,536 fixed.
- Synthetic (Table 3): Hochreiter-Schmidhuber problem 2b, p = 128, n = 1,024 / 8,192 / 1,048,576; two-layer
  GILR-LSTM vs CuDNN LSTM, 512 hidden (64 at the longest length); NVIDIA K80 (Sec 4.2).

## 8. Results worth remembering
- Table 1 (kernel speedup, minibatch 1): length 65,536: 38.5 (m=4), 41.8 (m=32), 17.5 (m=128); length 16: 0.06,
  0.06, 0.05 (parallel is slower on short sequences); length 4,096: 1.02, 2.94, 3.36. Text: "up to 40x".
  <- key number for the page.
- Table 2 (model speedup at 65,536): SRU 9.21, QRNN (filter 2) 6.68, QRNN (filter 10) 2.05, GILR-LSTM 1.41. "up to 9x".
- Table 3: at 1,048,576 steps GILR-LSTM converges in 9.7 +- 1.7 hours; CuDNN LSTM did not converge in several days.

## 9. Limitations stated
- Speedup only for long sequences; short sequences are slower (Table 1).
- Inference speedup only "where the entire input sequence is known at the start" (Sec 4.1.2).
- Dense Lambda costs O(n^3) per combine and n^2 T memory; the paper restricts to diagonal (Sec 2.1).
- Mamba (2312.00752) fixes the memory-traffic side: it fuses discretization + scan in SRAM and adds a state
  expansion N (Mamba Sec 3.3.2, App. D).

## 10. Outline feedback
- ch 7: the outline's operator (a1,b1) o (a2,b2) = (a1 a2, a2 b1 + b2) is right for diagonal a, but Martin and
  Cundy write it as (Lambda, x) . h = Lambda h + x plus chunk summaries (P_i, R_i). Cite it as "equivalent form".
- ch 7 work/depth readout: use Sec 2.1 (serial O(2nT) vs parallel O(6n(T/p + lg p))). Table 1's 0.06x at length
  16 is a good honest point: the scan pays off only at long lengths.
- ch 1 could cite this paper's framing for "linear recurrence = parallel over length" (Sec 3 intro: "non-linear depth
  ... is the number of layers and not the sequence length").
