# Design pass: Transformers and LLMs, explained in 3D (2026-10-01)

The page keeps its identity: Red Hat Display + Spectral, its q/k/v/acc tokens (blue query, orange key,
green value, purple accent), the 3D template and its slow drag-to-rotate stages.

## Changes
- Hero (new, illustrative): one prediction step. Six key blocks, a dashed query slot, attention arcs
  whose thickness is the weight, values flowing back, three next-token bars, and the winner ("because",
  the word that follows in the attention figure's sentence) filling the slot. Plays once (6.3 s), ends
  static, Replay button, final frame under reduced motion. A narrow layout (bars under the row) below
  canvas aspect 1.6. Replaces the decorative token chips under the lede.
- Section bar: current-section highlight, reading-progress line, and a Sections menu whenever the links
  do not fit (here at every width: ten links).
- Equations: six Unicode text blocks rendered with KaTeX (CSS from cdnjs 0.18.9 with SRI), terms colored
  by the page tokens. Four point at their figure; every link word lights a real layer.
- Prose alignment: `.wrap.prose` sections were centered at 700px while the others sat at the wrap's
  left edge; all prose now shares one left edge.
- Footer sources: one list item per reference instead of middle-dot joined runs.

## Equation transcription (original -> TeX), checked one by one
| id | original | TeX |
|---|---|---|
| eq-attn | Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + M ) V | \mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^\top}{\sqrt{d_k}}+M)V |
| (block) | h = x + Attention( Norm(x) ) -> out = h + MLP( Norm(h) ) | same, \longrightarrow |
| eq-rope | <R(m theta) q, R(n theta) k> = <q, R((n - m) theta) k> | same, \langle \rangle |
| (loss) | L = -(1/T) Sum_t log p_theta( x_t \| x_<t ) | -\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t}) |
| eq-kv | KV memory = 2 x layers x KV heads x head dim x tokens x bytes per value | same, \text{} words |
| eq-chin | L(N, D) = E + A / N^alpha + B / D^beta with E = 1.69, A = 406.4, B = 410.7, alpha = 0.34, beta = 0.28 | same, \frac; constants unchanged |

## Equation links (word -> what lights)
- eq-attn: softmax -> all bars glow; Q -> query labels grow; K -> key labels grow; sqrt(d_k) -> the
  "Score scale" control; M -> the masked cells glow red-orange.
- eq-rope: R(m theta) q -> query arrows lengthen + "Query position m"; R(n theta) k -> key arrows +
  "Key position n"; R((n - m) theta) -> both arrows + the readout.
- eq-kv: 2 -> every key and value cube glows; layers -> layer labels; KV heads -> attention-type
  buttons; tokens -> "tokens generated" label + "Context length".
- eq-chin: E -> a floor plane at the irreducible loss; A/N^alpha -> the parameter-axis labels;
  B/D^beta -> the token-axis labels.

## Not changed
No prose, no number, no figure behavior other than the glows. The page has no claims list, so no
fact check ran.
