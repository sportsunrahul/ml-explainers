# AlphaZero (Silver et al. 2017)

## 1. Identity
- Title: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. arXiv 1712.01815,
  version read v1 (2017-12-05; only version).
- Authors: David Silver (first), Demis Hassabis (last; 13 authors). PDF 19 pages; Science-style, no numbered
  sections (main text, then "Methods").

## 2. Core problem
Strong chess and shogi programs rely on handcrafted evaluation and domain-specific search heuristics. Can one
generic self-play RL algorithm, given only the rules, reach superhuman play in chess, shogi and Go?

## 3. What prior methods did and why that hurt (per the source)
Programs use "sophisticated search techniques, domain-specific adaptations, and handcrafted evaluation functions
that have been refined by human experts over several decades" (abstract); they "are highly tuned to their domain,
and cannot be generalised to other problems without significant human effort" (p1-2).

## 4. Key idea
Generalize AlphaGo Zero: a single network (p, v) = f_theta(s) guides a general-purpose MCTS; self-play games
labelled with the search policy and the final outcome train the network. The game rules serve as a perfect
simulator inside the search.

## 5. Mechanism (p3)
1. Network takes board position s, outputs move probabilities p (p_a = Pr(a | s)) and value v ~ E[z | s].
2. Each move: MCTS simulations select moves with "low visit count, high move probability and high value";
   the search returns pi over moves (proportional or greedy w.r.t. root visit counts).
3. Self-play: both players move by MCTS, a_t ~ pi_t. Terminal position scored: z = -1 loss, 0 draw, +1 win.
4. Train theta by gradient descent on Eq. 1; the latest parameters generate new games (no best-player gating,
   unlike AlphaGo Zero).
Differences from AlphaGo Zero (p3-4): optimizes expected outcome (draws allowed); no symmetry augmentation; single
continually updated network; same hyperparameters for all games except exploration noise scale.

## 6. Equations
- Eq. 1 (p3): `(\mathbf{p}, v) = f_\theta(s), && l = (z - v)^2 - \pmb{\pi}^\top \log \mathbf{p} + c ||\theta||^2`
  Source: "a loss function l that sums over mean-squared error and cross-entropy losses respectively ... where c is a
  parameter controlling the level of L2 weight regularisation." Gloss: z = game outcome (target, constant);
  v = predicted outcome; pi = MCTS visit distribution (target, constant); p = predicted move probabilities.

## 7. Config (p4)
| Setting | Value |
|---|---|
| Training | 700,000 steps, mini-batches of size 4,096, from random initialization |
| Self-play hardware | 5,000 first-generation TPUs |
| Training hardware | 64 second-generation TPUs |
| Evaluation matches | 100 games per opponent, 1 minute per move; AlphaZero on 1 machine with 4 TPUs |
| Opponents | Stockfish (2016 TCEC champion, chess), Elmo (2017 CSA champion, shogi), AlphaGo Zero trained 3 days (Go) |

## 8. Results
- Abstract (p1) verbatim: "Starting from random play, and given no domain knowledge except the game rules,
  AlphaZero achieved within 24 hours a superhuman level of play in the games of chess and shogi (Japanese chess) as
  well as Go, and convincingly defeated a world-champion program in each case."
- Time to beat the reference program (p4, Fig. 1): "In chess, AlphaZero outperformed Stockfish after just 4 hours
  (300k steps); in shogi, AlphaZero outperformed Elmo after less than 2 hours (110k steps); and in Go, AlphaZero
  outperformed AlphaGo Lee after 8 hours (165k steps)."
- Table 1 (p5), from AlphaZero's view, win/draw/loss per 50 games by colour: chess as White vs Stockfish 25/25/0, as
  Black 3/47/0; shogi as White vs Elmo 43/2/5, as Black 47/0/3; Go vs AG0 3-day as White 31/-/19, as Black 29/-/21.
  Text (p5): "AlphaZero convincingly defeated all opponents, losing zero games to Stockfish and eight games to Elmo
  ..., as well as defeating the previous version of AlphaGo Zero."
- Search (p5): AlphaZero searches 80 thousand positions per second in chess and 40 thousand in shogi, vs 70 million
  (Stockfish) and 35 million (Elmo).

## 9. Limitations (stated) and later fixes
- Needs the game rules (a perfect simulator) inside the search. MuZero (1911.08265) removes this: its App. A lists
  the three uses of rules AlphaZero relied on (transitions, legal actions, terminal states) and replaces them with
  a learned model.
- Two-player, undiscounted, terminal reward +-1 setting only (per MuZero App. A).

## 10. Outline feedback
- Two different opponents in Go: the 8-hour mark is vs AlphaGo Lee; Table 1 matches are vs AlphaGo Zero (3 days).
  Do not merge them into one sentence.
- Table 1 totals (28 wins, 72 draws, 0 losses in chess) are our sums of printed rows, not printed numbers. If the page
  shows totals, label them as sums or show the rows.
- In ch-model, place AlphaZero as "planning with a known model" (the chapter-5 setting at scale), then MuZero as
  "planning with a learned model".
