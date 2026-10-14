# MuZero (Schrittwieser et al.)

## 1. Identity
- Title: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. arXiv 1911.08265, version read v2
  (2020-02-21); v1 2019-11-19. PDF 21 pages.
- Authors: Julian Schrittwieser (first), David Silver (last; 12 authors).
- Sections: 1 Introduction, 2 Prior Work, 3 MuZero Algorithm, 4 Results, 5 Conclusions; Appendix A Comparison to
  AlphaZero (p11), B Search, C Hyperparameters (p13), D Data Generation, E Network Input, G Training (p15),
  H Reanalyze, I Evaluation.

## 2. Core problem
Tree search needs a simulator (game rules). Real problems have unknown dynamics. Model-based RL that reconstructs
observations had lagged model-free methods on visually rich domains such as Atari.

## 3. What prior methods did and why that hurt (per the source, Sec. 2)
- The "tripartite separation between representation learning, model learning, and planning is potentially
  problematic since the agent is not able to optimize its representation or model for the purpose of effective
  planning, so that, for example modeling errors may compound during planning."
- Pixel-level models: "planning at pixel-level granularity is not computationally tractable in large scale
  problems"; latent models that reconstruct or predict latents focus "the majority of the model capacity on
  potentially irrelevant detail".

## 4. Key idea
Learn a model that predicts only what planning needs, "the reward, the action-selection policy, and the value
function" (abstract), with no requirement that its hidden state reconstruct the observation or match the true
state. Plan over it with an AlphaZero-style MCTS.

## 5. Mechanism
The three learned functions (Sec. 3, p3-4):
1. Representation h: s^0 = h_theta(o_1, ..., o_t). Encodes past observations into the root hidden state.
2. Dynamics g: r^k, s^k = g_theta(s^{k-1}, a^k). Outputs an immediate reward and the next hidden state;
   deterministic in this paper ("the extension to stochastic transitions is left for future work").
3. Prediction f: p^k, v^k = f_theta(s^k). Outputs a policy and a value, "akin to the joint policy and value network
   of AlphaZero".
Acting (Fig. 1B): an MCTS runs at every time step over hidden states; action a_{t+1} is sampled from the search
policy pi_t, proportional to root visit counts.
Training (Fig. 1C): sample a trajectory from the replay buffer, run h on past observations, unroll g for K steps
with the real actions, match p^k to the search policy pi_{t+k}, v^k to the value target z_{t+k}, and r^k to the
observed reward u_{t+k}. All three functions train jointly by backpropagation through time.
Targets (constants, no gradient): pi_{t+k} from MCTS; z_t = u_{t+1} + gamma u_{t+2} + ... + gamma^{n-1} u_{t+n} +
gamma^n nu_{t+n} (n-step return bootstrapped from the search value nu); final outcomes in board games are rewards
in {-1, 0, +1}.
Knowledge of rules replaced (App. A): state transitions, legal actions inside the tree (masked only at the root),
and terminal nodes.

## 6. Equations
- Eq. 1 (Sec. 3, p4):
  `l_t(\theta) &= \sum_{k=0}^K l^r (u_{t+k}, r_t^k) + l^v(z_{t+k}, v^k_t) + l^p(\pi_{t+k}, \mathbf{p}^k_t) + c ||\theta||^2`
  Source: "where l^r, l^v, and l^p are loss functions for reward, value and policy respectively." Gloss: u = observed
  reward, r^k = predicted reward after k imagined steps; z = value target; v^k = predicted value; pi = MCTS visit
  distribution; p^k = predicted policy; c = L2 weight. Footnote: squared error for rewards and values in board
  games, cross-entropy in Atari; cross-entropy for the policy in both.
- Notation: the action chosen at time t is written a_{t+1} and its reward u_{t+1} (Fig. 1B). Action and reward share
  an index, unlike S&B (A_t, R_{t+1}).

## 7. Config
| Setting | Value | Location |
|---|---|---|
| Unroll steps K | 5 | Sec. 4, p4 |
| Training | 1 million mini-batches; size 2048 (board games), 1024 (Atari) | Sec. 4, p4 |
| Simulations per search | 800 (board games), 50 (Atari) | Sec. 4, p4 |
| Representation network | AlphaZero conv/residual architecture with 16 residual blocks (AlphaZero: 20); 256 hidden planes | Sec. 4, p4 |
| Discount (Atari) | 0.997 (same as R2D2) | App. C, p13 |
| Value bootstrap | to end of game (board games); n = 10 (Atari) | App. G, p15 |
| Reanalyze variant | n = 5; fresh MCTS policy targets for 80% of updates | App. H, p15 |

## 8. Results
- Abstract: "When evaluated on 57 different Atari games ... our new algorithm achieved a new state of the art. When
  evaluated on Go, chess and shogi, without any knowledge of the game rules, MuZero matched the superhuman
  performance of the AlphaZero algorithm that was supplied with the game rules."
- Go (Sec. 4, p5): "In Go, MuZero slightly exceeded the performance of AlphaZero, despite using less computation per
  node in the search tree (16 residual blocks per evaluation in MuZero compared to 20 blocks in AlphaZero)." Elo
  values exist only as Fig. 2 curves; no numbers are stated in the text.
- Atari (Sec. 4, p5): new state of the art for mean and median normalized score across 57 games, "outperforming the
  previous state-of-the-art method R2D2 (a model-free approach) in 42 out of 57 games, and outperforming the previous
  best model-based approach SimPLe in all games".
- Table 1 (p6), human-normalized scores: MuZero median 2041.1%, mean 4999.2% (20.0B frames, 12 hours, 1M training
  steps); R2D2 median 1920.6%, mean 4024.9% (37.5B frames). MuZero Reanalyze at 200M frames: median 731.1%, mean
  2168.9% (text p5 rounds to 731%, vs IMPALA 192%, Rainbow 231%, LASER 431%).
- Ablation (Sec. 4, p6): replacing the MuZero objective with Q-learning on Ms. Pacman "learned significantly slower
  than MuZero and converged to a much lower final score". Atari planning gains "plateaued at around 100 simulations".

## 9. Limitations (stated) and later fixes
- Deterministic dynamics; stochastic transitions left to future work (Sec. 3). Imperfect-information games such as
  poker "are not directly addressed" (Sec. 4 footnote).
- Planning gains in Atari smaller than in Go, "perhaps because of greater model inaccuracy" (Sec. 4).

## 10. Outline feedback
- viz-mbfamily, MuZero panel: the model predicts reward, value and policy only, never observations. "Trained in
  imagination" is not the right label for MuZero: the model is used for lookahead search (MCTS) at every step, and the
  network is trained on real trajectories with search targets. Label it "plans with the model" vs "learns from
  imagined data" (Dyna, World Models, Dreamer).
- Sources list: AlphaZero (1712.01815) is the known-rules baseline. It uses a perfect simulator, not a learned model;
  present it as such.
