# Reinforcement learning, from a score to PPO: outline

Reader: an ML engineer who knows deep learning basics (SGD, softmax, cross-entropy, CNNs, autograd) and is new
to RL. The page is a course, not a paper digest: each chapter teaches one idea with a live figure, worked
numbers, a self-check and two code exercises (NumPy from scratch, PyTorch practical). Papers come second.

Mode: course. Primary source for chapters 1-8 and 10-11 is Sutton and Barto, Reinforcement Learning: An
Introduction, 2nd ed. (2018), cited as "S&B" with section and equation numbers. arXiv papers carry chapters 9 and
11-15. Long page only, no narration, no atlas (cue ids and reset buttons still follow the cue contract).

## The story in one paragraph

Supervised learning hands you the right answer. RL hands you only a score, often late, for data that your own
choices produced. Those three facts (you must explore, you must assign credit across time, and your data shifts
when your policy shifts) are the spine of the page; every chapter fixes part of one of them. We start with the
smallest case (one situation, many actions: bandits), add situations (MDPs), learn what a situation is worth
(Bellman), plan when we know the rules (dynamic programming), learn from experience when we do not (Monte Carlo,
TD, Q-learning), scale to images (function approximation, DQN), then switch from learning values to learning the
policy itself (policy gradients, actor-critic, PPO, SAC). Two short bridges end the course: learning a model of
the world, and RL for language models. A practice guide closes the page.

## Running worlds (reused so the learner builds intuition on familiar ground)

- The yard (chapters 3-5, hero): a 5x4 gridworld. A robot starts bottom-left; the charger (+1, terminal) is
  top-right; a stairwell (-1, terminal) sits just below the charger; one wall cell; step reward -0.04; moves slip
  with probability 0.2 (0.1 to each side), as in the classic Russell-Norvig world. A "two chargers" variant (a small
  near reward and a big far one) shows how discount changes the best path. All values on the page are computed.
- Bandit testbed (chapter 2): 10 arms, S&B section 2.3 settings.
- Random walk (chapter 6): 5 states A-E (S&B Example 6.2); 19-state version for n-step (S&B section 7.1).
- Cliff walk (chapter 7): 4x12 (S&B Example 6.6).
- Two-state "w, 2w" world and Baird's counterexample (chapter 8, S&B section 11.2).
- CartPole (chapters 9-12): our own NumPy implementation of the classic dynamics (Barto, Sutton and Anderson
  1983), checked against Gymnasium CartPole-v1. DQN and PPO runs are recorded offline (build/sim/) and shown as
  scrubbers with episode playback; badge Computed.
- Short corridor (chapter 10, S&B Example 13.1), 1D continuous action toys (chapter 13), Dyna maze (chapter 14,
  S&B Example 8.1), a toy "language model" over short token strings (chapter 15).

## Page-wide devices

- Three-problems chips at each chapter top: exploration, credit assignment, data from your own policy. The chapter
  lights the one(s) it addresses. The first chapter introduces them; the guide revisits them.
- Notation card (a `details` near the top and in the guide): S_t, A_t, R_{t+1}, G_t, gamma, pi, V, Q, delta, A
  (advantage), theta, w. We use the S&B convention (reward after A_t is R_{t+1}) and say once that deep RL
  papers (DQN, PPO) write r_t for the same reward. This index shift is a classic confusion; name it.
- Color key (one meaning per color, whole page): blue = value estimates (V, Q, critic); yellow = rewards and
  returns; green = the policy and its actions (actor); red = errors, losses and gradients (TD error, loss);
  brown = the environment's fixed parts (walls, frozen networks, the model). Stated once in the intro.
- Live vs recorded: tabular algorithms run live in the page with a seeded RNG (reset restores the seed). Deep RL
  runs are recorded offline and replayed; the caption says which.
- Exercise block per chapter: tabs "NumPy" and "PyTorch", each a complete script that we ran, with its real
  printed output and a "Try this" prompt (change one thing, predict the effect). Code lives in
  `code/chNN-<slug>/{numpy_ex.py,torch_ex.py}`; the build inlines the files and their recorded output, so the page
  shows exactly the code that ran. NumPy tabs use our own tiny environments; PyTorch tabs use Gymnasium.

## Hero

The yard, live: episodes of a learning agent run fast; value colors ripple back from the charger; policy arrows
settle; ends on a static frame labelled with the loop (state, action, reward, next state). Computed from a real
Q-learning run with a fixed seed. Replay button.

---

## Act I. The problem

### 1. ch-score: Learning from a score, not an answer

- Learner's question: "I know how to train a classifier. What changes when nobody tells the model the right answer?"
- The one intuition: RL gets a score for what it did, not the right answer; the score can arrive late; and the
  agent's own choices decide which data it sees. Exploration, credit assignment and self-generated data are the
  three problems the whole course solves.
- Steps (h3): The loop (agent, environment, state, action, reward) -> Same task, two kinds of feedback (supervised
  label vs reward) -> Problem 1: a score does not say what would have been better -> Problem 2: the score comes
  late -> Problem 3: your data comes from your own choices -> The goal as one equation -> Where this came from
  (Thorndike's law of effect, Bellman, Sutton and Barto; RL in games, robots, LLM post-training) -> The road map.
- Main figure viz-loop (Toy): "Be the agent." A robot on a 6-cell track must reach the charger. Mode buttons:
  "Supervised" (a teacher shows the correct arrow each step), "RL" (you press left/right and get one score at the
  end), "Visited states" (a heat strip of which states your choices let you see). Prediction prompt: "You pressed
  left three times and scored 0. Which press was the mistake?"
- Equation: J(pi) = E_pi[G], G = R_1 + R_2 + ... + R_T (undiscounted here; discount comes in chapter 3). Worked
  example: rewards 0, 0, 0, +1 -> G = 1.
- Confusion to name: a reward is not a label. A label says what is right; a reward says how good what you did was.
- Self-check: "An agent gets +1 at step 50 of a game. Which of the three problems does this create?" (credit).
- In practice: if you can get labels, supervised learning is far cheaper; reach for RL when the only feedback is
  a score from acting.
- Exercise: NumPy: write the agent-environment loop for the track world with a random policy; average return over
  1000 episodes. PyTorch: run Gymnasium CartPole-v1 with an untrained `nn.Module` policy and
  `torch.distributions.Categorical`; print the mean episode length.
- Sources: S&B ch. 1 and section 3.1; DQN Nature, InstructGPT abstract (two "where RL shows up" numbers at most).

### 2. ch-bandits: Explore or exploit

- Learner's question: "If I only learn from scores, how do I find the best action without wasting tries?"
- The one intuition: always picking the best-looking action gets stuck on an early lucky guess; a little deliberate
  exploration finds better actions, and the long-run gain beats the short-run cost. The update "new estimate = old +
  step size x (target - old)" appears here first and in every later chapter.
- Steps: Play it yourself (5 slot machines, 20 pulls) -> Estimate by averaging -> The incremental update -> Greedy
  gets stuck -> epsilon-greedy -> Optimistic starts (one paragraph) -> UCB: explore where you are unsure -> Gradient
  bandit: learn preferences, not values (the first policy gradient; plant for chapter 10, including its baseline)
  -> Where this came from.
- Main figure viz-bandit (Toy, live): the reader pulls arms; a "Let epsilon-greedy play" button continues from the
  reader's state; reveal true means at the end.
- Second figure viz-testbed (Computed, live): average reward and % optimal action over 1000 steps for greedy,
  epsilon 0.01, epsilon 0.1 and UCB c = 2, on the S&B 10-armed testbed (runs computed in the page; run count shown).
- Equations: incremental mean (S&B Eq. 2.3), epsilon-greedy (text), UCB (S&B Eq. 2.10), gradient bandit update
  (S&B Eq. 2.12) with its softmax (S&B Eq. 2.11). Worked example after each.
- Confusion: "exploration wastes reward" (short run yes, long run no); epsilon-greedy explores even hopeless arms
  (UCB does not).
- Self-check: one incremental update by hand; why epsilon 0.01 eventually beats 0.1.
- In practice: A/B tests and recommendation are bandits; in full RL the same trade-off spans whole sequences.
- Exercise: NumPy: epsilon-greedy on the 10-armed testbed, print the final average reward per epsilon. PyTorch:
  gradient bandit with autograd (loss = -(R - baseline) * log pi(A)); assert the autograd update equals S&B Eq. 2.12.
- Sources: S&B ch. 2 (testbed settings, Figures 2.2-2.5 qualitative findings, Eqs. 2.3, 2.10-2.12).

## Act II. Values and planning

### 3. ch-mdp: MDPs, returns and discount

- Learner's question: "Bandits have one situation. What changes when actions move you to new situations?"
- The one intuition: an MDP is a bandit whose actions also change the situation; you now care about all future
  rewards, and the discount factor sets how far ahead you look.
- Steps: State and the Markov property -> Transitions can be random (the slipping robot) -> Policy -> Return and
  discount -> The return is recursive (G_t = R_{t+1} + gamma G_{t+1}; plant for Bellman) -> Why discount, and the
  horizon rule of thumb 1/(1 - gamma) (label as a rule of thumb) -> Discount changes the best path (two chargers) ->
  Episodes vs continuing tasks -> Where this came from.
- Main figure viz-yard (Toy, live): the yard; pick a start cell and a policy; "Run episode" animates slips; a
  return counter with a gamma slider; "Run 100 episodes" draws the return histogram (randomness from slips).
- Second figure viz-discount (Computed): bars gamma^k with a gamma slider; readout of the horizon rule of thumb;
  the two-chargers comparison (near +1 in 2 steps vs far +10 in 6 steps; crossover where gamma^4 = 0.1, so gamma
  is about 0.562; derived here).
- Equations: G_t (S&B Eq. 3.8), recursion (S&B Eq. 3.9). Worked example: rewards -1, -1, -1, +10 at gamma 0.9.
- Confusion: index convention (R_{t+1} follows A_t); "the reward is the goal" (reward design decides behavior).
- Self-check: compute a discounted return; which charger wins at gamma 0.6.
- In practice: gamma 0.99 is the common deep-RL default (PPO paper tables); reward design and scaling matter.
- Exercise: NumPy: implement the yard's step function and estimate a fixed policy's mean return by simulation.
  PyTorch: a vectorized reverse-cumsum `discounted_returns(rewards, gamma)` for a batch of episodes (reused in
  chapter 10), tested against a Python loop.
- Sources: S&B sections 3.1-3.4.

### 4. ch-bellman: Bellman equations

- Learner's question: "How good is it to be in a state, and can I compute that without simulating forever?"
- The one intuition: a state's value is the reward you expect now plus the discounted value of where you expect to
  land. Values are defined through each other, so they can be solved for.
- Steps: V_pi(s) as expected return -> One step of look-ahead (the backup diagram) -> The Bellman expectation
  equation, with numbers -> Q_pi(s, a), and V from Q -> Optimal values: the max replaces the average -> Acting from
  Q* needs no model (plant for Q-learning) -> For a fixed policy it is a linear system -> Where this came from
  (Bellman 1957).
- Main figure viz-backup (Computed): the yard with exact values; click a cell to open its backup tree (actions,
  slip probabilities, rewards, neighbor values) with the arithmetic filled in, showing left side = right side.
  Buttons: "Follow the policy" (expectation) vs "Best action" (optimality).
- Equations: S&B Eq. 3.14 (Bellman for v_pi), Eq. 3.19 (Bellman optimality for v*), Eq. 3.20 (for q*). Worked
  example on a 3-state chain at gamma 0.9.
- Confusion: the Bellman equation is a consistency condition, not an algorithm; chapter 5 turns it into one.
- Self-check: compute V(s) from given neighbor values and slip probabilities.
- In practice: Q is what model-free agents learn, because acting from Q needs no model.
- Exercise: NumPy: solve (I - gamma P_pi) v = r_pi exactly on the yard; check one Bellman equation by hand.
  PyTorch: the same with `torch.linalg.solve` on Gymnasium FrozenLake-v1's transition table (`env.unwrapped.P`).
- Sources: S&B sections 3.5-3.6.

### 5. ch-dp: Dynamic programming

- Learner's question: "If I know the rules of the world, how do I find the best policy?"
- The one intuition: turn the Bellman equation into an update and sweep it over all states; values spread out from
  rewards like ripples, and acting greedily on them gives a better policy each round.
- Steps: Iterative policy evaluation -> Each sweep shrinks the error by about a factor gamma (contraction, shown on
  a log plot) -> Policy improvement (greedy is at least as good) -> Policy iteration -> Value iteration -> The
  policy settles long before the values do -> Generalized policy iteration (evaluation and improvement interleaved;
  plant for actor-critic) -> The catch: you need the model and you sweep every state -> Where this came from.
- Main figure viz-dp (Computed, live): the yard; buttons "One sweep", "Run to convergence", "Improve policy",
  mode "Policy iteration" vs "Value iteration"; value heatmap, greedy arrows, and a max |delta V| vs sweep plot.
- Equations: S&B Eq. 4.5 (iterative policy evaluation), Eq. 4.9 (greedy improvement), Eq. 4.10 (value iteration).
  Worked example: one sweep on a 4-cell corridor.
- Confusion: "more sweeps always change the policy" (they do not; show when arrows stop changing).
- Self-check: after k sweeps, how far from the charger has information travelled?
- In practice: DP itself is rare (it needs the model); every later algorithm approximates these backups from samples.
- Exercise: NumPy: policy iteration and value iteration on the yard; reproduce S&B Figure 4.1 values for the 4x4
  example as a known check. PyTorch: vectorized value iteration on FrozenLake-v1 from `env.unwrapped.P`, then
  measure the greedy policy's success rate over 1000 episodes.
- Sources: S&B ch. 4 (Example 4.1 values, Eqs. 4.5, 4.9, 4.10, policy improvement theorem).

## Act III. Learning from experience

### 6. ch-mctd: Monte Carlo vs temporal difference

- Learner's question: "What if I do not know the rules and can only try things?"
- The one intuition: Monte Carlo waits for the end of the episode and averages true returns (unbiased but noisy);
  TD updates after one step toward reward plus its own guess of the next value (bootstrapping: biased early, much
  less noisy, and it learns online). n-step returns are a dial between the two.
- Steps: Learn V from episodes (MC, the chapter 2 update with target G) -> TD(0) and the TD error -> The driving-home
  analogy (S&B Example 6.1) -> The race on the random walk -> Why TD wins here: bias vs variance -> n-step returns, the
  dial (plant for GAE in chapter 11) -> Where this came from (Sutton 1988).
- Main figure viz-walk (Computed, live): the 5-state random walk; episodes animate; value bars vs true values
  (1/6 ... 5/6) for MC and TD side by side; RMS error curves with alpha buttons.
- Second figure viz-nstep (Computed, live): RMS error vs alpha for n = 1, 2, 4, ... on the 19-state walk (the shape
  of S&B Figure 7.2); readout of the best n.
- Equations: constant-alpha MC (S&B Eq. 6.1), TD(0) (S&B Eq. 6.2), TD error (S&B Eq. 6.5), n-step return
  (S&B Eq. 7.1). Worked TD update with numbers.
- Confusion: "a target built on a wrong guess must be worse" (early on, variance hurts more than bias).
- Self-check: one TD(0) update by hand; which method updates before the episode ends.
- In practice: almost all deep RL bootstraps (DQN, actor-critic); Monte Carlo returns survive in REINFORCE and GRPO.
- Exercise: NumPy: MC and TD(0) on the random walk, print RMS error after 100 episodes. PyTorch: TD(0) as SGD: a
  value table as `nn.Embedding`, loss = (target.detach() - V(s))^2 with SGD lr = alpha; assert it matches the
  tabular update (plant for "semi-gradient" in chapter 8).
- Sources: S&B ch. 5 (MC basics), ch. 6 (Examples 6.1, 6.2), section 7.1 (Figure 7.2 setup).

### 7. ch-qlearn: SARSA vs Q-learning

- Learner's question: "How do I learn the best policy itself from experience, not just values?"
- The one intuition: learn Q(s, a) by TD and act epsilon-greedily. SARSA learns the value of the policy it actually
  follows, exploration mistakes included (on-policy). Q-learning learns the value of the greedy policy however it
  explores, by taking a max in its target (off-policy). On a cliff edge that one difference changes the path.
- Steps: Why Q, not V -> SARSA -> Q-learning, the same transition with numbers -> The cliff -> On-policy vs
  off-policy, named (target policy vs behavior policy) -> Importance sampling: learning about one policy from
  another's data, by reweighting with the ratio pi/b (plant for PPO's ratio in chapter 12) -> Why one-step
  Q-learning needs no ratio -> Maximization bias and Double Q-learning (plant for Double DQN) -> Where this came from
  (Watkins 1989; Rummery and Niranjan 1994).
- Main figure viz-cliff (Computed, live): the 4x12 cliff; train SARSA and Q-learning side by side; learned paths
  drawn; sum of rewards per episode curves (epsilon 0.1).
- Second figure viz-ratio (Toy, computed): a two-action example; behavior policy b vs target policy pi; the
  importance-sampled estimate converges to pi's value, and its variance grows as pi moves away from b (slider).
- Equations: SARSA (S&B Eq. 6.7), Q-learning (S&B Eq. 6.8), importance sampling ratio (S&B Eq. 5.3), Double
  Q-learning (S&B Eq. 6.10). Worked updates for SARSA and Q-learning from one transition.
- Confusion: "Q-learning is better because it finds the shortest path" (online it falls off more often; which is
  better depends on whether exploration mistakes cost you).
- Self-check: two one-step updates by hand; which learned path belongs to which method.
- In practice: off-policy methods can reuse old data (replay buffers, chapter 9); on-policy methods such as PPO need
  fresh data after each update.
- Exercise: NumPy: SARSA and Q-learning on our cliff; print the learned path and the mean online reward.
  PyTorch: Q-learning on Gymnasium CliffWalking-v1 with a torch Q table; print the greedy path length.
- Sources: S&B sections 5.5, 6.4-6.7 (Example 6.6, Example 6.7, Eqs. 5.3, 6.7, 6.8, 6.10), Watkins and Dayan 1992.

## Act IV. Deep RL with values

### 8. ch-approx: Function approximation and the deadly triad

- Learner's question: "Tables need one entry per state. What do I do when states are images?"
- The one intuition: replace the table with a function that generalizes across similar states and train it by SGD
  toward TD targets. But function approximation, bootstrapping and off-policy data together can make values spiral
  to infinity: the deadly triad.
- Steps: How many Atari screens are there (210x160 frames, 128 colors) -> Values as a function v(s; w) -> The loss and
  the semi-gradient (the target is held fixed: `detach`) -> Generalization helps and hurts (update one state, its
  neighbors move) -> The w, 2w example: two states, one update rule, and w grows each step when gamma > 0.5 (derived
  with numbers) -> The triad named; any two are safe, all three can diverge -> Baird's counterexample (one paragraph)
  -> What deep RL does about it (next chapter) and what the 2018 study measured -> Where this came from.
- Main figure viz-triad (Computed, live): the two-state example; sliders gamma and alpha; buttons "Off-policy
  updates" vs "On-policy updates" and "TD target" vs "Monte Carlo target"; w over steps on a log axis.
- Second figure viz-generalize (Toy, live): a 1D track with a linear value function over radial features; click to
  update one state and watch its neighbors move.
- Equations: semi-gradient TD(0) (S&B section 9.3 update), the w, 2w recursion (S&B section 11.2, derived on the
  page with numbers).
- Confusion: "divergence means my code has a bug" (the triad alone can do it).
- Self-check: which two triad ingredients does tabular Q-learning have, and why is it safe?
- In practice: if values blow up, lower the learning rate, use a target network, use n-step or Monte Carlo targets,
  clip gradients.
- Exercise: NumPy: reproduce w, 2w divergence and on-policy convergence. PyTorch: semi-gradient TD(0) with a small
  MLP on the random walk; compare with the "full gradient" version (no detach) and print both errors.
- Sources: S&B sections 9.1-9.3, 11.1-11.3 (Baird, the w, 2w example, the triad); van Hasselt et al. 2018
  (1812.02648); DQN Nature (frame size and palette).

### 9. ch-dqn: DQN and its family

- Learner's question: "How did one network learn many Atari games from pixels?"
- The one intuition: DQN is Q-learning with a CNN, kept stable by two tricks: a replay buffer that shuffles old
  experience (breaking correlations; allowed because Q-learning is off-policy) and a target network, a lagged copy
  that stops the target from chasing itself. Later fixes each patch one weakness; Rainbow combines them.
- Steps: The setup (stacked frames in, one Q per action out) -> The loss -> Trick 1, replay -> Trick 2, target network
  -> What it achieved -> Overestimation (max over noisy estimates is biased up) -> Double DQN -> The family (dueling,
  prioritized replay, multi-step, distributional, noisy nets, Rainbow) as a tree -> Where this came from.
- Main figure viz-dqnrun (Computed, recorded): our NumPy DQN on CartPole, three variants over 5 seeds each (full,
  no target network, no replay): return curves with seed bands; a scrubber plays a recorded episode at chosen
  checkpoints.
- Second figure viz-maxbias (Computed, live): N actions with true value 0 and noisy estimates; the mean of the max
  grows with N and noise; the double estimator stays near 0.
- Architecture walkthrough viz-dqnflow (archflow): forward 84x84x4 -> conv 8x8/4 (32) -> conv 4x4/2 (64) -> conv
  3x3/1 (64) -> fc 512 -> one Q per action (shapes computed from the Nature config); backward: the loss grades only
  Q(s, a) of the action taken; gradient reaches the online network; the target network is frozen and copied every
  C steps.
- Equations: DQN loss (Nature Eq. 1 or 2, whichever the PDF numbers), Double DQN target (van Hasselt et al. Eq. 4 as
  numbered in the source). Worked target computation with numbers.
- Confusion: "the target network is trained" (it is a lagged copy, never trained directly).
- Self-check: why replay is fine for Q-learning but wrong for SARSA.
- In practice: DQN-family fits discrete actions and cheap simulators; the Nature hyperparameter table as defaults.
- Exercise: NumPy: DQN on our CartPole with a 2-layer MLP and hand-written backprop (the code behind viz-dqnrun).
  PyTorch: DQN on Gymnasium CartPole-v1, CleanRL style.
- Sources: 1312.5602v1, DQN Nature 2015 (classic/mnih2015), 1509.06461v3, 1511.06581v3, 1511.05952v4,
  1707.06887v1, 1710.02298v1.

## Act V. Learning the policy directly

### 10. ch-pg: Policy gradients

- Learner's question: "Instead of learning values and acting greedily, can I tune the policy directly for more reward?"
- The one intuition: make actions that led to high return more likely and actions that led to low return less
  likely. The gradient of expected return is the average of (return x gradient of the log-probability of the action
  taken): supervised learning on your own actions, weighted by how well they worked.
- Steps: Why learn the policy (continuous or huge action sets; sometimes the best policy is random: the short
  corridor) -> A softmax policy -> You cannot backprop through a sample -> The log-derivative trick, derived step by
  step -> REINFORCE -> Reward-to-go (an action cannot affect past rewards) -> The weighted cross-entropy view -> The
  variance problem and the baseline (unbiased because E[grad log pi] = 0; derived) -> Where this came from (Williams
  1992; the policy gradient theorem, Sutton et al. 2000).
- Main figure viz-pg (Toy, live): a 3-action softmax policy on a bandit; each sample shows the update arrow on the
  chosen action's logit; baseline on/off; learning curve.
- Second figure viz-corridor (Computed): the short corridor; J(p) for p = P(right); the stochastic optimum vs the two
  epsilon-greedy values (S&B Example 13.1).
- Third figure viz-variance (Computed, live): histograms of gradient estimates with and without a baseline.
- Equations: log-derivative trick (derived here), policy gradient theorem (S&B Eq. 13.5), REINFORCE (S&B Eq. 13.8),
  REINFORCE with baseline (S&B Eq. 13.11). Worked update for a 2-action softmax.
- Confusion: "we differentiate through the environment" (no: only log pi is differentiated).
- Self-check: compute one REINFORCE update by hand.
- In practice: normalize returns or advantages, batch many episodes; GRPO for LLMs is REINFORCE with a group
  baseline (chapter 15).
- Exercise: NumPy: REINFORCE with a baseline on our CartPole with a linear softmax policy (hand-written gradient).
  PyTorch: REINFORCE on CartPole-v1 with an MLP (loss = -(G_t - b) * log pi).
- Sources: S&B ch. 13 (Example 13.1, Eqs. 13.5, 13.8, 13.11), Williams 1992, Sutton et al. 2000.

### 11. ch-ac: Actor-critic and GAE

- Learner's question: "REINFORCE waits for the episode to end and is noisy. Can a learned value function help?"
- The one intuition: add a critic that estimates values; use it as the baseline and to bootstrap. The advantage (how
  much better this action did than expected) drives the actor. GAE blends n-step advantage estimates with one knob,
  lambda, that trades bias for variance: the n-step dial from chapter 6 again.
- Steps: The critic, learned by TD -> The advantage, and why the TD error estimates it -> One-step actor-critic ->
  k-step advantages -> GAE, worked backwards over a short trajectory -> lambda = 0 and lambda = 1 as special cases ->
  One network, two heads; parallel actors (A2C/A3C) -> Generalized policy iteration again (critic = evaluation, actor
  = improvement) -> Where this came from.
- Main figure viz-gae (Computed): a 10-step trajectory with rewards and critic values; a lambda slider shows each
  delta and its weight (gamma lambda)^l as fading bars; readout of A_t.
- Second figure viz-gaebv (Computed, live): bias and variance of the advantage estimate vs lambda on a toy chain
  whose critic is wrong by a set amount.
- Architecture walkthrough viz-acflow (archflow): obs -> trunk -> policy logits and value; losses: policy (-A log pi),
  value (V - target)^2, entropy bonus; backward: the advantage is a constant for the actor (detached), the value loss
  trains the critic.
- Equations: TD residual as advantage (GAE paper), k-step estimators, GAE (GAE paper Eq. 16 as numbered in the source),
  lambda = 0 and 1 cases. Worked GAE with gamma = 1, lambda = 0.5.
- Confusion: "TD error and advantage are different things" (the TD error's expectation is the advantage when V is exact).
- Self-check: compute GAE for three given deltas.
- In practice: gamma 0.99 and lambda 0.95 are common (PPO tables); normalize advantages per batch.
- Exercise: NumPy: a `gae(rewards, values, dones, gamma, lam)` function tested on lambda = 0 and 1 identities; one-step
  actor-critic on our CartPole. PyTorch: A2C on CartPole-v1 with `gymnasium.vector` and GAE.
- Sources: 1506.02438v6 (GAE), 1602.01783v2 (A3C), S&B section 13.5.

### 12. ch-ppo: TRPO to PPO

- Learner's question: "On-policy data is expensive. Why can I not take many gradient steps on one batch, and how does
  PPO make that safe?"
- The one intuition: the batch came from the old policy; once the new policy moves far from it, the batch no longer
  describes it and an update can wreck the policy (and the next batch is then bad too). TRPO limits how far the policy
  moves with a KL constraint. PPO gets the same effect cheaply: it clips the probability ratio so the objective stops
  paying for moves beyond 1 +- epsilon.
- Steps: The failure (too big a step, live) -> Reusing data: the ratio r_t (chapter 7's importance sampling) and the
  surrogate objective; at theta_old its gradient equals the policy gradient -> TRPO: maximize the surrogate subject to
  KL <= delta (intuition for the improvement bound; Kakade and Langford 2002) -> PPO-clip, worked with epsilon 0.2 ->
  The whole algorithm (N actors x T steps, GAE, K epochs of minibatch Adam, value loss, entropy bonus) -> What the
  paper measured -> Implementation details matter (2020 studies) -> Where this came from.
- Main figure viz-clip (Computed, exact): L^CLIP as a function of r for A > 0 and A < 0 (PPO Figure 1, redrawn);
  drag r; the flat, zero-gradient region is shaded; readout of the objective and its gradient.
- Second figure viz-overstep (Computed, live): K epochs on one batch for a tiny policy: the unclipped surrogate drives
  the ratio far and the true return drops; the clipped one stops.
- Architecture walkthrough viz-ppoflow (archflow): actor and critic; pi_old is frozen (a constant inside the ratio);
  advantages are constants; what each loss term trains.
- Equations: L^CPI (PPO Eq. 6), L^CLIP (PPO Eq. 7), adaptive KL penalty (PPO Eq. 8), the combined objective (PPO Eq. 9),
  the TRPO problem (as numbered in PPO's background section). Worked: A = -2, r = 0.7, epsilon = 0.2.
- Confusion: "clipping clips the gradient" (it zeroes the gradient only for samples that already moved past epsilon in
  the direction the advantage favors).
- Self-check: L^CLIP and its gradient for a given A and r.
- In practice: PPO defaults from the paper's hyperparameter tables; watch approximate KL and clip fraction; PPO is the
  RL step in RLHF (chapter 15).
- Exercise: NumPy: PPO-clip on our CartPole with a hand-written MLP (the code behind the interlude). PyTorch: PPO on
  CartPole-v1, CleanRL style, with `gymnasium.vector`.
- Sources: 1707.06347v2, 1502.05477v5, Kakade and Langford 2002, 2005.12729v1, 2006.05990v1.

### Interlude ch-step: One PPO update, end to end (unnarrated)

A stepper through one recorded update of our NumPy PPO on CartPole: rollout (envs x steps) -> rewards and values ->
GAE -> advantage normalization -> minibatches -> ratios and clip fraction -> loss terms -> Adam step -> approximate KL.
Each stage shows real numbers from the recorded update and links back to the chapter that derived its equation. A
training scrubber below shows return, entropy, approximate KL and clip fraction over updates, with episode playback.
Badge Computed.

### 13. ch-continuous: Continuous actions: Gaussian policies, DDPG, TD3, SAC

- Learner's question: "A robot joint's torque is a real number. How do I choose an action I cannot list?"
- The one intuition: two routes. A stochastic policy outputs a Gaussian and the policy gradient works as before. Or,
  off-policy: learn Q(s, a) like DQN and train a deterministic actor to climb Q's slope with respect to the action
  (DDPG). TD3 fixes DDPG's overestimation with twin critics, delayed actor updates and target smoothing. SAC adds an
  entropy bonus so the policy stays as random as it can while still succeeding.
- Steps: The max over actions is the problem -> Gaussian policies and their log-probability, with numbers -> The
  deterministic policy gradient: the chain rule through the critic -> DDPG = DQN's tricks + a deterministic actor +
  exploration noise -> Overestimation again -> TD3's three fixes -> Maximum entropy RL and SAC (reparameterization,
  tanh squashing, learned temperature) -> Choosing PPO vs SAC/TD3 -> Where this came from.
- Main figure viz-dpg (Toy, live): a 1D action; the critic's Q(a) curve; the actor's action climbs dQ/da; with a noisy,
  overestimating critic it climbs to a false peak; "Twin critics (min)" removes it.
- Second figure viz-entropy (Toy, live): a Gaussian policy on a two-peak reward; pure reward collapses the spread onto
  one peak; an alpha slider shows how the entropy bonus keeps spread.
- Architecture walkthrough viz-sacflow (archflow): actor (mean, log std, tanh) and twin critics with target copies;
  the actor's gradient passes through the critics, which are not updated by the actor loss ("frozen but the gradient
  passes through").
- Equations: deterministic policy gradient (DDPG Eq. 6 as numbered), TD3 clipped double-Q target (TD3 Eq. 10 as
  numbered), SAC objective (SAC Eq. 1). Worked TD3 target with numbers.
- Confusion: "a deterministic policy cannot explore" (noise is added to actions while collecting data).
- Self-check: compute a TD3 target from two critic values.
- In practice: SAC or TD3 when samples are costly and actions continuous; PPO when parallel simulation is cheap (our
  reading, labelled).
- Exercise: NumPy: Gaussian REINFORCE vs the deterministic gradient on the 1D toy. PyTorch: SAC on Gymnasium
  Pendulum-v1 (CleanRL style), print the final mean return.
- Sources: 1509.02971v6, 1802.09477v3, 1801.01290v2, 1812.05905 (learned temperature), Silver et al. 2014 (DPG).

## Act VI. Bridges

### 14. ch-model: Model-based RL (short bridge to the world-models page)

- Learner's question: "Model-free agents need millions of steps. Could an agent learn how the world works and practice
  in its head?"
- The one intuition: learn a model of transitions and rewards from real experience; then use it to make extra
  imagined experience or to search ahead. Dyna shows the idea with a table; World Models, Dreamer and MuZero scale it
  with learned latent models.
- Steps: Model and planning (chapter 5 again) -> Dyna-Q -> A wrong model plans wrong -> Three scaled-up designs, one
  paragraph each (World Models, Dreamer, MuZero) with what each model predicts and what is trained in imagination ->
  Where to read more: the world-models page.
- Main figure viz-dyna (Computed, live): the S&B Dyna maze; planning steps 0, 5, 50; steps-per-episode curves.
- Small SVG viz-mbfamily (Sourced, static): the three designs side by side.
- Equations: none new (Dyna's planning step is the Q-learning update on remembered transitions).
- Exercise: NumPy: Dyna-Q on the maze. PyTorch: fit an MLP dynamics model of CartPole from random data and roll it
  out against the real environment to see errors compound.
- Sources: S&B ch. 8 (Example 8.1), 1803.10122v4, 1912.01603v3, 2301.04104v2, 1911.08265v2, 1712.01815v1.

### 15. ch-llm: RL for language models (short bridge to the post-training page)

- Learner's question: "How do these methods train chat models and reasoning models?"
- The one intuition: an LLM is a policy. The state is the prompt plus tokens so far, each token is an action, and the
  reward usually arrives once, at the end, from a reward model or a checker. RLHF runs PPO with a KL penalty that keeps
  the model near its starting point; GRPO drops the critic and uses the mean reward of a group of answers to the same
  prompt as the baseline (REINFORCE with a baseline, chapter 10); DPO skips RL by turning preferences into a
  classification loss.
- Steps: The mapping table (state, action, episode, reward, policy, value, gamma = 1) -> Rewards from preferences
  (Bradley-Terry) -> PPO for RLHF and its KL penalty (the trust region of chapter 12 again) -> GRPO -> Verifiable rewards
  and reasoning (DeepSeek-R1) -> DPO in one paragraph -> Where to read more: the post-training page.
- Main figure viz-grpo (Toy, live): one prompt, a group of 8 sampled answers, their rewards, the group mean and spread,
  the advantages, and how each answer's token probabilities move.
- Second figure viz-llmmodels (Sourced, static SVG): which networks are trained, frozen or absent in PPO-RLHF, GRPO and
  DPO.
- Equations: InstructGPT objective (InstructGPT Eq. 2), GRPO advantage and objective (DeepSeekMath, as numbered), DPO
  loss (DPO Eq. 7). Shown, glossed, and linked rather than re-derived.
- Exercise: NumPy: GRPO on a toy sequence policy (bit strings, reward from a checker). PyTorch: the same toy with a
  tiny autoregressive model, GRPO plus a KL penalty to a frozen reference; print reward and KL over training.
- Sources: 1706.03741v4, 2203.02155v1, 2305.18290v3, 2402.03300v3, 2501.12948v2.

### Guide ch-guide: RL in practice

- Which algorithm (a decision tree: discrete vs continuous, simulator cost, on- vs off-policy, offline data).
- What to plot (return, value loss, entropy, approximate KL, clip fraction), seeds and spread (Henderson et al. 2017),
  normalization (observations, rewards, advantages), reward design and reward hacking, starting hyperparameters from the
  papers.
- The three problems revisited: which chapter gave which tool.
- Timeline and family tree from Bellman to GRPO.
- Where to go deeper: S&B chapters per topic, Spinning Up, CS285, CleanRL files, and each paper with section and
  equation pointers.
- Sources: 1709.06560 (Henderson et al.), 2006.05990v1, 2005.12729v1, plus the chapters above.

## Extraction groups (step 1)

| Group | Sources | Chapters |
|---|---|---|
| A | S&B ch. 1-4 | score, bandits, mdp, bellman, dp |
| B | S&B ch. 5-8 + Sutton 1988 + Watkins-Dayan 1992 | mctd, qlearn, model (Dyna) |
| C | S&B ch. 9-13 + Williams 1992 + Sutton et al. 2000 + 1812.02648 | approx, pg, ac |
| D | DQN 2013 + Nature 2015 + 1509.06461 + 1511.06581 + 1511.05952 + 1707.06887 + 1710.02298 | dqn |
| E | 1502.05477 + 1506.02438 + 1602.01783 + 1707.06347 + Kakade-Langford 2002 + 2005.12729 + 2006.05990 + 1709.06560 | ac, ppo, step, guide |
| F | 1509.02971 + 1802.09477 + 1801.01290 + 1812.05905 + Silver 2014 | continuous |
| G | 1803.10122 + 1912.01603 + 2301.04104 + 1911.08265 + 1712.01815 + 1706.03741 + 2203.02155 + 2305.18290 + 2402.03300 + 2501.12948 | model, llm |
