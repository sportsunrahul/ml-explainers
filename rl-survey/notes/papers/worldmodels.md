# World Models (Ha and Schmidhuber 2018)

## 1. Identity
- Title: World Models. arXiv 1803.10122, version read v4 (2018-05-09); v1 2018-03-27.
- Authors: David Ha (first), Jurgen Schmidhuber (last; 2 authors). PDF 21 pages, ICML-style template.
- Sections: 2 Agent Model (2.1 VAE (V), 2.2 MDN-RNN (M), 2.3 Controller (C), 2.4 Putting V, M, C Together),
  3 Car Racing Experiment (3.1-3.4), 4 VizDoom Experiment (4.1-4.5), 5 Iterative Training Procedure,
  6 Related Work, 7 Discussion, Appendix A.1-A.5.

## 2. Core problem
Learn a generative model of an RL environment from raw pixels (no reward), then learn a policy from the model's
features, and ask whether the agent can train entirely inside its own "dream" and transfer the policy back.

## 3. What prior methods did and why that hurt (per the source)
- Deep RL on CarRacing-v0 needed frame preprocessing (edge detection) and frame stacking and reached 591-652
  average (Sec. 3.3, p6).
- Earlier work that learned dynamics models did not fully replace the real environment; deterministic models
  are "easily exploitable by the agent if it is not perfect"; Bayesian models (PILCO) help "to some extent,
  however, they do not fully solve the problem" (Sec. 4.5, p9).

## 4. Key idea
Put almost all capacity in a world model (V compresses each frame, M predicts the next latent) trained by
backprop on random-policy data, and make the controller C tiny (a linear map, 867 parameters) trained by an
evolution strategy (CMA-ES). Because M outputs a distribution with a temperature tau, M can serve as a
stochastic simulator ("dream") to train C in.

## 5. Mechanism
Agent loop (Sec. 2.4 pseudocode, p4): `z = vae.encode(obs); a = controller.action([z, h]);
obs, reward, done = env.step(a); h = rnn.forward([a, z, h])`.

CarRacing procedure (Sec. 3.2, p5), transcribed:
1. Collect 10,000 rollouts from a random policy.
2. Train VAE (V) to encode frames into z in R^32.
3. Train MDN-RNN (M) to model P(z_{t+1} | a_t, z_t, h_t).
4. Define Controller (C) as a_t = W_c [z_t h_t] + b_c.
5. Use CMA-ES to solve for a W_c and b_c that maximizes the expected cumulative reward.

Take Cover procedure (Sec. 4.2, p7), transcribed:
1. Collect 10,000 rollouts from a random policy.
2. Train VAE (V) to encode each frame into a latent vector z in R^64, and use V to convert the images collected
   from (1) into the latent space representation.
3. Train MDN-RNN (M) to model P(z_{t+1}, d_{t+1} | a_t, z_t, h_t).
4. Define Controller (C) as a_t = W_c [z_t h_t].
5. Use CMA-ES to solve for a W_c that maximizes the expected survival time inside the virtual environment.
6. Use learned policy from (5) on actual environment.

What is trained by what, and what is fixed:
- V: ConvVAE, 64x64x3 input, 4 conv + 4 deconv layers, L2 reconstruction + KL loss, 1 epoch on random data
  (App. A.1, p13-14).
- M: LSTM + mixture density output (5 Gaussians, diagonal), teacher forcing, 20 epochs (App. A.2, p14). In Doom M
  also predicts death; done = true if the predicted probability is above 50% (App. A.2).
- C: CMA-ES, population 64, each agent runs 16 rollouts, fitness = average cumulative reward of the 16 (App. A.4,
  p14). V and M are trained first and stay fixed while C evolves.
- "the world model (V and M) has no knowledge about the actual reward signals from the environment ... Only the
  Controller (C) Model has access to the reward information" (Sec. 3.1, p5).
- Training inside the dream happens only in the VizDoom Take Cover experiment (Sec. 4). In CarRacing, C is
  trained in the real environment; "Car Racing Dreams" (Sec. 3.4) only shows that trained C driving inside M.

## 6. Equations
- Eq. 1 (Sec. 2.3, p3), controller: `a_t = W_c \; [z_t \; h_t]\; + b_c`
  Source: "C is a simple single layer linear model that maps z_t and h_t directly to action a_t at each time
  step". Gloss: z_t = VAE latent of the current frame; h_t = LSTM hidden state of M; [z_t h_t] = concatenation;
  W_c, b_c = weight matrix and bias. App. A.3: tanh bounds each action (steer in [-1, 1], gas and brake in [0, 1]).
- M (unnumbered, Sec. 2.2, p3): `P(z_{t+1} \; | \; a_t, z_t, h_t)`, a mixture of Gaussians; temperature tau
  scales sampling uncertainty.
- Consistency check (computed by us, not printed): CarRacing C has 3 actions, so (32 + 256) x 3 + 3 = 867. Doom C
  sees both the cell vector c and output h of a 512-unit LSTM (App. A.3) plus z (64), one action, no bias:
  64 + 512 + 512 = 1,088. Both match the printed counts.

## 7. Config
| Setting | CarRacing-v0 | VizDoom Take Cover |
|---|---|---|
| Random rollouts for V and M | 10,000 | 10,000 |
| Frame input | 64x64x3 | 64x64x3 |
| Latent size N_z | 32 | 64 |
| LSTM hidden units | 256 | 512 |
| Mixture components | 5 | 5 |
| VAE / MDN-RNN / C parameters | 4,348,547 / 422,368 / 867 | 4,446,915 / 1,678,785 / 1,088 |
| C trained in | real environment | dream (M), tau = 1.15 |
| CMA-ES | population 64, 16 rollouts per agent (App. A.4) | same section, stated generally |
| "Solved" threshold | average reward 900 over 100 consecutive trials (Sec. 3 footnote, p4) | average survival > 750 time steps over 100 consecutive rollouts (Sec. 4.1, p7) |
| Episode cap | -- | 2100 time steps (about 60 s) |
Parameter tables (p5, p7) have no caption and so no printed table number.

## 8. Results
- Table 1 (p6), CarRacing-v0 average score: Full World Model 906 +- 21 over 100 random trials; V model only
  632 +- 251; V model with hidden layer 788 +- 141; DQN 343 +- 18; A3C (continuous) 591 +- 45; A3C (discrete)
  652 +- 10; Gym leaderboard (ceobillionaire) 838 +- 11. Text: "To our knowledge, our method is the first reported
  solution to solve this task." (Sec. 3.3, p6)
- App. A.4 (p14): "After 1800 generations, an agent was able to achieve an average score of 900.46 over 1024
  random rollouts."
- Doom: about 900 time steps in the virtual environment (Sec. 4.3, p7); about 1100 over 100 consecutive trials in
  the actual environment, "far beyond the required score of 750 time steps" (Sec. 4.4, p8).
- Table 2 (p10), virtual score / actual score by tau: 0.10: 2086 +- 140 / 193 +- 58; 0.50: 2060 +- 277 /
  196 +- 50; 1.00: 1145 +- 690 / 868 +- 511; 1.15: 918 +- 546 / 1092 +- 556; 1.30: 732 +- 269 / 753 +- 139.
  Random policy 210 +- 108; Gym leader 820 +- 58. App. A.5 (p15): at tau = 1.15 the best agent scored 959 over 1024
  rollouts in DoomRNN, and 1092 +- 556 over 100 rollouts in DoomTakeCover-v0.
- Model exploitation (Sec. 4.5, p8-9): the agent found "an adversarial policy to move around in such a way so that
  the monsters in this virtual environment governed by the M model never shoots a single fireball". At tau = 0.1,
  "Whatever policy learned inside of this dream will achieve a perfect score of 2100 most of the time, but will
  obviously fail when unleashed into the harsh reality of the actual world, underperforming even a random policy."

## 9. Limitations (stated) and later fixes
- Controller exploits model errors (Sec. 4.5); tau is a tuning knob between realism and exploitability.
- Random-policy data is enough only for simple tasks; "For more complicated tasks, an iterative training procedure
  is required" (Sec. 5, p10).
- Standalone VAE "may encode parts of the observations that are not relevant to a task" (Sec. 7, p12-13).
- Limited LSTM capacity; catastrophic forgetting (Sec. 7).
- Sec. 7 suggests training "in the dream directly using the backpropagation algorithm"; Dreamer (1912.01603) does
  this, and cites prior work's "derivative-free optimization" as a gap (Dreamer Sec. 1).

## 10. Outline feedback
- viz-mbfamily: for World Models, "trained in imagination" is true only for VizDoom Take Cover. The headline
  CarRacing 906 comes from a C trained in the real environment. Say so in the panel.
- The C optimizer is CMA-ES (an evolution strategy), not a gradient RL method. Name it, since the page teaches
  gradient methods.
- "A wrong model plans wrong" step: Table 2 at tau = 0.10 (virtual 2086, actual 193, below random 210) is a sourced,
  numeric example of exploiting a learned model. It pairs well with the Dyna maze.
- Notation: Sec. 5 writes P(x_{t+1}, r_{t+1}, ... | x_t, a_t, h_t), so the reward after a_t is r_{t+1} (S&B style).
