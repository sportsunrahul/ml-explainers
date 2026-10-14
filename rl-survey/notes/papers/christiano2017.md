# Deep RL from Human Preferences (Christiano et al. 2017)

## 1. Identity
- Title: Deep reinforcement learning from human preferences. arXiv 1706.03741, version read v4 (2023-02-17); v1
  2017-06-12. NIPS 2017 paper (v4 is a later arXiv update).
- Authors: Paul Christiano (first), Dario Amodei (last; 6 authors). PDF 17 pages.
- Sections: 1 Introduction (1.1 Related Work), 2 Preliminaries and Method (2.1 Setting and Goal, 2.2 Our Method:
  2.2.1 Optimizing the Policy, 2.2.2 Preference Elicitation, 2.2.3 Fitting the Reward Function, 2.2.4 Selecting
  Queries), 3 Experimental Results (3.1 tasks with unobserved rewards, 3.2 Novel behaviors, 3.3 Ablation Studies),
  4 Discussion and Conclusions, App. A Experimental Details, App. B Instructions Provided to Contractors.

## 2. Core problem
Many goals are hard to write as a reward function. Can an RL agent learn a task from a human's comparisons of
short clips, with so little feedback that human time is affordable?

## 3. What prior methods did and why that hurt (per the source)
Hand-written rewards are unavailable for complex goals; learning from demonstrations needs demonstrators; direct
human feedback as reward would need far too much human time ("we need to decrease the amount of feedback required
by several orders of magnitude", Sec. 1).

## 4. Key idea
Fit a reward model r_hat to pairwise human preferences over trajectory segments (a Bradley-Terry model), and train
the policy with ordinary RL on r_hat, while collecting new comparisons online.

## 5. Mechanism (Sec. 2.2, p4), transcribed
Maintain a policy pi : O -> A and a reward estimate r_hat : O x A -> R, both deep networks, updated by three
asynchronous processes:
1. The policy pi interacts with the environment to produce trajectories {tau^1, ..., tau^i}. Its parameters are
   updated by a traditional RL algorithm to maximize the sum of predicted rewards r_t = r_hat(o_t, a_t).
2. Select pairs of segments (sigma^1, sigma^2) from those trajectories and send them to a human for comparison.
3. Optimize r_hat by supervised learning to fit the comparisons collected so far.
Details: A2C for Atari, TRPO for MuJoCo (Sec. 2.2.1); predicted rewards normalized to zero mean and constant std;
clips are 1 to 2 seconds (Sec. 2.2.2); the human may prefer one, mark equal (mu uniform), or mark incomparable
(dropped). Modifications (Sec. 2.2.3): an ensemble of predictors on bootstrap resamples; 1/e of data held out for
validation; l2 regularization tuned to keep validation loss 1.1-1.5 times training loss; "we assume there is a 10%
chance that the human responds uniformly at random". Queries are chosen where ensemble predictions vary most
(Sec. 2.2.4).

## 6. Equations
- Eq. 1 (Sec. 2.2.3, p5), preference model:
  `\hat{P}\bdelims*{\sigma^1 \succ \sigma^2} = \frac{\exp{\sum \rest\of{o^1_t, a^1_t}}}{\exp{\sum\rest\of{o^1_t, a^1_t}} + \exp{\sum\rest\of{o^2_t, a^2_t}}}`
  Macros: \rest = \hat{r}; \of{x} = (x); \bdelims*{x} = [x]; \db = \mathcal{D}.
  Source: the human's probability of preferring a segment "depends exponentially on the value of the latent reward
  summed over the length of the clip"; footnote: Eq. 1 "does not use discounting". Gloss: sigma^1, sigma^2 = two clips;
  r_hat(o, a) = predicted reward per step; the sum runs over each clip's steps.
- Loss (unnumbered display, Sec. 2.2.3, p5):
  `\mathrm{loss}\of{\rest} = -\sum_{\of{\sigma^1, \sigma^2, \mu} \in \db} \mu\of{1} \log \hat{P}\bdelims*{\sigma^1 \succ \sigma^2} + \mu\of{2} \log \hat{P}\bdelims*{\sigma^2 \succ \sigma^1}`
  Source: "This follows the Bradley-Terry model for estimating score functions from pairwise preferences". Gloss: mu =
  the human label as a distribution over {1, 2}; cross-entropy between label and Eq. 1. The paper compares it to Elo.
- Notation: rewards are r_t = r_hat(o_t, a_t), indexed with the action (deep-RL r_t for S&B's R_{t+1}).

## 7. Measurement setup (Sec. 3.1)
- Feedback from contractors given a 1-2 sentence task description; each answered "several hundred to several
  thousand" comparisons; average query took 3-5 seconds; experiments with real human feedback "required between 30
  minutes and 5 hours of human time" (p6).
- Synthetic oracle: prefers the clip with higher true reward. Baseline: RL on the true reward.
- MuJoCo: eight simulated robotics tasks; 700 human queries vs 350, 700, 1400 synthetic queries; curves average 5
  runs, human feedback a single run (Fig. 2, p6).
- Atari: seven games (those of Mnih et al. 2013); 5,500 human queries; curves average 3 runs (Fig. 3, p7).

## 8. Results
- Abstract (p1): solves "complex RL tasks without access to the reward function, including Atari games and simulated
  robot locomotion, while providing feedback on less than 1% of our agent's interactions with the environment."
- MuJoCo (Sec. 3.1.1, p6-7): "With 700 labels we are able to nearly match reinforcement learning on all of these
  tasks." By 1400 labels it "performs slightly better than if it had simply been given the true reward".
- Atari (Sec. 3.1.2, p7): harder; matches or exceeds RL on some games (BeamRider, Pong), fails on Qbert with human
  labels; on Enduro human shaping beats A3C.
- Novel behaviors (Sec. 3.2, p8-9): "The Hopper robot performing a sequence of backflips (see Figure 4). This
  behavior was trained using 900 queries in less than an hour." Half-Cheetah on one leg: 800 queries "in under an
  hour". Enduro keeping alongside cars: about 1,300 queries and 4 million frames. These used feedback from the authors.
- Sec. 4 (p10): learning a separate reward model makes it "possible to reduce the interaction complexity by roughly 3
  orders of magnitude."

## 9. Limitations (stated) and later fixes
- Learned rewards are "less stable and higher variance" than true rewards (Sec. 3.1.1). Online feedback "prevents it
  from exploiting weaknesses of the learned reward function" (Sec. 1). Ablation (Sec. 3.3, p9-10): with offline
  reward predictor training, "on Pong offline training sometimes leads our agent to avoid losing points but not to
  score points; this can result in extremely long volleys" (a reward-hacking example for the guide). Query selection
  by ensemble disagreement is "a crude approximation" (Sec. 2.2.4).
- InstructGPT (2203.02155) applies the same pairwise reward-model idea to language models (its Related work cites
  this paper as the origin of RLHF).

## 10. Outline feedback
- ch-llm "Rewards from preferences (Bradley-Terry)": cite "Eq. 1" here for the preference model; the loss is
  unnumbered (cite Sec. 2.2.3). DPO Eq. 1 is the same model written for (prompt, response) pairs.
- The "less than 1%" claim is about the fraction of the agent's interactions that received feedback, not of time
  steps labelled by the reward model; quote it exactly.
