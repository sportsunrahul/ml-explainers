---
chapter: ch-planning
title: Planning in latent space
target_seconds: 150
---
A world model is a function that takes the current state and an action and predicts the next state. [[cue:btn-mpc-reset]]
With one, you can act without ever being told what to do.
Imagine many sequences of actions, look at where each ends, keep the one that ends closest to where you want to be.
No reward, no policy, no trial and error in the real world.

The toy runs the loop.
A two-room world with a wall and a doorway; blue is the agent, yellow is the goal.
Guess first: with a horizon of six steps, can the planner get around the wall at all?
Plan. [[cue:btn-mpc-plan]]
Eighty imagined futures, the best eight kept, the distribution refit, four times.
Execute one step, look again, replan. [[cue:btn-mpc-step]]
And run. [[cue:btn-mpc-run]]
It finds the door, because each replanning step only has to make progress, not solve the whole route.

Three designs did this with JEPA-style models, and they differ in what the state is.
DINO-WM freezes a pretrained DINOv2 encoder and uses its grid of patch features as the state, training only a nineteen-million-parameter predictor on top.
It reaches ninety percent on push-T, where DreamerV3 reaches thirty.
Its most telling table swaps the encoder: a single-vector summary of the same image halves the success rate.
Spatial detail is what planning needs, and a global summary throws it away.
And gradient descent as the planner instead of sampling collapses to twenty-two percent: the landscape over actions is not smooth enough to follow downhill.

PLDM goes the other way: train the encoder and the predictor together from sixty-four-pixel frames, two million parameters, with explicit anti-collapse terms, and plan.
On good data it reaches ninety-eight percent where a pixel-reconstruction world model gets twenty-six.
It generalises to unseen maze layouts from as few as five training maps.
And it is honest about where it loses: without door crossings in the data, a goal-conditioned RL method stitches better.

The third is the contrast.
Navigation World Models predict pixels, through a diffusion model, and score a plan by how much the imagined final frame looks like the goal image.
It works, and it can imagine places it has never seen, but each plan takes thirty seconds.
V-JEPA 2.1's latent planner later matched its error ten times faster.

Every number here points the same way as chapter one did.
Score plans by a distance in embedding space; keep where things are, not only what they are.
What none of these settle is whether the encoder itself can be trained for planning, from pixels, without a pretrained backbone and without the asymmetry tricks.
