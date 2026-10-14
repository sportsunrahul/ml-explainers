---
chapter: ch-lewm
title: End to end from pixels, and up the hierarchy
target_seconds: 160
---
DINO-WM needs a pretrained encoder; PLDM needs seven loss terms with six weights to tune. [[cue:btn-hier-reset]]
LeWorldModel takes LeJEPA's answer and applies it to dynamics.
An encoder and an action-conditioned predictor, trained together from raw pixels, with exactly two terms. [[cue:eqstep-eq-lewm-1]]
Predict the next embedding. [[cue:eqstep-eq-lewm-2]]
And keep the embeddings Gaussian with SIGReg.
No stop-gradient, no moving average, no pretrained weights.
Fifteen million parameters, a single GPU, a few hours.

Without the regulariser, the authors say plainly, the prediction loss alone collapses to a constant.
With it, the planner reaches ninety-six percent on push-T against DINO-WM's seventy-four, and loses on two of the four tasks.
The number they lead with is speed.
One state token instead of a hundred and ninety-six patch tokens makes a full plan take under a second where DINO-WM takes forty-seven.
The trade is the one the last chapter warned about: a single vector keeps the agent's position well, and less of the object's pose.

Now the last box of the 2022 diagram.
H-JEPA stacks these models.
Level one is a LeWorldModel whose step spans five environment steps.
Each level above encodes the level below's latent into its own, pools the actions in between into a macro-action, and predicts one stride further out.
With a stride of two, each level covers twice the time of the one below.
All levels train jointly, and gradients from the upper losses flow down through the lower encoders.

Planning runs top-down.
The top level plans a short path to the goal in its coarse latent. [[cue:btn-hier-top]]
Its first prediction becomes the subgoal for the level below, which plans toward it; the cost is measured in the upper latent. [[cue:btn-hier-mid]]
And level one returns primitive actions. [[cue:btn-hier-low]]
Before you press flat, guess why planning eight fine steps at once is harder than two coarse ones. [[cue:btn-hier-flat]]
The fan of candidates is wide, and the cost signal is weak far from the goal.

The numbers: on visual AntMaze with goals seventy-five steps away, a flat model succeeds eighteen percent of the time, a three-level hierarchy seventy-three.
The control that isolates the idea is a hierarchy whose upper encoders are identity maps, so every level plans in the same latent: about half as good.
And what the upper levels learn to forget is exactly what the sketch predicted: the ant's leg pose and a moving distractor become unrecoverable up the stack, while the agent's position stays.
One caveat from the same paper: on real robot video, where the scene changes every episode, SIGReg alone lets the encoder keep the scene and drop the arm, the slow-feature trap again, and an inverse-dynamics term had to be added.
