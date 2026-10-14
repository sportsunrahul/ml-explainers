---
chapter: ch-vjepa2
title: V-JEPA 2, scale, then act
target_seconds: 200
---
V-JEPA 2 is two papers in one. [[cue:btn-scale-reset]]
The first scales the video recipe until a frozen encoder can anticipate what a person will do next.
The second freezes that encoder, bolts on a small predictor that reads robot actions, trains it on under sixty-two hours of robot video, and plans with it on a real arm.

Four ways to scale, and here are the stairs.
Start from V-JEPA at eighty-four point two, averaged over six tasks. [[cue:btn-scale-0]]
Twenty-two million videos instead of two million: eighty-five point two. [[cue:btn-scale-1]]
A billion-parameter encoder: eighty-six point seven. [[cue:btn-scale-2]]
Training nearly three times longer, with the learning rate held flat until the probes stop improving: eighty-seven point five. [[cue:btn-scale-3]]
And the trick that makes the last step affordable: only the final twelve thousand steps run at sixty-four frames and three eighty-four pixels. [[cue:btn-scale-4]]
Eighty-eight point two, at about an eighth of the cost of training that size throughout.

The frozen encoder now reads motion better than anything before it, and on Epic-Kitchens it predicts what a cook will do one second from now with a forty-four percent relative gain over the best prior model.
One claim card worth a pause: a video JEPA can be read as a surprise meter.
Run it over a clip, measure the prediction error, and clips where an object vanishes behind a wall surprise it more.
Ninety-eight percent on IntPhys, with no physics training at all, where pixel models and multimodal language models sit near chance.
That was shown on the 2024 V-JEPA, not on V-JEPA 2.

Now the second paper.
Freeze the encoder and throw away its mask-filling predictor. [[cue:btn-acflow-reset]]
In its place, a new predictor of three hundred million parameters, trained on twenty-three thousand robot trajectories, successes and failures alike, no labels, no rewards.
Each frame goes through the frozen encoder on its own. [[cue:btn-acflow-clip]] [[cue:btn-acflow-encode]]
The arm's state and its action at every step become tokens. [[cue:btn-acflow-tokens]]
And the predictor rolls time forward: given everything up to step k, predict the encoder's feature map of frame k plus one. [[cue:btn-acflow-predict]]
Averaged over fifteen steps with true inputs, [[cue:eqstep-eq-ac-1]]
the guess, [[cue:eqstep-eq-ac-2]]
against the frozen encoder's features of the next frame. [[cue:eqstep-eq-ac-3]]
Plus a rollout term where the predictor feeds itself for two steps. [[cue:btn-acflow-loss]]
Guess: does any gradient reach the encoder? [[cue:btn-acflow-backward]]
None.
Both sides of the loss are constants from a frozen network; only the predictor moves. [[cue:btn-acflow-update]]

To act, the robot is given a goal image.
Encode the current frame and the goal; then search for the action whose predicted next feature map is closest to the goal's.
The search is the cross-entropy method, and the toy runs the real algorithm. [[cue:btn-planner-reset]]
Sample sixty candidate action sequences. [[cue:btn-planner-sample]]
Guess which survive, then refine: keep the best six, refit the distribution, sample again. [[cue:btn-planner-refine]]
And again. [[cue:btn-planner-refine]]
Execute the first action, look, and replan. [[cue:btn-planner-step]]
The real system uses eight hundred samples, a horizon of one, and takes sixteen seconds per decision on a single consumer GPU.

In two labs the model had never seen: reaching, every time; grasping a cup, sixty-five percent; pick-and-place of the cup, eighty.
A behaviour-cloning policy trained on a million trajectories grasps the cup fifteen percent of the time in the same setting, and a planner on a pixel-generating video model takes four minutes per decision and never grasps it.
The limits are honest too: the actions rotate with the camera angle, and the goal must be given as an image.
