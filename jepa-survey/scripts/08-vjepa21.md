---
chapter: ch-vjepa21
title: V-JEPA 2.1, dense features
target_seconds: 120
---
V-JEPA 2 understood scenes but could not place things. [[cue:btn-dense-reset]]
On semantic segmentation its frozen features scored twenty-four where DINOv2 scored nearly fifty.
The reason is in the loss.
Only the masked tokens were graded, so the visible tokens were free to become global summaries with no obligation to stay about their own patch. [[cue:btn-dense-v2]]

V-JEPA 2.1 grades every token. [[cue:btn-dense-v21]]
The masked ones as before, [[cue:eqstep-eq-dense-1]]
plus the visible ones, each weighted by how close it sits to a hole, [[cue:eqstep-eq-dense-2]]
one over the square root of the distance. [[cue:eqstep-eq-dense-3]]
Note the target for a visible token is still the clean clip's embedding from the other encoder, so predicting it is not free.

Here is the prediction for you: why not weight every visible token at one?
Press the ablation. [[cue:btn-dense-flat]]
Because then the easiest solution is to copy.
The predictor echoes the context features, the encoder stops building a global picture, and Something-Something falls from seventy-three to fifty-one.
The weighted, warmed-up version lands at sixty-two with segmentation up by eleven points. [[cue:btn-dense-v21]]
Even so, ten points of motion understanding are lost at this step.

The second ingredient wins them back: deep self-supervision.
The encoder's outputs at four depths are fused, the predictor produces four outputs per token, and each is graded against the other encoder at the same depth.
Something-Something recovers to seventy-two, segmentation climbs to thirty-nine.
Then data, a separate image tokenizer so pictures and videos share one encoder, a two-billion-parameter model, and the long-clip cooldown: forty-eight on segmentation, seventy-eight on Something-Something.

What it bought: twenty-three points of segmentation over V-JEPA 2 with the same frozen encoder, depth estimation that edges DINOv3's seven-billion-parameter model, and nothing global lost at the two-billion size.
On the robot, with the same planner, grasping goes from sixty to seventy percent; with a longer horizon, eighty.
