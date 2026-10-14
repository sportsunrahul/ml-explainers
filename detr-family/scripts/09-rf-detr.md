---
chapter: ch-rf-detr
title: RF-DETR: train once, pick your latency
target_seconds: 105
---
Real-time detectors usually come as a ladder of sizes. [[cue:btn-budget-reset]]
Nano, small, medium, large, each designed by hand and trained separately.
RF-DETR asks: why?

It takes a DETR with a pretrained DINOv2 backbone, and trains one network that contains many models.
At every training step it picks a random patch size, a random image resolution, and a random number of attention windows.
So the same weights learn to work under all of those settings.
The number of decoder layers and queries is cut later, at inference.

Once training is done, a search tries thousands of configurations, and keeps the best trade-offs between accuracy and speed.
Picking a model is now just picking a latency budget.

Give it three milliseconds.
You get the nano model: 48 AP at 2.3 milliseconds.
That's 5.3 AP more than D-FINE's nano, at a similar speed.

Give it seven milliseconds. [[cue:btn-budget-7]]
Resolution and decoder depth grow, while the patch size stays put.
Now eighteen. [[cue:btn-budget-18]]
This one crosses 60: 60.1 AP at 17.2 milliseconds, the first real-time detector above that line.

But it doesn't win everywhere.
Turn the baselines on, and you can see D-FINE ahead at the medium, large and extra large sizes. [[cue:btn-base-on]]

And notice what never changed, all the way from 2020.
A fixed set of queries, one-to-one matching, and no NMS.
Everything else, the attention, the queries, the training signal and the backbone, was rebuilt around that one idea.
