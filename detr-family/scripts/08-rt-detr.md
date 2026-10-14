---
chapter: ch-rt-detr
title: RT-DETR: DETRs beat YOLOs once you count NMS
target_seconds: 100
---
By now, DETRs trained fast. [[cue:btn-thr-0]]
But real-time detection still belonged to YOLO.
RT-DETR starts by pointing at something odd in the comparison.
YOLO speed numbers usually leave out NMS.

And NMS time isn't a constant.
It depends on how many boxes survive the confidence threshold.
Look at the lowest threshold.
Thousands of boxes go into NMS, and for YOLOv8 the NMS step alone takes 2.36 milliseconds.

Raise the threshold. [[cue:btn-thr-2]]
Fewer boxes, less NMS time. [[cue:btn-thr-3]]
But accuracy slips too, from 52.9 down to 51.2 AP.
So a YOLO trades speed against accuracy, through a hand-tuned knob.
A set predictor doesn't have that knob at all.

RT-DETR then attacks its own remaining cost, the encoder.
It runs self-attention only on the coarsest feature map, where each cell already stands for a whole object part.
The finer scales are fused with cheap convolutions instead.
It picks its starting queries so that a high class score also means a well-placed box.
And because every decoder layer is trained to predict, you can drop layers at inference to go faster, with no retraining.

Here's the result, measured end to end, with NMS included for the YOLOs.
RT-DETR with a ResNet-50 gets 53.1 AP at 108 frames per second.
YOLOv8 large gets 52.9 AP at 71.
Barely more accurate, and about half again as fast.
