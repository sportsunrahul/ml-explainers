# Audio QA report (voice A)

Model revisions: {'base': 'fd4b254389122332181a7c3db7f27e918eec64e3'}. Median rate 2.83 words/s; speaker-similarity floor 0.65.

- Sentences: 199; retried: 21; still flagged after up to 3 retries: 12.
- Mean WER of chosen takes: 0.009.

## Still flagged (listen to these)

- **ch-matching #9**: rate=4.76wps vs median 2.83
  - script: Look at the grid.
  - heard:  Look at the grid.
  - file: out/A/wav/ch-matching-9.wav
- **ch-queries #14**: speaker sim 0.49
  - script: The paper checks this for real.
  - heard:  The paper checks this for real.
  - file: out/A/wav/ch-queries-14.wav
- **ch-deformable #18**: wer=0.18, rate=1.56wps vs median 2.83
  - script: 43.8 AP in 50 epochs, against DETR's 42 in 500.
  - heard:  43.8 AP in 50 epochs against DETRs 42 in 500.
  - file: out/A/wav/ch-deformable-18.wav
- **ch-anchor-queries #17**: wer=0.17
  - script: For box coordinates between zero and one, that makes attention almost flat.
  - heard:  For box coordinates between 0 and 1, that makes a tension almost flat.
  - file: out/A/wav/ch-anchor-queries-17.wav
- **ch-anchor-queries #18**: rate=1.34wps vs median 2.83
  - script: DAB-DETR uses twenty instead.
  - heard:  DAB-DETR uses 20 instead.
  - file: out/A/wav/ch-anchor-queries-18.wav
- **ch-denoising #21**: wer=0.27
  - script: And the four-scale DINO reaches 49 AP in only 12 epochs.
  - heard:  And the 4-scale DENO reaches 49 AP in only 12 epochs.
  - file: out/A/wav/ch-denoising-21.wav
- **ch-rt-detr #0**: rate=1.77wps vs median 2.83
  - script: By now, DETRs trained fast.
  - heard:  By now, DETRs trained fast.
  - file: out/A/wav/ch-rt-detr-0.wav
- **ch-rt-detr #9**: rate=1.84wps vs median 2.83
  - script: Fewer boxes, less NMS time.
  - heard:  Fewer boxes, less NMS time.
  - file: out/A/wav/ch-rt-detr-9.wav
- **ch-rt-detr #18**: wer=0.21
  - script: Here's the result, measured end to end, with NMS included for the YOLOs.
  - heard:  Here's the result measured end-to-end with NMS included for the YOLOs.
  - file: out/A/wav/ch-rt-detr-18.wav
- **ch-rt-detr #19**: rate=1.62wps vs median 2.83
  - script: RT-DETR with a ResNet-50 gets 53.1 AP at 108 frames per second.
  - heard:  RT-DETR with a ResNet-50 gets 53.1 AP at 108 frames per second.
  - file: out/A/wav/ch-rt-detr-19.wav
- **ch-rt-detr #20**: rate=1.59wps vs median 2.83
  - script: YOLOv8 large gets 52.9 AP at 71.
  - heard:  YOLOv8 large gets 52.9 AP at 71.
  - file: out/A/wav/ch-rt-detr-20.wav
- **ch-rf-detr #2**: rate=0.93wps vs median 2.83
  - script: RF-DETR asks: why?
  - heard:  RF-DETR asks, Why?
  - file: out/A/wav/ch-rf-detr-2.wav

## Retried and fixed

- ch-set-problem #0: first take flagged (rate=6.10wps vs median 2.83); fixed on seed 2235.
- ch-set-problem #4: first take flagged (rate=4.44wps vs median 2.83); fixed on seed 2239.
- ch-set-problem #20: first take flagged (internal silence 0.86s); fixed on seed 2255.
- ch-queries #6: first take flagged (short-sentence errors=2); fixed on seed 4241.
- ch-deformable #0: first take flagged (speaker sim 0.64); fixed on seed 6235.
- ch-deformable #19: first take flagged (rate=1.81wps vs median 2.83); fixed on seed 6254.
- ch-anchor-queries #1: first take flagged (speaker sim 0.62); fixed on seed 7236.
- ch-denoising #13: first take flagged (rate=1.78wps vs median 2.83); fixed on seed 8249.
- ch-rf-detr #15: first take flagged (rate=1.82wps vs median 2.83); fixed on seed 10251.
