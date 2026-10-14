# Fact check: ch-sam2 (explainer.src.html L2162-2448, inline viz-sam2flow script, SCENES.sam2)

Sources: sources/sam2/latex/sam2.1_arxiv.tex (SAM 2 v2), tab/*.tex; stm/latex/final.tex; xmem/latex/*.tex;
cutie/latex/*.tex; hiera/latex/hiera_arxiv.tex; abs.html dates.

## (a) WRONG

None found. Every number checked holds at its loc, and no sentence states something the source contradicts outright.

## (b) MISLEADING / over-claimed / unsupported

1. ch-sam2, archflow stage "7. Where the gradient goes": "In the 16-frame fine-tune, Hiera is frozen ... the FPN
   still learns." The JS readout repeats it ("the FPN and everything after it learn"). The paper says only "freeze the
   image encoder" (sam2.1_arxiv.tex:1078). App. D.1 describes the FPN inside the "Image encoder" paragraph
   (sam2.1_arxiv.tex:954), so the paper's "image encoder" most plausibly includes the FPN. The page states the FPN
   detail as fact with no "our reading" label. Fix: say "the image encoder is frozen" and either drop the FPN clause or
   label it our reading.
2. ch-sam2, "What memory buys", third bullet heading "The pointers matter, the recurrence does not." The paper says
   object pointers "do not enhance average performance across the 9 zero-shot datasets" and help on SA-V val and
   LVOSv2 (sam2.1_arxiv.tex:898). In Table 11 the no-pointer row is best on the 9 zero-shot average (70.9 vs 70.7)
   and on MOSE dev (73.1 vs 73.0) (sam2.1_arxiv.tex:909-911). The GRU row (65.3) is above the no-pointer row (64.5)
   on SA-V val, so "the recurrence does not [matter]" is also a simplification. Fix: "Pointers help on SA-V val and
   LVOSv2, not on the 9 zero-shot average."
3. Same bullet: 68.3 / 64.5 / 65.3 / 69.0 / 0.93 are from the ablation setup ("a smaller model setup with 512 input
   resolution", sam2.1_arxiv.tex:708), not the released 1024 model. The page does not say so; a reader can compare
   68.3 with the 77.0 / 76.8 Table 6 numbers in the previous bullet. Add "(ablation setup, 512 px)".
4. ch-sam2, SA-V paragraph: "That is 53x more masks than any earlier video segmentation dataset." The 53x counts the
   automatic masklets; without them it is 15x (sam2.1_arxiv.tex:399, 450, macro L154). The paper compares with VOS
   datasets in Table 3 ("the largest VOS dataset"), not all video segmentation datasets. Add "with automatic masklets
   (15x manual only)" and say "video object segmentation dataset".
5. ch-sam2, "Fewer clicks" bullet: "SAM 2 reaches 64.7 J&F". Table 4 reports J&F except on VOST, where it reports J
   (sam2.1_arxiv.tex:535). Minor; "average accuracy" is the paper's wording (Table 4 caption, L528).
6. ch-sam2, claim card "Claim": "without making them worse". The paper says "comparable quality" and "better
   alignment" than Phase 2 (sam2.1_arxiv.tex:230, 375); 10.9% of Phase 3 masks fall below IoU 0.75 vs Phase 1
   (Table 1, L362). The card tests the claim honestly, so this is wording only. Prefer "at comparable quality".
7. ch-sam2, "Two details carry a lot": "a correction click must become a memory that outranks them". The paper keeps
   prompted memories in a separate FIFO of up to M frames without temporal codes (sam2.1_arxiv.tex:319, 323); it
   says nothing about ranking. Label as our reading or say "is kept in the prompted-frame queue".

Checked and fine (no action): the occlusion head (L300, L968); prediction of multiple masks and propagation of the
highest-IoU one (L298); 6 recent memories with temporal codes, prompted without (L323); 64-dim memory, 256-dim
pointer split into 4 x 64 (L973); stride 16+32 to FPN, 4+8 skips (L954); 2d-RoPE in memory attention (L956);
learned occlusion embedding (L968); loss ratio and supervision rules (L1080); IoU on all masks, 7 correction
clicks (L988); AdamW, ~150k, 6e-5 / 3.0e-4, batch 256 (Table 12b, L1047-1062); 16-frame 50k at half lr
(L1078); 256 A100 x 108 h for "the released SAM 2" (L1782); up to 2 prompted frames of 8, GT mask / click / box
(L326); "The paper does not state how many prompted memories are kept" is true (M has no value, L319). The
limitations quotes are verbatim (App. C, L939, L941). Chapter cross-references (ch. 4, 7, 12) point to the right
chapters.

## (c) Honesty labels

- viz-sam2: data-toy, figcaption starts "Toy model", SCENES.sam2 calls badge('toy', ...). Consistent. The staged
  drift after the occlusion is disclosed in the caption.
- viz-sam2flow (ArchFlow): data-computed, badge "Computed", figcaption "Computed: ..." with the 64x64 memory size and
  1/4-resolution logits labelled our reading. Hiera stage widths and depths in the JS (T [96..768] [1-2-7-2], S
  [1-2-11-2], B+ [112..896] [2-3-16-3], L [144..1152] [2-6-36-4]) match hiera_arxiv.tex:391-395. Spatial sizes
  256/128/64/32 follow from a 1024 input and strides 4-32. Consistent.
- Toy readout on frame 1: "segmented exactly as SAM would" is an approximation inside a toy-labelled figure; no
  action needed.

## (d) Equations

- eq-memread: labelled "Standard scaled dot-product cross-attention (our notation)". The paper prints no equation
  and describes the block in words (Sec. 4, sam2.1_arxiv.tex:293). The transcription is faithful. It omits the
  residual, the output projection and the self-attention / MLP sublayers, which the eqno text names. "Sec. 4" is the
  correct section number (Model is \section 4, L279). Fine.

## (f) What was checked

All 68 claim spans in ch-sam2 (64 unique ids) against their locs:
- video-memory.json: stm-every-n (final.tex:560-561), xmem-stores (introduction.tex:37-38), xmem-34k
  (methods.tex:185). The prose around them holds: STM matches every query pixel with every memory pixel
  (final.tex:158, 397); XMem has sensory (every frame), working and long-term prototype memory (methods.tex:11, 24,
  133); prior methods "struggle to handle videos longer than a minute" (introduction.tex:10); Cutie reads object-level
  memory (00-abstract.tex:2); all three start from a first-frame annotation. Dates: STM 2019-04-01, XMem 2022-07-14,
  Cutie 2023-10-19 (abs.html).
- sam2-*: architecture (L284-326, L950-976), Table 5 (L566-586), Sec. 7 FPS (L612), Table 1 (L355-362), controlled
  experiment (L1234), SA-V macros and text (L139-168, L399-450, L1110), Table 4 (L520-528), Table 6
  (tab/tab-mask2masklet-results-oss.tex), Table 9c (L773-774), Table 11 (L907-911), Table 12 and App. D.2
  (L984-1082), model card (L1782); sam2-hiera-bplus-channels (hiera_arxiv.tex:394).
- No extra-f* ids are used in this section. No uncertain or code_only claims are used in this section.
- Unsourced prose: Hiera is MAE-pretrained (L291), streaming encoder run once per frame (L291), prompt encoder
  identical to SAM (L296), image segmentation as a one-frame video (task section). No errors beyond (b).
