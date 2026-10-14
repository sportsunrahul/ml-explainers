# Fact-check B (llava15, qwenvl, qwen2vl, qwen3vl, florence2, idefics2, molmo, chameleon, meta, page)

Method: every value was compared to the compiled PDF text (pdftotext) and to the LaTeX (checking for %-commented and
\begin{comment} blocks). Section and table numbers below are the ones printed in the compiled PDF, not LaTeX labels.
Versions all match sources/<key>/abs.html. All 16 meta.json dates match the "Submitted on" v1 date in abs.html.
page.json (CLIP abstract sentence) is verbatim, sentence 2 of the abstract.

Headline: no claim VALUE was found numerically wrong. Problems are (a) two claims that misdescribe a rendered table as
commented-out, (b) one unsupported architecture qualifier, and (c) systematic wrong section/table numbers in loc
(Qwen-VL, Qwen2-VL, Qwen3-VL, LLaVA-1.5, Florence-2, Molmo, Chameleon).

## A. WRONG / should-be-uncertain

| id | problem | what the source says | exact location |
|---|---|---|---|
| llava15-global-ctx | loc says "commented block in appendix source; not in published text" and uncertain=true. It IS in the published paper. Values are correct (patch only 62.9/1425.8/31.9; +global 63.8/1497.5/35.1). Set uncertain=false, loc = main text. | Table with "high-res patch only" and "+global context" rows is rendered; paragraph "Global context" says the ablation on a 7B model boosts all three benchmarks. (Only the table CAPTION is commented out.) | Sec. 4.2 Results, paragraph "Global context" (sec/4_eval.tex l.110-125; PDF text shows the rows) |
| idefics2-connector-ablation | loc says "inside a comment block in the LaTeX" and uncertain=true. The LaTeX has a commented duplicate followed by a LIVE identical table, so it is rendered. Values correct (44.5 / 51.8 / 60.3). Set uncertain=false. | Table "Ablation on the modality projection": Linear Projection 44.5, Mapping Network 51.8, Perceiver 60.3; text: "perceiver resampler significantly outperforms these two options" | Appendix, "Comparing various pooling strategies", Table 11 (appendix.tex l.~100-115 live copy) |
| idefics2-captions | "(cross-attention)" qualifier is not in the source. Values 49.8 / 52.9 are correct. Should not assert the architecture (or mark that part uncertain). | Text and caption never state which architecture the captions ablation used. | Sec. 4.1 Image-text pairs, Table 6 |

Already-uncertain claim that is worth a note: chameleon-prev-tokenizers (CM3 / CM3Leon tokenizer specs) exists only in a
\begin{comment} figure with example-image placeholders (paper.tex l.130-150). It is NOT in the PDF (CM3Leon appears only in
Related Work, without these numbers). uncertain=true is right; recommend not displaying it at all.

## B. MISLOCATED (value correct; loc wrong)

### llava15 (Approach is Sec. 3: 3.1 Preliminaries, 3.2 Response Format Prompting, 3.3 Scaling the Data and Model, 3.4 Scaling to Higher Resolutions)
| id | claimed loc | correct loc |
|---|---|---|
| llava15-connector | Sec. 3.2 | Sec. 3.3 (MLP vision-language connector) |
| llava15-resolution | Sec. 3.2 Additional scaling | Sec. 3.3 Additional scaling |
| llava15-format-prompt | Sec. 3.1 | Sec. 3.2 Response Format Prompting (plus Appendix data table) |
| llava15-format-prompt-mc | Sec. 3.2 Academic task oriented data | Sec. 3.3 |
| llava15-train-time | Sec. 3.2 Computational cost | Sec. 3.3 Computational cost |

### qwenvl (Methodology = Sec. 2: 2.1 Model Architecture, 2.2 Inputs and Outputs; Training = Sec. 3: 3.1 Pre-training, 3.2 Multi-task Pre-training, 3.3 SFT; Evaluation = Sec. 4; Appendix B data format, C hyperparameters, E extra experiments). Every section reference is shifted.
| id | claimed loc | correct loc |
|---|---|---|
| qwenvl-adapter-queries | Sec. 3.1; Hyperparameter table | Sec. 2.1; Table 8 in Appendix C |
| qwenvl-adapter-layers | Sec. 3.1 | Sec. 2.1 |
| qwenvl-patch-stride | Sec. 3.1 Visual Encoder | Sec. 2.1 |
| qwenvl-vit-init | Sec. 3.1 | Sec. 2.1 |
| qwenvl-llm-init | Sec. 3.1 | Sec. 2.1 |
| qwenvl-res-stage1 | Sec. 4.1.1; Hyperparameter table | Sec. 3.1; Appendix C Table 8 |
| qwenvl-res-stage2 | Sec. 4.1.2; Hyperparameter table | Sec. 3.2; Appendix C Table 8 |
| qwenvl-box-range | Sec. 3.2 | Sec. 2.2 Inputs and Outputs |
| qwenvl-box-format | Sec. 3.2; Appendix data format | Sec. 2.2; Appendix B.1 |
| qwenvl-box-example | Appendix C.1 | Appendix B.1 (Box B.1, Referring Grounding) |
| qwenvl-stage1-frozen | Sec. 4.1.1 | Sec. 3.1 |
| qwenvl-stage2-frozen | Sec. 4.1.2 | Sec. 3.2 |
| qwenvl-stage3-frozen | Sec. 4.2 | Sec. 3.3 |
| qwenvl-stage1-data | Sec. 4.1.1 | Sec. 3.1 (Table 2) |
| qwenvl-sft-size | Sec. 4.2 | Sec. 3.3 |
| qwenvl-flickr-cider | Sec. 5.1 | Sec. 4.1 (Table 4) |
| qwenvl-window-speed | Appendix D.3 | Appendix E.3 (Table 10) |

### qwen2vl (Experiments Sec. 3: 3.1 Compare to SOTAs, 3.2 Quantitative Results, 3.3 Ablation Study with 3.3.1 Dynamic Resolution, 3.3.2 M-RoPE, 3.3.3 Model Scaling)
| id | claimed loc | correct loc |
|---|---|---|
| qwen2vl-px-per-token | Sec. 2.1 + Sec. 3.6.1 | Sec. 2.1 + Sec. 3.3.1 |
| qwen2vl-minmax-pixels | Sec. 3.6.1 | Sec. 3.3.1 |
| qwen2vl-fixed-vs-dyn-infovqa | Sec. 3.6.1 | Sec. 3.3.1 (Table 7) |
| qwen2vl-fixed-vs-dyn-ocrbench | Sec. 3.6.1 | Sec. 3.3.1 (Table 7) |
| qwen2vl-mrope-extrap | Sec. 3.6.2 | Sec. 3.3.2 |
| qwen2vl-mrope-ablation-video | Sec. 3.6.2 | Sec. 3.3.2 (Table 8) |
| qwen2vl-mrope-ablation-mathvista | Sec. 3.6.2 | Sec. 3.3.2 (Table 8) |
| qwen2vl-docvqa | Table comparison (Sec. 3.2) | Table 2, in Sec. 3.1 |

### qwen3vl (Sec. 2: 2.1 Interleaved MRoPE, 2.2 DeepStack, 2.3 Video Timestamp)
| id | claimed loc | correct loc |
|---|---|---|
| qwen3vl-deepstack-levels | Sec. 2.3 DeepStack para 2 | Sec. 2.2 (single paragraph) |
| qwen3vl-deepstack-layer-idx | Sec. 2.3 | Sec. 2.2 |
| qwen3vl-mrope-problem | Sec. 2.2 | Sec. 2.1 |
| qwen3vl-timestamp-format | Sec. 2.4 | Sec. 2.3 |
| qwen3vl-timestamp-formats | Sec. 2.4 | Sec. 2.3 |

### florence2 (Sec. 3 Model, 4 Data Engine, 5 Dataset, 6 Experiments; PDF tables: T2 annotation stats, T4 zero-shot, T5 captioning/VQA, T6 region tasks, T10 model scaling, T11 data scaling, T12 basic components)
| id | claimed loc | correct loc |
|---|---|---|
| florence2-params | Sec. 4.1 Setup | Sec. 6.1 Setup |
| florence2-training | Sec. 4.1 Setup | Sec. 6.1 Setup |
| florence2-zs-coco-cap | Sec. 5.2 text | Sec. 6.2 text |
| florence2-fld-text-lengths | Table 3 (annotation statistics) | Table 2 (Table 3 is semantic-element stats) |
| florence2-ft-coco-det | Table 5 (region-level) | Table 6 |
| florence2-ft-refcoco | Table 5 | Table 6 |
| florence2-ft-res | Table 5 | Table 6 |
| florence2-freeze-enc | Table 7 (basic components) | Table 12 (Table 7 is COCO detection) |

### idefics2
| id | claimed loc | correct loc |
|---|---|---|
| idefics2-divergence | Sec. 3.2 para 4 | Sec. 3.2 para 3 (the "loss divergences ... aggressively lowering the learning rate" sentences precede the "12.9 points" paragraph) |

### molmo (PDF tables: T2 model ablations (a)-(f), T3 data ablations (a)-(c), T4 counting ablations (a)-(d), T5 Elo ablations, T11 pointing eval; appendix A Model Details, B Training Details, E Ablation Details)
| id | claimed loc | correct loc |
|---|---|---|
| molmo-abl-crop | Table 3d | Table 2d |
| molmo-abl-pool | Table 3f | Table 2f |
| molmo-abl-enc-full | Table 3a | Table 2a |
| molmo-abl-cap-scale | Table 4a | Table 3a |
| molmo-abl-pretrain-data | Table 4b | Table 3b |
| molmo-abl-sft | Table 4c | Table 3c |
| molmo-cnt-strategy | Table 5a | Table 4a |
| molmo-cnt-order | Table 5b | Table 4b |
| molmo-cnt-tokens | Table 5d | Table 4d |
| molmo-elo-abl | Table 6 (ablation_elo) | Table 5 (Table 6 is hyperparameters) |
| molmo-point-format | Appendix A Pointing | Appendix B.2 Fine-Tuning Task Details, paragraph "Pointing" (A is Model Details) |
| molmo-pixmocap-speech | Intro para 4 | Intro para 5 ("describe images in speech for 60 to 90 seconds"; para 4 is "High-quality multimodal data...") |

### chameleon (Sec. 4: 4.1 Prompts, 4.2 Baselines and Evaluations with 4.2.1 Absolute and 4.2.2 Relative, 4.3 Inter-annotator Agreement)
| id | claimed loc | correct loc |
|---|---|---|
| chameleon-he-headline | Sec. 4.3 Relative Evaluation | Sec. 4.2.2 Relative Evaluation |
| chameleon-he-raw | Sec. 4.3 | Sec. 4.2.2 |
| chameleon-he-metric | Sec. 4.3 footnote | Sec. 4.2.2 footnote |
| chameleon-he-fulfill | Sec. 4.3 Absolute Evaluation | Sec. 4.2.1 Absolute Evaluation |

Minor (not counted): chameleon claims cite "Table complete_win_rates"; the LaTeX file is tables/complete_win_rates.tex but
it holds four tables labelled win_rate_gemini+, win_rate_gpt4v+, win_rate_gemini, win_rate_gpt4v. Also chameleon-he-headline
(60.4% / 51.6% from the text) conflicts with the table Overall row (58.8% for Gemini+; 41.5 + 0.5*34.5 = 58.75); the paper
text's 60.4% is really the mixed-modal-prompt row. The separate chameleon-he-table-gemini+ claim already records this;
consider marking chameleon-he-headline uncertain or wording it as "as stated in the text".

## C. Errors in notes/papers/*.md

llava15.md
- Says the global-context ablation is "commented out ... mark uncertain". Wrong: it is rendered (Sec. 4.2, "Global context"). Remove the "mark uncertain" advice.
- "LLaVA-1.5-HD (Sec 3.3 + Appendix A)": the higher-resolution section is Sec. 3.4.
- "Why (Sec 3.1)" for InstructBLIP/format-prompt reasoning: it is Sec. 3.2.
- Everything else checked (all ablation rows, HD gains, hparams, mixture sizes, 11/12 wording, commented "11/12" caption) is correct.

qwenvl.md
- Section numbers shifted by one throughout ("Sec. 3.1, Table 1" is 2.1; "Sec. 3.2" box I/O is 2.2; "Sec. 4" pipeline is Sec. 3; "App. D.2 / D.3" are E.2 / E.3; "Hyperparameter table" is Appendix C Table 8; "Sec. 4.1.1"-style refs do not exist).
- "ViT patch features are the KEYS (and values)": the paper says only that image features are used as keys; "and values" is not stated.
- Equations: none in this paper; mechanism description otherwise matches (single-layer random-init cross-attn, 256 learnable queries, 2D absolute pos-enc on query-key pairs).

qwen2vl.md
- No factual errors found. M-RoPE figure reading (text after a 3x3x4 block starts at (4,4,4)) verified against images/mrope.png. Only caution: notes cite no numbered sections for the ablations, so nothing to fix there; the claims file is where the 3.6.x numbers came from.

qwen3vl.md
- "Interleaved MRoPE (Sec. 2.2)", "DeepStack (Sec. 2.3)", "Text-based video timestamps (Sec. 2.4)": actual 2.1 / 2.2 / 2.3. (Text of Sec. 2 itself cross-references DeepStack as "Section 2.2".)
- Mechanism descriptions (interleaved t/h/w, 3 ViT levels -> first 3 LLM blocks, `<3.0 seconds>` prefix, seconds and HMS) verified. The note that ViT layer indices and patch size are not stated is correct.

florence2.md
- Table numbers: "Table 3 (annotation stats)" -> Table 2; "Table 5, region tasks" -> Table 6; "TextVQA ... Table 4 lists 73.5" -> Table 5 (captioning/VQA); "Table 5 lists Florence-2-B RefCOCOg test 82.2" -> Table 6; "Table 8 model scaling" -> Table 10; "Table 9 data scaling" -> Table 11; "Table 7 basic components" -> Table 12; "(Sec. 4.1 Setup)" -> Sec. 6.1. Table 4 (zero-shot), 13 (tasks), 15 (config) are right.
- Loss equation and V' = LayerNorm(Linear(V)), X = [V', T_prompt] transcription verified against sec/model.tex; correct.

idefics2.md
- "Appendix connector ablation ... only in a commented-out block so it is NOT rendered", and the closing bullet "Appendix connector table ... is inside a \begin{comment} block; verify it is visible": wrong. A live copy is rendered (Table 11). Its values (44.5/51.8/60.3) can be shown.
- "Which architecture each table used: ... captions = cross-attention arch. Pooling, aspect-ratio = fully autoregressive": the source states architecture only for the LM-backbone and vision-encoder swaps (cross-attention) and the architecture table; captions, pooling and aspect-ratio architectures are not stated in the text. Mark as inferred.
- Everything else (tables, 12.9 vs 9.2 discrepancy, "7 points" vs 6.4, 60.7 vs 60.6, 729 -> 64, splitting numbers, final spec) verified.

molmo.md
- Table numbers throughout: "Ablation Table 6" (counting) -> Table 4; "Ablation Elo (Table 4, second study)" -> Table 5; "Tab 3a-3f" -> Table 2a-2f; "Tab 4a-4c" -> Table 3a-3c; "Table 6" hyperparameters is correct as Table 6 only in that role.
- All numbers checked against the PDF tables (Table 1 rows, ablations, Elo, pointing) are correct, including the y1 typo and the 72.2/76.8 text-vs-table mismatch.

chameleon.md
- No wrong equations or mechanism statements found (norm-reordering equations, z-loss, QK-Norm, 20% divergence, 9.2T / 4.4T / ~10T all verified). The tokenizer-comparison figure noted as possibly commented out is confirmed absent from the PDF.

## D. Counts

- Claims checked: 208 (llava15 30, qwenvl 20, qwen2vl 22, qwen3vl 21, florence2 27, idefics2 32, molmo 30, chameleon 29, meta 16, page 1)
- WRONG (loc/description contradicts compiled paper): 2 (llava15-global-ctx, idefics2-connector-ablation)
- MISLOCATED: 60 (llava15 5, qwenvl 17, qwen2vl 8, qwen3vl 5, florence2 8, idefics2 1, molmo 12, chameleon 4)
- SHOULD-BE-UNCERTAIN: 1 (idefics2-captions architecture qualifier); plus 1 already-uncertain claim to consider dropping (chameleon-prev-tokenizers, commented-out source)
- Numeric values found wrong: 0
