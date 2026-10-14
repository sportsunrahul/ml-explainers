# Platform tie-ins (from cogniac-kb and the data-quality skill, 2026-10-09)

Source: a read-only agent over /Users/rahulmishra/git_repo/cogniac-kb and ~/.claude/skills/data-quality.
Status: KB-level only. Verify every item against the service code before it goes on the page
(code is the source of truth). In claims.json use source "platform", loc = repo path + line.
Internals (dead code, bug history, PR numbers) go on the page only if the user approves.

## 1. Selection of media for human feedback (uncertainty sampling)
- Service `feedback-service` reads SQS queue `feedbackservice` (filled by `subject-service`) and calls
  `feedback-service-api` to create FeedbackRequest records. KB: repos/feedback-service.md, repos/feedback-service-api.md.
- `subject-service` "computes feedback-uncertainty metrics that drive the feedback-service scheduler"
  (repos/subject-service.md). Each message carries `feedback_metrics[3]`; the KB does not define the 3 metrics.
- `hypothesis-set`: "feedback prioritization by uncertainty" (repos/hypothesis-set.md).
- `AppPool`: "3 top-K min-heaps (FB_K_METRIC, default 10) for per-metric argmax selection", plus
  "a Vitter reservoir (FB_K_CONTRA, default 200) for the contra-model random sample". KB says the random
  path is "dead code... never actually populates in production". TO VERIFY.
- Rate: "target feedback per hour based on model accuracy and validation set size"; app fields
  `requested_feedback_per_hour`, `system_feedback_per_hour` (api/application-core-api.md).
  Env: FB_K_METRIC, FB_K_CONTRA, PERIODIC_HANDLER_WAIT_TIME, HANDLE_FORCE_FB_ONLY.
- Overrides on the assertion message: `force_feedback` (queue `feedbackservice_force`), `force_review`,
  `force_random_feedback`.
- Bug history (KB): until PR #321/#322, `feedback_metrics` "silently collapsed to [0,0,0]" for box, point,
  segmentation and OCR apps. INTERNAL; ask before use.

## 2. Consensus (label aggregation)
- `consensus` in {'True','False','None','Sidelined'} (architecture/data-model.md); 'Uncertain' also appears
  (repos/subject-expiration-point-count-cron.md).
- "determined by one or more human labelers agreeing" (architecture/glossary.md);
  "Multiple agreeing feedback signals -> consensus" (architecture/media-lifecycle.md).
- `hypothesis-set` "tracks the historical accuracy of a prediction source (a model or a user)";
  Bayesian updates for classification, detection_fullframe, regression; non-Bayesian for box/point/seg/OCR.
  Table `UserConsensusHistory`. Fixed annotators per media or tie-break rule: not found.

## 3. Splits
- `media-service` assigns "a deterministic train/validation split ... deterministic on md5(domain_unit) with
  val_prob 0.25 (media_service_app/_set_assignment.py)" (repos/media-service.md).
- `set_assignment` in {training, validation}. No platform-wide third test split found; a holdout exists
  for CN Railroad only (repos/cn-holdout-evaluation.md).
- Implication for the page (my inference, check): validation media are consensus media, and consensus
  media were picked by uncertainty-driven feedback, so validation is not a random sample of the stream.

## 4. Mislabel detection
- data-quality skill: no cleanlab / confident learning / k-fold (pyproject.toml, scripts/src/analyze.py).
  Heuristics: cross-model vote (`run_model_mislabel`, leaderboard models' predictions from
  s3://cogniac-consensus-release-assertions/.../{validation,training}/1.json; classification flags when
  >50% and >=2 models disagree at CONF_THRESHOLD=0.5; counting median off by >30%; seg prob <0.2 or >0.8;
  OCR vs primary model), IsolationForest+LOF on DINOv2 embedding + label features, K=10 neighbour check
  (NEIGHBOR_Z_THRESHOLD=1.5), near-duplicate box IoU < 0.5 -> `inconsistent_duplicate`.
- Caveat: training-split predictions come from models trained on those labels, so not out-of-sample
  (confident learning requires out-of-sample probabilities).
- Side finding, not to fix: SKILL.md says "top-10 leaderboard models"; code loads leaderboard[1:10] plus primary.
- `cn-dev-tools` `candidate_review` shortlists likely-wrong labels from DINOv2 crop embeddings (repos/cn-dev-tools.md).

## 5. Vocabulary
feedback request (table ApplicationFeedback), pending feedback, feedback-count-cache, force feedback,
force_review, force_random_feedback, replay (app-replay-service-v2, can carry force_feedback),
ops review (/1/ops/*, review_items), Sidelined, consensus release (consensus_release_id), contra model,
labeling model ("assists human reviewers during the feedback/labeling process").
