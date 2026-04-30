# Tinker Progress Report

Date: 2026-04-28
Branch: `pranav/tinker`

## Project Objective

The final project is about improving multi-turn, underspecified conversation with
on-policy RL. The proposal argues that existing single-turn alignment/evaluation
does not directly solve multi-turn settings where a user reveals missing
information over several turns.

The intended research direction is:

- build from the Laban et al. sharded-instruction setting, where a fully
  specified task is split into shards that must be elicited across turns;
- compare sparse reward RL baselines against richer feedback methods;
- implement an RLRF/self-critic style objective that gives denser credit
  assignment than a single terminal reward;
- report accuracy, convergence over RL steps, model-scale effects, task-type
  effects, and possible OOD changes.

The work so far now includes the baseline Tinker harness, sparse GRPO, dense
RLRF-style reward shaping, and a Tinker-compatible SDPO-style feedback
distillation path. The current pilot result is mixed: dense feedback and the
conservative SDPO setting avoid the sparse-GRPO regression, but neither has yet
improved held-out accuracy beyond the base model on the 10-row scaled
`lost_math_200` test split.

Update after sparse baselines: the first dense/RLRF-style reward mode is
implemented and has one 30-step Tinker pilot. It is not full SDPO logit
self-distillation yet; it is the rule-based dense feedback layer needed to test
whether rewarding information-seeking behavior reduces premature final answers.

Update after reviewing the proposal again: SDPO-style feedback distillation is
still part of the proposed method. Dense reward shaping alone is not enough. The
Tinker path now has an opt-in `SHARDED_REWARD_MODE=sdpo` mode that keeps dense
turn rewards and adds feedback-conditioned self-teacher distillation using
Tinker's supported top-k prompt logprob + `cross_entropy` soft-target loss.
This is the closest Tinker-compatible analog of the local/JHU `verl` SDPO path,
which still contains the full logit-KL implementation.

## Code Changes So Far

Tinker support was added as a separate path from the existing local/JHU `verl`
training path.

New files:

- `requirements-tinker.txt`: optional Tinker dependencies.
- `run_tinker_sft.sh`: launcher for Tinker SFT training.
- `run_tinker_grpo.sh`: launcher for Tinker GRPO training.
- `run_tinker_eval.sh`: launcher for Tinker held-out evaluation.
- `scripts/tinker_sft.py`: minimal SFT loop using Tinker cross-entropy updates
  on this repo's JSON datasets.
- `scripts/tinker_grpo.py`: minimal GRPO-style training loop using Tinker
  sampling, forward/backward, optimizer steps, and checkpoint storage.
- `scripts/tinker_eval.py`: evaluates a base model or Tinker checkpoint against
  this repo's JSON datasets.
- `tests/test_tinker_grpo_helpers.py`: focused tests for data loading, reward
  wiring, ToolUse nested JSON parsing, SciKnowEval format diagnostics, and
  Tinker checkpoint path detection.
- `tests/test_tinker_sft_helpers.py`: focused tests for SFT target formatting
  and synthetic sharded demonstration expansion.

Modified files:

- `README.md` and `INSTALL.md`: documented Tinker setup, training, and eval.
- `verl/utils/reward_score/feedback/tooluse.py`: fixed nested JSON
  `Action Input` parsing. The previous regex broke valid ToolUse calls such as
  nested `headers` and `data`.
- `verl/utils/reward_score/feedback/mcq.py`: fixed the `incorrect_format` flag
  so `1` means invalid format and `0` means valid format.

Important Tinker behavior handled:

- Qwen3 should use `RENDERER_NAME=qwen3_disable_thinking` for these tasks.
  Otherwise samples spend the token budget in `<think>` blocks and truncate
  before the answer/tool call.
- Tinker training checkpoints under `/weights/` are not directly usable for
  sampling. Eval needs `/sampler_weights/`. The trainer now saves both final
  paths, and eval can auto-export sampler weights if given a `/weights/` path.
- Training/eval logs are written to `_logs/tinker_sft`, `_logs/tinker_grpo`,
  and `_logs/tinker_eval`.
  The trainer now clears same-run JSONL files at startup to avoid appending
  stale results when reusing run names.
- The API key is read only from `TINKER_API_KEY`; it is not written to tracked
  files.

## Verification

Local checks currently pass:

```bash
./.venv/bin/python -m py_compile scripts/tinker_sft.py scripts/tinker_grpo.py scripts/tinker_eval.py tests/test_tinker_grpo_helpers.py tests/test_tinker_sft_helpers.py
./.venv/bin/python -m pytest tests/test_tinker_grpo_helpers.py tests/test_tinker_sft_helpers.py
bash -n run_tinker_sft.sh run_tinker_grpo.sh run_tinker_eval.sh
```

Latest focused test result: `9 passed`.

The Tinker path has been tested end-to-end with real remote training and
evaluation. The local/JHU path has been intentionally left untouched, but an
actual JHU SLURM/GPU run has not been proven from this machine yet.

## ToolUse Results

ToolUse was the first end-to-end Tinker target because it quickly exposed
whether sampling, rewards, GRPO updates, and checkpoint/eval plumbing worked.

Key findings:

- Qwen3 thinking mode caused many truncated `<think>` samples.
- `qwen3_disable_thinking` fixed the format problem.
- The nested ToolUse JSON parser bug was real and affected reward correctness.

Training run:

```text
tooluse-grpo-qwen3-disable-thinking-100
```

- Train samples: `800` from `200` train examples.
- Train reward: `275/800 = 34.375%`.
- Format errors: `13/800`.
- Final training checkpoint:
  `tinker://724a8e23-52ca-5887-b2dd-09573944381c:train:0/weights/tooluse-grpo-qwen3-disable-thinking-100-final`

Held-out ToolUse test:

| Model | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| Base Qwen3 disable-thinking | `40/68` | `58.82%` | `0/68` |
| GRPO 100-step checkpoint | `41/68` | `60.29%` | `0/68` |

Interpretation: this is a valid end-to-end sparse-GRPO baseline run. It shows
the Tinker path works and does not degrade ToolUse, but the gain is small
(`+1/68`) and should not be presented as strong learning evidence.

## SciKnowEval Results

The first SciKnowEval attempt used `MAX_TOKENS=128`. That was not usable:
almost all model outputs truncated before `</answer>`, producing near-zero
reward and high format-error rates.

Rerunning with `MAX_TOKENS=512` produced usable base baselines:

| Domain | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| biology | `9/50` | `18.0%` | `10/50` |
| chemistry | `56/210` | `26.7%` | `65/210` |
| material | `49/94` | `52.1%` | `14/94` |
| physics | `28/80` | `35.0%` | `44/80` |

The format issue is reduced but not gone. Chemistry and physics still have
substantial truncation because responses often contain long reasoning chains.

Biology 512-token GRPO smoke:

```text
sciknoweval-biology-grpo-qwen3-disable-thinking-512-smoke
```

- Steps: `10`
- Samples: `160`
- Rewarded: `35/160 = 21.875%`
- Format errors: `36/160 = 22.5%`
- Examples with at least one success: `23/40`
- Mixed reward groups: `23/40`
- Nonzero loss steps: `10/10`
- Final sampler checkpoint:
  `tinker://b2f0d2d9-d117-515e-8295-f1ab58d958e4:train:0/sampler_weights/sciknoweval-biology-grpo-qwen3-disable-thinking-512-smoke-final-sampler`

Interpretation: the 512-token biology smoke has real GRPO signal. Unlike the
128-token run, it is not dominated by invalid format. This justifies the
100-step biology pilot.

Biology 512-token GRPO 100-step run:

```text
sciknoweval-biology-grpo-qwen3-disable-thinking-512-100
```

- Steps: `100`
- Samples: `1600` from `400` train examples.
- Rewarded: `442/1600 = 27.625%`.
- Format errors: `231/1600 = 14.44%`.
- Examples with at least one success: `244/400`.
- Mixed reward groups: `226/400`.
- Mean step reward: `0.27625`.
- First-half mean step reward: `0.2625`.
- Second-half mean step reward: `0.29`.
- Zero-reward steps: `2/100`.
- Nonzero loss steps: `96/100`.
- Final training checkpoint:
  `tinker://236bae41-8019-5ee3-b444-19fd6ceecf7f:train:0/weights/sciknoweval-biology-grpo-qwen3-disable-thinking-512-100-final`
- Final sampler checkpoint:
  `tinker://236bae41-8019-5ee3-b444-19fd6ceecf7f:train:0/sampler_weights/sciknoweval-biology-grpo-qwen3-disable-thinking-512-100-final-sampler`

Held-out biology eval:

| Model | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| Base Qwen3 disable-thinking | `9/50` | `18.0%` | `10/50` |
| Biology GRPO 100-step checkpoint | `13/50` | `26.0%` | `4/50` |

Example-level comparison against the base model:

- improved examples: `7`
- regressed examples: `3`
- unchanged correct examples: `6`
- unchanged wrong examples: `34`
- prior format errors fixed: `7`
- new format errors introduced: `1`

Interpretation: the 100-step biology run has substantial sparse-GRPO signal and
a positive held-out result. The gain is modest (`+4/50`, or `+8` accuracy
points), but this is stronger than the ToolUse result because it improves both
accuracy and format validity on a held-out split. This is enough to justify
running additional sparse-GRPO SciKnowEval pilots before implementing dense
RLRF.

Material 512-token GRPO 100-step run:

```text
sciknoweval-material-grpo-qwen3-disable-thinking-512-100
```

- Steps: `100`
- Samples: `1600` from `400` train examples.
- Rewarded: `793/1600 = 49.56%`.
- Format errors: `170/1600 = 10.62%`.
- Examples with at least one success: `257/400`.
- All-success examples: `143/400`.
- All-fail examples: `143/400`.
- Mixed reward groups: `114/400`.
- Mean step reward: `0.495625`.
- First-half mean step reward: `0.44`.
- Second-half mean step reward: `0.55125`.
- Zero-reward steps: `4/100`.
- Nonzero loss steps: `70/100`.
- Final training checkpoint:
  `tinker://e1dfcdef-5227-5813-a864-46fd3afaa550:train:0/weights/sciknoweval-material-grpo-qwen3-disable-thinking-512-100-final`
- Final sampler checkpoint:
  `tinker://e1dfcdef-5227-5813-a864-46fd3afaa550:train:0/sampler_weights/sciknoweval-material-grpo-qwen3-disable-thinking-512-100-final-sampler`

Interpretation: material has the strongest sparse-GRPO training signal so far.
The second half is meaningfully better than the first half, and the late run has
several high-reward steps.

Held-out material eval:

| Model | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| Base Qwen3 disable-thinking | `49/94` | `52.1%` | `14/94` |
| Material GRPO 100-step checkpoint | `57/94` | `60.6%` | `11/94` |

Example-level comparison against the base model:

- improved examples: `13`
- regressed examples: `5`
- unchanged correct examples: `44`
- unchanged wrong examples: `32`
- prior format errors fixed: `6`
- new format errors introduced: `3`

Interpretation: material gives a second positive held-out SciKnowEval result.
The gain is `+8/94`, or about `+8.5` accuracy points, and format validity also
improves slightly. Together with biology, this gives early evidence that the
sparse-GRPO Tinker baseline can improve single-turn scientific MCQ tasks before
we move to the final multi-turn RLRF setting.

Chemistry 768-token GRPO smoke:

```text
sciknoweval-chemistry-grpo-qwen3-disable-thinking-768-smoke
```

- Steps: `10`
- Samples: `160` from `40` train examples.
- Rewarded: `68/160 = 42.5%`.
- Format errors: `3/160 = 1.875%`.
- Examples with at least one success: `28/40`.
- All-success examples: `10/40`.
- All-fail examples: `12/40`.
- Mixed reward groups: `18/40`.
- Mean step reward: `0.425`.
- Zero-reward steps: `0/10`.
- Nonzero loss steps: `9/10`.
- Final sampler checkpoint:
  `tinker://5ae3aa96-a935-5d9d-bc3a-f43d37c3ebd5:train:0/sampler_weights/sciknoweval-chemistry-grpo-qwen3-disable-thinking-768-smoke-final-sampler`

Interpretation: increasing chemistry to `MAX_TOKENS=768` largely fixes the
format/truncation problem seen in the 512-token base eval, at least on the
training smoke. The smoke has enough mixed groups and reward signal to justify
a 100-step chemistry run, but held-out chemistry should also be re-evaluated at
768 tokens so the comparison is fair.

Chemistry 768-token GRPO 100-step run:

```text
sciknoweval-chemistry-grpo-qwen3-disable-thinking-768-100
```

- Steps: `100`
- Samples: `1600` from `400` train examples.
- Rewarded: `663/1600 = 41.44%`.
- Format errors: `95/1600 = 5.94%`.
- Examples with at least one success: `273/400`.
- All-success examples: `77/400`.
- All-fail examples: `127/400`.
- Mixed reward groups: `196/400`.
- Mean step reward: `0.414375`.
- First-half mean step reward: `0.44875`.
- Second-half mean step reward: `0.38`.
- Zero-reward steps: `0/100`.
- Nonzero loss steps: `92/100`.
- Final training checkpoint:
  `tinker://52cc2022-1cc3-5b55-aa0e-e5bf7145270a:train:0/weights/sciknoweval-chemistry-grpo-qwen3-disable-thinking-768-100-final`
- Final sampler checkpoint:
  `tinker://52cc2022-1cc3-5b55-aa0e-e5bf7145270a:train:0/sampler_weights/sciknoweval-chemistry-grpo-qwen3-disable-thinking-768-100-final-sampler`

Interpretation: chemistry has clean formatting and many mixed groups, so the
run is a valid sparse-GRPO baseline. Unlike material, the training reward does
not trend upward over the second half. The held-out eval is therefore especially
important before deciding whether chemistry benefits from this sparse setup.

Held-out chemistry eval at `MAX_TOKENS=768`:

| Model | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| Base Qwen3 disable-thinking | `87/210` | `41.4%` | `10/210` |
| Chemistry GRPO 100-step checkpoint | `85/210` | `40.5%` | `7/210` |

Example-level comparison against the base model:

- improved examples: `22`
- regressed examples: `24`
- unchanged correct examples: `63`
- unchanged wrong examples: `101`
- prior format errors fixed: `7`
- new format errors introduced: `4`

Interpretation: chemistry is a mixed/negative held-out result. The checkpoint
improves format validity, but accuracy is slightly below the matching 768-token
base model (`-2/210`). This matters for the final report: sparse GRPO is not
uniformly beneficial across domains, and chemistry may need either more data,
different sampling, a smaller learning rate, or the planned dense-feedback
method rather than more of the same sparse reward setup.

Physics 768-token base eval:

| Model | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| Base Qwen3 disable-thinking, 512 tokens | `28/80` | `35.0%` | `44/80` |
| Base Qwen3 disable-thinking, 768 tokens | `36/80` | `45.0%` | `28/80` |

Interpretation: increasing physics from `512` to `768` tokens improves both
accuracy and format validity, but the format-error rate is still high
(`35.0%`). Physics remains the noisiest SciKnowEval domain under the current
prompt.

Physics 768-token GRPO smoke:

```text
sciknoweval-physics-grpo-qwen3-disable-thinking-768-smoke
```

- Steps: `10`
- Samples: `160` from `40` train examples.
- Rewarded: `73/160 = 45.625%`.
- Format errors: `63/160 = 39.375%`.
- Examples with at least one success: `24/40`.
- All-success examples: `13/40`.
- All-fail examples: `16/40`.
- Mixed reward groups: `11/40`.
- Mean step reward: `0.45625`.
- Zero-reward steps: `0/10`.
- Nonzero loss steps: `8/10`.
- Final sampler checkpoint:
  `tinker://27f0db81-8c9d-5361-9761-f58c6620fabf:train:0/sampler_weights/sciknoweval-physics-grpo-qwen3-disable-thinking-768-smoke-final-sampler`

Interpretation: the physics smoke has nonzero reward on every step and can be
used to run a 100-step sparse-GRPO baseline, but it is substantially noisier
than the other domains because nearly 40% of samples are still malformed. A
100-step run is useful for completing the sparse baseline matrix, but a
concise-answer prompt is likely needed before treating physics as a clean
optimization target.

Physics 768-token GRPO 100-step run:

```text
sciknoweval-physics-grpo-qwen3-disable-thinking-768-100
```

- Steps: `100`
- Samples: `1600` from `400` train examples.
- Rewarded: `931/1600 = 58.19%`.
- Format errors: `272/1600 = 17.0%`.
- Examples with at least one success: `302/400`.
- All-success examples: `167/400`.
- All-fail examples: `98/400`.
- Mixed reward groups: `135/400`.
- Mean step reward: `0.581875`.
- First-half mean step reward: `0.56625`.
- Second-half mean step reward: `0.5975`.
- Zero-reward steps: `0/100`.
- Nonzero loss steps: `80/100`.
- Final training checkpoint:
  `tinker://f47c27d2-8f1a-5979-b399-c37c89ceebb5:train:0/weights/sciknoweval-physics-grpo-qwen3-disable-thinking-768-100-final`
- Final sampler checkpoint:
  `tinker://f47c27d2-8f1a-5979-b399-c37c89ceebb5:train:0/sampler_weights/sciknoweval-physics-grpo-qwen3-disable-thinking-768-100-final-sampler`

Interpretation: physics has strong training reward and a mild upward trend from
the first half to the second half. The format-error rate is much lower than the
10-step smoke but still higher than the other domains. The held-out eval below
checks whether this training signal transfers beyond the first 400 train
examples.

Held-out physics eval at `MAX_TOKENS=768`:

| Model | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| Base Qwen3 disable-thinking | `36/80` | `45.0%` | `28/80` |
| Physics GRPO 100-step checkpoint | `42/80` | `52.5%` | `11/80` |

Example-level comparison against the base model:

- improved examples: `11`
- regressed examples: `5`
- unchanged correct examples: `31`
- unchanged wrong examples: `33`
- prior format errors fixed: `17`
- new format errors introduced: `0`

Interpretation: physics is a positive held-out result. The checkpoint gains
`+6/80`, or `+7.5` accuracy points, and cuts format errors by more than half.
This is important because physics was the noisiest domain under the current
prompt. The model still has `11/80` malformed outputs, so a concise-answer
prompt remains worth testing, but sparse GRPO is not merely overfitting the
training split.

Overall held-out SciKnowEval sparse-GRPO status:

| Domain | Base Accuracy | GRPO Accuracy | Delta | Base Format Errors | GRPO Format Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology, 512 tokens | `18.0%` | `26.0%` | `+8.0` pts | `10/50` | `4/50` |
| material, 512 tokens | `52.1%` | `60.6%` | `+8.5` pts | `14/94` | `11/94` |
| chemistry, 768 tokens | `41.4%` | `40.5%` | `-1.0` pts | `10/210` | `7/210` |
| physics, 768 tokens | `45.0%` | `52.5%` | `+7.5` pts | `28/80` | `11/80` |

Interpretation: three of four SciKnowEval domains show held-out accuracy gains
from the 100-step sparse-GRPO checkpoint, and all four reduce format errors.
Chemistry is the exception on accuracy, which is useful evidence for the final
project: sparse terminal reward can help, but it is not uniformly reliable.
That motivates the proposed dense-feedback/RLRF method rather than weakening
the case for it.

## How This Advances the Final Project

This work is progress toward the final project in three concrete ways.

First, it establishes an on-policy RL execution path on Tinker. The project
needs repeated RL experiments across model sizes/tasks; relying only on local
or cluster setup would make iteration slower. We now have remote sampling,
training, checkpointing, and held-out eval working.

Second, it produces sparse-reward GRPO baselines. The final proposal explicitly
needs comparisons against sparse reward methods. ToolUse and SciKnowEval are
not the final sharded multi-turn benchmark, but they validate the sparse-GRPO
machinery before we add multi-turn state and dense self-critic feedback.

Third, it surfaces practical reward/eval issues early:

- renderer choice matters for Qwen3;
- token budget can dominate measured reward;
- reward parsers need to be robust before training;
- held-out eval needs sampler checkpoints, not training checkpoints;
- GRPO only gets useful signal from mixed reward groups.

These are exactly the kinds of operational problems that would otherwise
confound the final RLRF experiments.

## Remaining Gaps

The current work is not yet the final project method.

Remaining gaps:

- dense RLRF-style reward shaping is implemented and evaluated once, but has
  not yet beaten base Qwen on held-out accuracy;
- Tinker-compatible SDPO-style feedback distillation is implemented but has not
  yet been run and evaluated end to end;
- full-logit SDPO remains a local/JHU `verl` capability rather than an exact
  Tinker loss;
- no CURIO-style curiosity baseline has been run;
- no model-scale sweep has been run;
- no OOD evaluation has been run;
- local/JHU end-to-end execution still needs a real cluster smoke test.

## Next Steps

1. Run base eval on the converted Lost in Conversation math pilot.

   The new pilot lives at `datasets/sharded_multiturn/lost_math` with `36`
   train and `4` test rows converted from `microsoft/lost_in_conversation`.

2. Run a short sparse-GRPO smoke on the same pilot.

   Start small because examples have up to `11` hidden shards, so one rollout
   can require many Tinker sample calls.

   Completed first smoke:

   - base eval: `2/4 = 50.0%`, format errors `1/4`
   - GRPO smoke train: `7/20 = 35.0%`, mixed groups `2/5`
   - GRPO smoke checkpoint eval: `2/4 = 50.0%`, logged format errors `1/4`
   - sample-level comparison: `1` improved, `1` regressed, `1` unchanged
     correct, `1` unchanged wrong
   - final sampler checkpoint:
     `tinker://80b32044-f890-542a-acab-d61bdd131d0e:train:0/sampler_weights/lost-math-grpo-smoke-final-sampler`

   Interpretation: the smoke checkpoint did not beat the base aggregate. It
   did reveal a real failure mode where the model invented an
   `Additional information ...` shard itself and answered from that
   hallucinated fact. The sharded scorer now treats assistant-authored
   `Additional information` messages as invalid format before larger runs, so
   the same regressed sample would now be counted as a format error. A local
   rescore of the saved JSONL samples keeps reward at `2/4` and increases
   format errors from `1/4` to `2/4`.

   Completed guarded smoke:

   - GRPO guarded smoke train: `11/20 = 55.0%`
   - format errors: `7/20 = 35.0%`
   - environment-impersonation failures: `0/20`
   - mixed groups: `2/5`
   - logged checkpoint eval: `1/4 = 25.0%`, format errors `3/4`
   - checkpoint eval after local rescore with corrected `<answer>...</answer>`
     parsing: `2/4 = 50.0%`, format errors `2/4`
   - checkpoint eval environment-impersonation trajectories: `1/4`
   - final sampler checkpoint:
     `tinker://b1a02fc0-91ed-508d-82bc-d0a0f42d08e4:train:0/sampler_weights/lost-math-grpo-guard-smoke-final-sampler`

   Completed user-detail prompt smoke:

   - base eval with revised prompt: `2/4 = 50.0%`, format errors `0/4`
   - GRPO user-detail smoke train: `9/20 = 45.0%`
   - format errors: `7/20 = 35.0%`
   - environment-impersonation failures: `4/20`, all in one train group
   - mixed groups: `2/5`
   - final sampler checkpoint:
     `tinker://00149cfc-aace-5243-9622-99496d3d317b:train:0/sampler_weights/lost-math-grpo-user-detail-smoke-final-sampler`

   Interpretation: the revised prompt fixed held-out base formatting, but the
   explicit `User-provided detail ...` label is still copyable. The sharded
   environment now emits unlabeled hidden-shard messages while retaining the
   guard for both old labeled phrasings.

   Completed unlabeled shard smoke:

   - base eval with unlabeled shards: `2/4 = 50.0%`, format errors `0/4`
   - GRPO unlabeled smoke train: `12/20 = 60.0%`
   - format errors: `4/20 = 20.0%`
   - environment-impersonation failures: `0/20`
   - mixed groups: `0/5`
   - final sampler checkpoint:
     `tinker://72709eee-9d32-57a3-8701-327e9c143824:train:0/sampler_weights/lost-math-grpo-unlabeled-smoke-final-sampler`

   Interpretation: this is the cleanest sharded environment so far, but the
   smoke had no mixed groups and all Tinker losses were zero. It should be
   treated as a no-op train run, not as learned improvement. The runner now
   skips zero-advantage samples so future no-op runs are obvious in the logs,
   and it supports `SHUFFLE_SEED` to sample a different training slice without
   regenerating data.

   Completed shuffled unlabeled sparse-GRPO smoke:

   - run: `lost-math-grpo-unlabeled-shuffle7-10`
   - train rows used: `10`
   - rewarded rollouts: `16/40 = 40.0%`
   - format errors: `12/40 = 30.0%`
   - environment-impersonation failures: `0/40`
   - mixed groups: `6/10`
   - optimizer steps with nonzero advantages: `6/10`
   - skipped zero-advantage steps: `4/10`
   - checkpoint eval: `2/4 = 50.0%`, format errors `0/4`
   - comparison to base: same two successes and same two failures
   - final sampler checkpoint:
     `tinker://b7977a46-7133-5ee8-ab83-d5cc679f5754:train:0/sampler_weights/lost-math-grpo-unlabeled-shuffle7-10-final-sampler`

   Interpretation: this is the first clean sharded Lost Math sparse-GRPO run
   that is worth evaluating. It preserves zero environment-impersonation
   failures and has enough mixed reward groups for real updates. Held-out eval
   ties the clean base rather than improving it; the remaining failure mode is
   premature final answers before enough hidden shards are revealed.

   Scaled Lost Math split prepared:

   - output: `datasets/sharded_multiturn/lost_math_200`
   - cap requested: `--max-records 200`
   - convertible math rows available from cached source: `103`
   - train/test: `93/10`
   - reward kind: `number`
   - hidden shards per row: `3` to `11`, mean `4.68`
   - local parquet preprocessing: passed
   - Tinker dry runs for train and eval launchers: passed
   - scaled base eval: `5/10 = 50.0%`, format errors `2/10`
   - clean 30-step scaled sparse-GRPO pilot: `53/120 = 44.2%` rewarded
     rollouts from terminal logs
   - optimizer steps with nonzero advantages: `10/30`
   - skipped zero-advantage steps: `20/30`
   - assistant-turn datums used for nonzero-advantage updates: `212`
   - checkpoint eval: `4/10 = 40.0%`, format errors `2/10`
   - comparison to base: `0` improved, `1` regressed, `4` unchanged correct,
     `5` unchanged wrong
   - final sampler checkpoint from completed run:
     `tinker://0e50e3c3-b7ea-5d28-9016-194b29aedb57:train:0/sampler_weights/lost-math-103-grpo-unlabeled-shuffle7-30-final-sampler`

   A second same-name run was accidentally launched before the first finished,
   so the local JSONL training logs for `lost-math-103-grpo-unlabeled-shuffle7-30`
   are interleaved and should not be used for aggregate analysis. The runner now
   creates a run lock to prevent concurrent duplicate run names.

   Interpretation: the scaled sparse-GRPO checkpoint regresses held-out
   accuracy relative to base (`40%` versus `50%`) without improving format
   validity. The one regressed example is a premature final answer: base waited
   for enough hidden shards and answered correctly, while the GRPO checkpoint
   answered too early. This is useful negative evidence for the project because
   it shows the sparse terminal baseline does not solve the core multi-turn
   information-gathering problem.

   Completed dense/RLRF 30-step pilot:

   - run: `lost-math-103-dense-rlrf-shuffle7-30`
   - train rows used: `30`
   - optimizer steps with nonzero advantages: `24/30`
   - skipped zero-advantage steps: `6/30`
   - assistant-turn datums used for nonzero-advantage updates: `168`
   - mean dense training reward across steps: `0.2485`
   - first-half mean dense training reward: `0.2619`
   - second-half mean dense training reward: `0.2352`
   - final sampler checkpoint:
     `tinker://c5134071-73d9-57e7-873b-12cb184f0d7f:train:0/sampler_weights/lost-math-103-dense-rlrf-shuffle7-30-final-sampler`

   Dense checkpoint eval:

   - run: `lost-math-103-dense-rlrf-shuffle7-30-eval`
   - reward: `5/10 = 50.0%`
   - format errors: `2/10 = 20.0%`
   - comparison to base: same `5` successes and same `5` failures
   - comparison to sparse GRPO: fixes the sparse regression on
     `sharded-GSM8K/1027`; dense waits for all `5` hidden shards and answers
     `$12`

   Interpretation: dense shaping materially improves the density of the
   training signal over sparse GRPO (`24/30` update steps instead of `10/30`)
   and avoids the observed sparse-regression failure. However, it ties the base
   model on held-out accuracy rather than improving it. This supports the
   project motivation but is not yet a positive final-method result.

3. Scale the Lost in Conversation conversion if reward signal is usable.

   The next sparse-GRPO run should use `datasets/sharded_multiturn/lost_math_200`
   with the unlabeled shard environment. Start with a base eval on the 10-row
   test split, then run a 30-step shuffled sparse-GRPO pilot and evaluate its
   final sampler checkpoint if the run has mixed reward groups and nonzero
   updates.

4. Implement dense-feedback/RLRF.

   Implemented first pass:

   - `SHARDED_REWARD_MODE=sparse` remains the default sparse terminal baseline.
   - `SHARDED_REWARD_MODE=dense` or `rlrf` enables per-turn dense rewards.
   - clarification while hidden shards remain gets partial positive reward;
   - premature final answers before all hidden shards are revealed get zero
     dense reward with explicit feedback;
   - final answers after all shards are revealed use the existing exact/numeric
     scorer;
   - dense advantages are centered by turn index within each rollout group.

   Completed first dense pilot:

   - dense train signal: `24/30` nonzero optimizer steps, compared with
     `10/30` for sparse GRPO;
   - dense eval: `5/10`, matching base Qwen and beating sparse GRPO's `4/10`;
   - dense fixed the sparse-regressed `sharded-GSM8K/1027` example.

   Completed 60-step dense pilot:

   - run: `lost-math-103-dense-rlrf-shuffle7-60`
   - train rows used: `60`
   - optimizer steps with nonzero advantages: `40/60`
   - skipped zero-advantage steps: `20/60`
   - assistant-turn datums used for nonzero-advantage updates: `268`
   - mean dense training reward across steps: `0.2446`
   - first-half mean dense training reward: `0.2443`
   - second-half mean dense training reward: `0.2449`
   - checkpoint eval: `5/10`, format errors `2/10`
   - comparison to dense 30-step and base: same `5` successes and same `5`
     failures
   - comparison to sparse GRPO: still fixes the sparse-regressed
     `sharded-GSM8K/1027` example

   The 60-step result suggests that simply scaling this exact dense reward is
   not enough. That is why the next implementation step is SDPO-style
   self-distillation from the dense feedback traces rather than only more dense
   scalar-reward training.

5. Implement SDPO-style feedback distillation on Tinker.

   Implemented first pass:

   - `SHARDED_REWARD_MODE=sdpo` is now a distinct sharded mode.
   - It uses the same dense turn rewards and turn-index-centered GRPO
     advantages as `SHARDED_REWARD_MODE=dense`.
   - For failed sharded rollouts, it selects the first turn where the dense
     teacher identified a behavioral mistake.
   - It builds a feedback-conditioned self-teacher prompt with the fully
     specified instruction, hidden shard schedule, rubric feedback, and an
     on-policy successful peer attempt when one exists.
   - The default Tinker distillation objective teacher-forces the student's
     sampled response under that teacher prompt, requests `SDPO_TOPK=20`
     top-k prompt logprobs, and trains the original student prompt with
     `cross_entropy` soft targets.
   - `SDPO_TOPK=0` remains available as a generated-target CE fallback.

   This is closer to the proposal and SDPO paper than dense reward shaping
   because it adds token-level teacher distribution targets. It is still not a
   byte-for-byte copy of the local/JHU `verl` SDPO objective, because Tinker
   exposes top-k CE targets rather than the full-logit KL path used in
   `verl/trainer/ppo/core_algos.py`.

   Suggested conservative smoke after the first top-k run:

   ```bash
   PYTHON_BIN=.venv/bin/python \
   DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
   RENDERER_NAME=qwen3_disable_thinking \
   SHARDED_REWARD_MODE=sdpo \
   SDPO_TOPK=20 \
   SDPO_DISTILL_WEIGHT=0.1 \
   SDPO_SKIP_FIRST_N_TOKENS=3 \
   SHUFFLE_SEED=7 \
   MAX_STEPS=5 \
   BATCH_SIZE=1 \
   ROLLOUT_N=4 \
   MAX_TURNS=0 \
   MAX_TOKENS=256 \
   ./run_tinker_grpo.sh lost-math-103-sdpo-topk-w01-skip3-shuffle7-smoke
   ```

   Completed SDPO top-k smoke:

   - run: `lost-math-103-sdpo-topk-shuffle7-smoke`
   - train rows used: `5`
   - reward means by step: `0.2333`, `0.3542`, `0.2500`, `0.1557`, `0.4000`
   - optimizer steps with trainable signal: `4/5`
   - skipped constant-signal steps: `1/5`
   - dense/GRPO assistant-turn datums used: `30`
   - SDPO top-k distillation datums used: `7`
   - SDPO top-k token positions used: `362`
   - steps with SDPO distillation signal: `3/5`
   - final sampler checkpoint:
     `tinker://eee8cb04-5b63-54ce-8138-917dac3ea139:train:0/sampler_weights/lost-math-103-sdpo-topk-shuffle7-smoke-final-sampler`

   Interpretation: this validates the combined Tinker objective end to end.
   The run exercised dense sharded GRPO and feedback-conditioned top-k
   self-teacher distillation in the same training loop. It is a smoke test, not
   evidence of held-out improvement yet. The next step is to evaluate this smoke
   checkpoint and then run a 30-step SDPO pilot if the checkpoint eval does not
   show a severe regression.

   Completed SDPO top-k smoke eval:

   - run: `lost-math-103-sdpo-topk-shuffle7-smoke-eval`
   - reward: `4/10 = 40.0%`
   - format errors: `2/10 = 20.0%`
   - comparison to base and dense 30/60-step checkpoints: `-1/10`
   - comparison to sparse GRPO 30-step checkpoint: tied at `4/10`

   Interpretation: the 5-step SDPO smoke did not improve held-out accuracy, but
   it also did not show a catastrophic failure. It was reasonable to proceed to
   the 30-step pilot.

   Completed SDPO top-k 30-step training:

   - run: `lost-math-103-sdpo-topk-shuffle7-30`
   - train rows used: `30`
   - optimizer steps with trainable signal: `27/30`
   - skipped constant-signal steps: `3/30` (`4`, `10`, `27`)
   - dense/GRPO assistant-turn datums used: `184`
   - SDPO top-k distillation datums used: `66`
   - SDPO top-k token positions used: `3487`
   - steps with any SDPO distillation signal: `22/30`
   - steps with any dense/GRPO signal: `23/30`
   - SDPO-only optimizer steps: `4` (`14`, `18`, `19`, `24`)
   - mean training reward across steps: `0.2462`
   - first-half mean training reward: `0.2508`
   - second-half mean training reward: `0.2416`
   - telemetry warning: one non-fatal Tinker telemetry connection failure at
     step `20`; training continued and saved the final checkpoints
   - final sampler checkpoint:
     `tinker://3a2b98dd-8862-5190-934c-66ba227dacc5:train:0/sampler_weights/lost-math-103-sdpo-topk-shuffle7-30-final-sampler`

   Interpretation: the 30-step SDPO run has substantially denser trainable
   signal than sparse GRPO and denser token-level signal than scalar dense
   shaping. The training reward curve itself is roughly flat, so the held-out
   eval is necessary before claiming improvement.

   Completed SDPO top-k 30-step eval:

   - run: `lost-math-103-sdpo-topk-shuffle7-30-eval`
   - reward: `4/10 = 40.0%`
   - format errors: `2/10 = 20.0%`
   - comparison to sparse GRPO: tied at `4/10`
   - comparison to base and dense 30/60-step checkpoints: `-1/10`
   - regressed example relative to base/dense: `sharded-GSM8K/1027`

   Interpretation: the first SDPO top-k configuration did not improve held-out
   accuracy. It regressed the same premature-final-answer example that dense
   reward shaping had fixed. This points to the distillation term being too
   strong or too noisy for the 30-row pilot, so future SDPO top-k runs should
   use the conservative defaults `SDPO_DISTILL_WEIGHT=0.1` and
   `SDPO_SKIP_FIRST_N_TOKENS=3`, or explicitly ablate the generated-target
   fallback with `SDPO_TOPK=0`.

   Completed conservative SDPO top-k 30-step training:

   - run: `lost-math-103-sdpo-topk-w01-skip3-shuffle7-30`
   - setting change from first SDPO pilot: `SDPO_DISTILL_WEIGHT=0.1`,
     `SDPO_SKIP_FIRST_N_TOKENS=3`
   - train rows used: `30`
   - optimizer steps with trainable signal: `26/30`
   - skipped constant-signal steps: `4/30` (`4`, `10`, `26`, `27`)
   - dense/GRPO assistant-turn datums used: `193`
   - SDPO top-k distillation datums used: `69`
   - SDPO top-k token positions used: `4708`
   - steps with any SDPO distillation signal: `24/30`
   - steps with any dense/GRPO signal: `22/30`
   - SDPO-only optimizer steps: `4` (`12`, `14`, `24`, `29`)
   - mean training reward across steps: `0.2511`
   - first-half mean training reward: `0.2607`
   - second-half mean training reward: `0.2414`
   - summed SDPO loss scale: `143.1`, down from `2469.7` in the
     `SDPO_DISTILL_WEIGHT=1.0` run
   - final sampler checkpoint:
     `tinker://37083a23-bc06-5f37-bb45-49c19d44d048:train:0/sampler_weights/lost-math-103-sdpo-topk-w01-skip3-shuffle7-30-final-sampler`

   Interpretation: the conservative run still exercised SDPO frequently but
   reduced the top-k CE loss scale by roughly an order of magnitude relative to
   the first SDPO pilot. This is the correct next checkpoint to evaluate before
   trying the `SDPO_TOPK=0` generated-target ablation.

   Completed conservative SDPO top-k 30-step eval:

   - run: `lost-math-103-sdpo-topk-w01-skip3-shuffle7-30-eval`
   - reward: `5/10 = 50.0%`
   - format errors: `2/10 = 20.0%`
   - comparison to base and dense 30/60-step checkpoints: tied
   - comparison to sparse GRPO and first SDPO top-k 30-step checkpoint: `+1/10`
   - success set: `sharded-GSM8K/1027`, `sharded-GSM8K/40`,
     `sharded-GSM8K/543`, `sharded-GSM8K/752`, `sharded-GSM8K/435`

   Interpretation: reducing SDPO weight and skipping the first three generated
   tokens fixed the first SDPO pilot's regression on `sharded-GSM8K/1027`. The
   conservative SDPO setting is now behaviorally tied with dense reward shaping:
   it preserves the project-specific sparse-regression fix, but it is still not
   a held-out accuracy win over base.

   Completed generated-target SDPO 30-step training:

   - run: `lost-math-103-sdpo-generated-w01-shuffle7-30`
   - settings: `SDPO_TOPK=0`, `SDPO_DISTILL_WEIGHT=0.1`,
     `SDPO_SKIP_FIRST_N_TOKENS=3`
   - train rows used: `30`
   - optimizer steps with trainable signal: `25/30`
   - skipped constant-signal steps: `5/30` (`1`, `4`, `10`, `13`, `27`)
   - dense/GRPO assistant-turn datums used: `175`
   - generated-target SDPO distillation datums used: `71`
   - steps with any SDPO distillation signal: `22/30`
   - steps with any dense/GRPO signal: `19/30`
   - SDPO-only optimizer steps: `6` (`12`, `14`, `15`, `16`, `21`, `24`)
   - mean training reward across steps: `0.2401`
   - first-half mean training reward: `0.2536`
   - second-half mean training reward: `0.2267`
   - summed generated-target SDPO loss scale: `479.4`, compared with `143.1`
     for conservative top-k SDPO
   - final sampler checkpoint:
     `tinker://7f65e995-9f35-52a8-84b2-f0d5add82f19:train:0/sampler_weights/lost-math-103-sdpo-generated-w01-shuffle7-30-final-sampler`

   Interpretation: generated-target SDPO exercised the distillation path about
   as often as conservative top-k SDPO, but its CE loss was larger and the
   training reward declined in the second half. Evaluate this checkpoint before
   deciding whether generated-target CE is a useful fallback or simply noisier
   than top-k soft targets.

   Completed generated-target SDPO 30-step eval:

   - run: `lost-math-103-sdpo-generated-w01-shuffle7-30-eval`
   - reward: `4/10 = 40.0%`
   - format errors: `2/10 = 20.0%`
   - comparison to sparse GRPO and first top-k SDPO: tied
   - comparison to base, dense 30/60-step, and conservative top-k SDPO: `-1/10`
   - regressed example relative to base/dense/conservative top-k:
     `sharded-GSM8K/1027`
   - success set: `sharded-GSM8K/40`, `sharded-GSM8K/543`,
     `sharded-GSM8K/752`, `sharded-GSM8K/435`

   Interpretation: generated-target CE is not the better Tinker SDPO variant
   for this pilot. It reproduces the same held-out regression as sparse GRPO and
   the over-weighted top-k SDPO run. Conservative top-k SDPO remains the best
   SDPO-style setting tested so far because it preserves the dense-feedback fix
   for premature final-answer behavior.

6. Scale the next comparison to a mixed task split.

   Created `datasets/sharded_multiturn/lost_math_actions_200` from cached
   `microsoft/lost_in_conversation` rows with no task filter. The converter can
   currently score `math` and `actions` rows, which gives a larger and more
   proposal-aligned split than math alone:

   - converted rows: `208`
   - train rows: `187`
   - test rows: `21`
   - train task mix: `95` math, `92` actions
   - test task mix: `8` math, `13` actions
   - hidden shards per row: train `3` to `11`, test `3` to `7`
   - JSON files are committed; parquet files can be regenerated with
     `HF_HOME=.cache/huggingface ./.venv/bin/python data/preprocess.py --data_source datasets/sharded_multiturn/lost_math_actions_200`

   This is the right next project step because it tests whether the dense/SDPO
   behavior survives a task-type shift instead of only tuning on the 10-example
   math held-out set.

   Recommended run order:

   1. Base Qwen eval on the 21-row mixed test split.
   2. Sparse GRPO 60-step pilot on the mixed train split.
   3. Dense RLRF 60-step pilot on the same mixed train split.
   4. Conservative top-k SDPO 60-step pilot on the same mixed train split.

   Keep the same renderer and decoding settings across all runs so the result
   table remains a fair comparison.

   Completed mixed-split base eval:

   - run: `lost-math-actions-base-test`
   - reward: `4/21 = 19.05%`
   - format errors: `4/21 = 19.05%`
   - math subset: `4/8 = 50.0%`
   - actions subset: `0/13 = 0.0%`
   - failures with premature final-answer behavior: `15/21`
   - format-error examples: `sharded-BFCL/parallel_18`,
     `sharded-BFCL/parallel_46`, `sharded-GSM8K/214`,
     `sharded-BFCL/parallel_42`
   - successful examples: `sharded-GSM8K/435`, `sharded-GSM8K/283`,
     `sharded-GSM8K/799`, `sharded-GSM8K/140`

   Interpretation: the mixed split exposes a much stronger project-relevant
   failure mode than the math-only pilot. Base Qwen handles some math examples
   but fails every held-out actions example, mostly by answering before enough
   shards are revealed. The next run should be sparse GRPO on this split, then
   dense RLRF and conservative top-k SDPO under the same 60-step budget.

   Prompt-confound follow-up:

   The default sharded system prompt and SDPO teacher prompt are intentionally
   explicit, but that verbosity may itself affect multi-turn behavior. The
   upstream Lost-in-Conversation math prompt is much lighter (`Q: ...` / `A:`
   with a short math system prompt), so this branch now supports a prompt
   ablation:

   - `SHARDED_PROMPT_STYLE=minimal`: the minimal domain system prompt plus a
     `Q: ...` / `A:` initial user message, with plain shard text in follow-up
     turns.
   - `SHARDED_PROMPT_STYLE=linc_math`: same minimal prompt shape; this remains
     as a compatibility alias.
   - `SHARDED_ALLOW_UNTAGGED_FINAL=1`: score non-question untagged responses as
     final-answer attempts, so removing the XML instruction does not create a
     pure formatting failure.
   - `SDPO_TEACHER_PROMPT_STYLE=minimal_teacher`: the default SDPO teacher;
     it only rewrites the system prompt with the added underspecification
     suffix and otherwise keeps the original conversation state.
   - `SDPO_TEACHER_PROMPT_STYLE=enhanced`: use the richer privileged teacher
     prompt with the full task description and feedback context.
   - `SDPO_TEACHER_PROMPT_STYLE=brief`: legacy shorter privileged teacher
     prompt with a "This may be under-specified..." prefix.

   Completed minimal-prompt base eval:

   - run: `lost-math-actions-base-minimal-untagged-test`
   - reward: `4/21 = 19.05%`
   - format errors: `0/21 = 0.0%`
   - comparison to default-prompt base: same task success, fewer format errors

   Interpretation: prompt verbosity and XML formatting were real measurement
   confounds for format errors, but not for task success. With only the
   underspecified user task as input, base Qwen still solves only `4/21`, so the
   mixed split remains a valid test of intrinsic under-specification handling.

   Completed minimal-prompt sparse-GRPO 60-step baseline:

   - training run: `lost-math-actions-sparse-grpo-minimal-shuffle7-60`
   - eval run: `lost-math-actions-sparse-grpo-minimal-shuffle7-60-eval`
   - reward: `4/21 = 19.05%`
   - format errors: `0/21 = 0.0%`
   - trainable RL steps: `13/60`
   - skipped optimizer steps: `47/60`
   - trainable samples used: `304`
   - held-out actions subset: `0/13`
   - successful examples: `sharded-GSM8K/435`, `sharded-GSM8K/1187`,
     `sharded-GSM8K/140`, `sharded-GSM8K/1113`

   Interpretation: sparse terminal GRPO does not improve over the clean base
   model on the mixed split. The run is also severely starved for useful
   gradients because most rollout groups have identical terminal rewards. This
   strengthens the case for dense RLRF-style feedback and SDPO-style feedback
   distillation under the same minimal student prompt.

   Completed minimal-prompt dense/RLRF 60-step pilot:

   - training run: `lost-math-actions-dense-rlrf-minimal-shuffle7-60`
   - eval run: `lost-math-actions-dense-rlrf-minimal-shuffle7-60-eval`
   - reward: `5/21 = 23.81%`
   - format errors: `1/21 = 4.76%`
   - trainable RL steps: `48/60`
   - skipped optimizer steps: `12/60`
   - trainable samples used: `596`
   - held-out math subset: `5/8 = 62.5%`
   - held-out actions subset: `0/13 = 0.0%`
   - successful examples: `sharded-GSM8K/435`, `sharded-GSM8K/1187`,
     `sharded-GSM8K/799`, `sharded-GSM8K/140`, `sharded-GSM8K/1113`

   Interpretation: dense/RLRF is the first clean-prompt method to beat both
   base Qwen and sparse GRPO on the mixed held-out split (`+1/21`). It also
   produces much more training signal than sparse terminal reward
   (`48/60` trainable steps versus `13/60`). The gain is still limited to math
   examples; every held-out actions example remains unsolved. This means the
   result supports the dense-feedback direction, but the next decisive test is
   SDPO-style feedback distillation or a stronger action-specific feedback
   signal under the same minimal student prompt.

   Completed minimal-prompt SDPO 60-step pilot:

   - training run: `lost-math-actions-sdpo-brief-w01-skip3-minimal-shuffle7-60`
   - eval run: `lost-math-actions-sdpo-brief-w01-skip3-minimal-shuffle7-60-eval`
   - reward: `3/21 = 14.29%`
   - format errors: `3/21 = 14.29%`
   - trainable steps: `60/60`
   - skipped optimizer steps: `0/60`
   - RL samples used: `600`
   - SDPO distillation samples used: `179`
   - held-out math subset: `3/8 = 37.5%`
   - held-out actions subset: `0/13 = 0.0%`
   - successful examples: `sharded-GSM8K/359`, `sharded-GSM8K/140`,
     `sharded-GSM8K/1113`

   Interpretation: conservative top-k SDPO gives the best training coverage but
   the worst clean-prompt held-out result in this mixed pilot. The current
   feedback-distillation target is therefore not reliably transferring to
   better final-task behavior. The strongest current result remains dense/RLRF
   without SDPO (`5/21`), and the remaining bottleneck is the actions subset,
   where all methods are still `0/13`.

   Action-schema correction:

   The first mixed split was useful for finding a real data-interface problem:
   action rows expected BFCL-style JSON function calls, but the converted
   `sharded_multiturn` rows did not preserve the upstream `function` schemas.
   For example, `sharded-BFCL/parallel_144` initially asks "What is the
   factorial of 5?" while the reference answer is a `math.factorial` JSON call.
   Without the available function schema, natural-language answers are
   reasonable but still scored wrong.

   The converter now preserves upstream action metadata as `functions`,
   `language`, and `test_category`; action rows convert to `kind=tool_call`; and
   `SHARDED_PROMPT_STYLE=tool_schema` restores the function schema and expected
   JSON-call output format without revealing hidden shards. A corrected split
   was generated at `datasets/sharded_multiturn/lost_math_actions_tools_200`.
   It preserves the same train/test IDs and task mix as
   `lost_math_actions_200`:

   - train rows: `187` (`95` math, `92` actions)
   - test rows: `21` (`8` math, `13` actions)
   - action rows with function schemas: `92/92` train, `13/13` test

   The previous clean-prompt mixed results should be reported as an important
   diagnostic, not as the final action-task comparison. The next fair mixed
   comparison should rerun base, sparse, dense, and SDPO on
   `lost_math_actions_tools_200` with `SHARDED_PROMPT_STYLE=tool_schema`.

   Initial corrected-split base eval:

   - run: `lost-math-actions-tools-base-test`
   - observed reward before tool-call wrapper parsing fix: `3/21 = 14.29%`
   - observed format errors: `6/21 = 28.57%`
   - successful examples: `sharded-GSM8K/799`, `sharded-GSM8K/140`,
     `sharded-GSM8K/1113`

   Follow-up inspection showed Qwen often emitted a common JSON tool-call shape
   such as `{"function.name": "math.factorial", "arguments": {"number": 5}}`
   instead of the canonical `{ "math.factorial": {"number": [5]} }` shape.
   The scorer now accepts both shapes, and the prompt example was changed to
   use a concrete function name instead of the ambiguous literal placeholder
   `"function.name"`. Rescoring the same saved base samples locally after that
   fix gives `5/21`, including `2/13` action rows. The next logged Tinker eval
   should rerun the base command under a new run name so the official metrics
   reflect the updated scorer and prompt.

   Corrected-scorer base eval:

   - run: `lost-math-actions-tools-base-fixed-scorer-test`
   - reward: `5/21 = 23.81%`
   - format errors: `5/21 = 23.81%`
   - math subset: `3/8 = 37.5%`
   - actions subset: `2/13 = 15.38%`
   - successful action examples: `sharded-BFCL/parallel_144`,
     `sharded-BFCL/parallel_195`
   - successful math examples: `sharded-GSM8K/799`, `sharded-GSM8K/140`,
     `sharded-GSM8K/1113`

   This is the new fair baseline for the corrected mixed math+actions split.
   The next sparse, dense/RLRF, SDPO, and SFT comparisons should use this
   dataset, prompt style, scoring code, and run as the base reference.

   Completed corrected-split sparse-GRPO baseline:

   - training run: `lost-math-actions-tools-sparse-grpo-shuffle7-60`
   - eval run: `lost-math-actions-tools-sparse-grpo-shuffle7-60-eval`
   - reward: `4/21 = 19.05%`
   - format errors: `5/21 = 23.81%`
   - trainable RL steps: `20/60`
   - skipped optimizer steps: `40/60`
   - trainable samples used: `472`
   - math subset: `2/8 = 25.0%`
   - actions subset: `2/13 = 15.38%`
   - successful action examples: `sharded-BFCL/parallel_144`,
     `sharded-BFCL/parallel_195`
   - successful math examples: `sharded-GSM8K/799`, `sharded-GSM8K/1113`

   Interpretation: sparse GRPO still regresses relative to the corrected base
   (`4/21` versus `5/21`). It preserves the two base action successes but loses
   one math example, and two-thirds of optimizer steps are skipped because the
   terminal reward often produces no non-zero advantages. This keeps dense/RLRF
   as the main method to evaluate next.

   Completed corrected-split dense/RLRF baseline:

   - training run: `lost-math-actions-tools-dense-rlrf-shuffle7-60`
   - eval run: `lost-math-actions-tools-dense-rlrf-shuffle7-60-eval`
   - reward: `4/21 = 19.05%`
   - format errors: `7/21 = 33.33%`
   - trainable RL steps: `38/60`
   - skipped optimizer steps: `22/60`
   - trainable samples used: `448`
   - math subset: `2/8 = 25.0%`
   - actions subset: `2/13 = 15.38%`
   - successful action examples: `sharded-BFCL/parallel_144`,
     `sharded-BFCL/parallel_195`
   - successful math examples: `sharded-GSM8K/799`, `sharded-GSM8K/140`

   Interpretation: dense/RLRF produces more trainable update steps than sparse
   (`38/60` versus `20/60`), but it still regresses against the corrected base
   held-out score and increases format errors. On this corrected mixed split,
   the current dense reward is not sufficient to improve final task accuracy.
   This shifts the next project step toward a supervised sharded baseline and/or
   SDPO-style feedback distillation under the corrected tool-schema interface.

   Corrected-split reporting split:

   - Math-sharded subset (`8` held-out examples): base Qwen is currently
     strongest at `3/8`; sparse GRPO and dense/RLRF are both `2/8`. This is the
     subset closest to the original multi-turn clarification proposal because
     hidden shards are missing facts needed to solve a final answer.
   - Action/tool-sharded subset (`13` held-out examples): all corrected runs so
     far are tied at `2/13`. This subset is a useful extension, but it is not
     identical to the math clarification setup because many action shards are
     incremental tool-call requests. The current dense rubric rewards asking for
     information and penalizes intermediate final/tool-call outputs, which may
     conflict with the natural action-task behavior.
   - Mixed aggregate (`21` held-out examples): corrected base is `5/21`, sparse
     GRPO is `4/21`, and dense/RLRF is `4/21`. The aggregate should be reported
     as a small corrected pilot, not as conclusive evidence against the project
     idea. Each held-out example changes the aggregate by `4.76` percentage
     points, so larger evals or repeated seeds are needed for stronger claims.

   Max-token sensitivity re-eval:

   The 256-token eval budget was likely truncating some verbose Qwen math
   answers before the final answer. Re-evaluating the corrected base, sparse
   GRPO, and dense/RLRF checkpoints with `MAX_TOKENS=512` changes the main
   corrected-pilot conclusion:

   - base run: `lost-math-actions-tools-base-fixed-scorer-test-512`
     - reward: `5/21 = 23.81%`
     - format errors: `6/21 = 28.57%`
     - math subset: `3/8 = 37.5%`
     - actions subset: `2/13 = 15.38%`
   - sparse run: `lost-math-actions-tools-sparse-grpo-shuffle7-60-eval-512`
     - reward: `5/21 = 23.81%`
     - format errors: `7/21 = 33.33%`
     - math subset: `3/8 = 37.5%`
     - actions subset: `2/13 = 15.38%`
   - dense/RLRF run: `lost-math-actions-tools-dense-rlrf-shuffle7-60-eval-512`
     - reward: `6/21 = 28.57%`
     - format errors: `4/21 = 19.05%`
     - math subset: `4/8 = 50.0%`
     - actions subset: `2/13 = 15.38%`

   Interpretation: with the less truncating eval budget, dense/RLRF is the
   strongest corrected run so far. The gain comes from the math-sharded subset,
   which is closest to the proposal's underspecified multi-turn clarification
   setting. Sparse GRPO ties base accuracy but has worse format errors. The
   action/tool subset remains tied across methods, suggesting the current
   clarify-until-all-shards dense rubric is not yet well matched to incremental
   tool-call tasks.

   Completed corrected-split SFT baseline:

   - training run: `lost-math-actions-tools-sft-shuffle7-60`
   - eval run: `lost-math-actions-tools-sft-shuffle7-60-eval-512`
   - reward: `2/21 = 9.52%`
   - format errors: `11/21 = 52.38%`
   - supervised training steps: `60/60`
   - math subset: `2/8 = 25.0%`
   - actions subset: `0/13 = 0.0%`
   - successful examples: `sharded-GSM8K/799`, `sharded-GSM8K/140`

   Interpretation: the simple SFT baseline does not validate the current
   demonstration recipe. It trains on generic clarification targets such as
   "Could you provide the next missing detail?" and final-answer targets after
   all shards are revealed, but this harms held-out behavior relative to base
   and dense/RLRF, especially on action/tool rows. The result suggests that
   supervised feedback/distillation needs task-aware targets or a better
   teacher, rather than generic clarification text.

7. Run a minimal local/JHU smoke test.

   The Tinker work is separate from the local/JHU `verl` path. Before merging or
   depending on this branch broadly, run a short JHU job to confirm the original
   path still works.
