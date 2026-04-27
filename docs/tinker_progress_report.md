# Tinker Progress Report

Date: 2026-04-27  
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

The work so far does not implement the final RLRF method yet. It makes progress
by getting managed on-policy RL, checkpointing, and held-out evaluation working
on Tinker without breaking the existing local/JHU `verl` path. That gives us a
baseline harness and enough operational confidence to run sparse-GRPO
comparisons before adding dense-feedback logic.

## Code Changes So Far

Tinker support was added as a separate path from the existing local/JHU `verl`
training path.

New files:

- `requirements-tinker.txt`: optional Tinker dependencies.
- `run_tinker_grpo.sh`: launcher for Tinker GRPO training.
- `run_tinker_eval.sh`: launcher for Tinker held-out evaluation.
- `scripts/tinker_grpo.py`: minimal GRPO-style training loop using Tinker
  sampling, forward/backward, optimizer steps, and checkpoint storage.
- `scripts/tinker_eval.py`: evaluates a base model or Tinker checkpoint against
  this repo's JSON datasets.
- `tests/test_tinker_grpo_helpers.py`: focused tests for data loading, reward
  wiring, ToolUse nested JSON parsing, SciKnowEval format diagnostics, and
  Tinker checkpoint path detection.

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
- Training/eval logs are written to `_logs/tinker_grpo` and `_logs/tinker_eval`.
  The trainer now clears same-run JSONL files at startup to avoid appending
  stale results when reusing run names.
- The API key is read only from `TINKER_API_KEY`; it is not written to tracked
  files.

## Verification

Local checks currently pass:

```bash
./.venv/bin/python -m py_compile scripts/tinker_grpo.py scripts/tinker_eval.py tests/test_tinker_grpo_helpers.py
./.venv/bin/python -m pytest tests/test_tinker_grpo_helpers.py
bash -n run_tinker_grpo.sh run_tinker_eval.sh
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

- no sharded multi-turn conversation environment is wired into Tinker yet;
- no dense RLRF/self-critic reward has been implemented;
- no CURIO-style curiosity baseline has been run;
- no model-scale sweep has been run;
- no OOD evaluation has been run;
- local/JHU end-to-end execution still needs a real cluster smoke test.

## Next Steps

1. Decide whether to implement concise-answer prompting before
   more sparse-GRPO runs.

   Chemistry shows that cleaner format alone does not guarantee accuracy gains.
   A concise prompt may reduce variance and make sparse reward less noisy.

2. Add a concise-answer mode for SciKnowEval.

   The current system prompt encourages reasoning and XML output. Many failures
   are still truncation. A prompt variant that asks for brief reasoning plus
   `<answer>` may reduce format errors without changing the reward function.

3. Wire the sharded multi-turn dataset.

   The final-project dataset needs turns, hidden shards, and an environment that
   reveals shards only when the model asks useful clarifying questions. This is
   the real bridge from single-turn RLVR-style tasks to the final multi-turn
   research question.

4. Implement dense-feedback/RLRF.

   Once sparse GRPO baselines are stable, add the teacher/self-critic signal
   conditioned on privileged/full task information. The output should be a
   denser reward signal than terminal correctness, so we can compare convergence
   and sample efficiency against sparse GRPO.

5. Run a minimal local/JHU smoke test.

   The Tinker work is separate from the local/JHU `verl` path. Before merging or
   depending on this branch broadly, run a short JHU job to confirm the original
   path still works.
