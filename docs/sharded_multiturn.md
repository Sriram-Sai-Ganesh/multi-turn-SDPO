# Sharded Multi-Turn Task

This is the first final-project task interface. It models the Laban et al.
setting where a fully specified instruction is split into hidden shards, and
the model must ask for missing information across turns before answering.

## JSON Format

Each row is a JSON object:

```json
{
  "idx": "example-id",
  "dataset": "sharded_multiturn",
  "kind": "number",
  "prompt": "The underspecified user request.",
  "shards": ["Hidden fact 1.", "Hidden fact 2."],
  "answer": "42",
  "full_prompt": "Optional fully specified instruction for later RLRF."
}
```

Supported `kind` values:

- `exact`: normalized string match.
- `number` or `numeric`: final numeric answer match.
- `mcq`: multiple-choice letter match.
- `contains`: reference answer must appear in the final answer.

The model is instructed to ask clarifying questions. The environment reveals
one shard after each non-final assistant turn. A final response is detected from
`<final>answer</final>`, `<final-answer>answer</final-answer>`, or
`<answer>answer</answer>`, or `Final answer: ...`. Final-tagged clarifying
questions and placeholder tags like `<final>...</final>` do not terminate the
conversation. This prevents models from skipping shard revelation by either
wrapping a clarification in final tags or echoing the instruction placeholder.
Missing final-answer formatting gets zero reward and a format feedback string.

Assistant responses that contain strings like `Additional information 2/5:` are
treated as invalid format and terminate the trajectory immediately. Only the
user/environment may reveal hidden shards; the assistant must not invent or
write shard messages itself. Current shard prompts pass only the hidden shard
text plus a short continuation instruction; the guard still catches earlier
labeled phrasings such as `Additional information i/n:` and
`User-provided detail i of n:`.

## Tinker Sparse GRPO

The existing Tinker runner now handles `dataset == "sharded_multiturn"` rows by
sampling a full multi-turn trajectory. It applies the terminal sparse reward to
every assistant turn from that trajectory.

Smoke training on the bundled tiny dataset:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/smoke \
RENDERER_NAME=qwen3_disable_thinking \
MAX_STEPS=1 \
BATCH_SIZE=1 \
ROLLOUT_N=2 \
MAX_TURNS=3 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh sharded-smoke
```

Base eval:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/smoke \
SPLIT=test \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=3 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh sharded-smoke-base-test
```

First observed Tinker training smoke:

- run: `sharded-smoke`
- checkpoint: `tinker://1aa60164-675f-57e6-8586-b55838c662a3:train:0/sampler_weights/sharded-smoke-final-sampler`
- reward: `2/2` rollouts succeeded
- trajectory shape: `3` assistant turns per rollout, `2` shards revealed
- training datums: `6`

This validates remote multi-turn sampling, shard revelation, transcript logging,
terminal scoring, and Tinker checkpointing. It does not validate learning signal
because both rollouts were correct, so GRPO advantages were zero.

First observed Tinker checkpoint eval:

- run: `sharded-smoke-checkpoint-test`
- reward: `0/1`
- format errors: `0/1`
- failure mode: the model wrote a clarifying question inside `<final>...</final>`
  on the first turn, so the original environment stopped before revealing any
  shards.

The final-answer parser now treats final-tagged clarifying questions as
non-final turns. Re-run eval after this fix to verify that the environment
reveals shards instead of ending early.

Second observed Tinker checkpoint eval:

- run: `sharded-smoke-checkpoint-test`
- reward: `0/1`
- format errors: `0/1`
- trajectory shape: `2` assistant turns, `1` shard revealed
- failure mode: after shard 1, the model echoed the instruction placeholder
  `reply with <final>...</final>`, which the parser treated as a final answer
  with prediction `...`.

The shard prompts no longer include literal `<final>...</final>` placeholders,
and the parser now ignores placeholder final tags.

Third observed Tinker checkpoint eval:

- run: `sharded-smoke-checkpoint-test`
- reward: `0/1`
- format errors: `1/1`
- trajectory shape: `3` assistant turns, `2` shards revealed
- failure mode: the model answered correctly with
  `<final-answer>The product of 4 and 6 is 24.</final-answer>`, but the parser
  only accepted `<final>...</final>`.

The parser now accepts both `<final>...</final>` and
`<final-answer>...</final-answer>`.

Final observed Tinker checkpoint eval after parser fixes:

- run: `sharded-smoke-checkpoint-test`
- reward: `1/1`
- format errors: `0/1`
- trajectory shape: `3` assistant turns, `2` shards revealed
- final response: `<final-answer>The product of 4 and 6 is 24.</final-answer>`

This confirms the full sharded multi-turn Tinker loop works on the smoke task:
the model asks for missing information, the environment reveals shards across
turns, the model answers after receiving all shards, and the scorer accepts the
final answer.

## Dense RLRF-Style Reward Mode

The final-project proposal asks whether richer feedback can improve the
multi-turn information-gathering behavior that sparse terminal reward fails to
teach reliably. This follows the SDPO/RLRF framing from Hübotter et al., where
feedback supplies denser credit assignment than a single scalar outcome, and it
targets the Laban et al. failure mode where models make early assumptions and
prematurely answer in sharded conversations.

The Tinker trainer therefore supports an opt-in dense sharded reward mode:

```bash
SHARDED_REWARD_MODE=dense ./run_tinker_grpo.sh my-dense-run
```

Aliases accepted by the runner: `dense`, `rlrf`, `rich_feedback`, `turn`, and
`turn_level`. The default remains `sparse`, so all existing base and sparse-GRPO
baseline commands stay reproducible.

Dense mode uses the hidden shard schedule as privileged teacher context. It
does not reveal hidden content to the policy. Instead, each assistant turn gets
a training score:

- `0.25` for a clarifying/information-seeking turn while hidden shards remain;
- `0.0` for a premature final answer before all hidden shards are revealed;
- `0.0` for malformed final-answer tags or assistant-authored hidden-detail
  messages;
- normal final-answer accuracy once all hidden shards have been revealed.

For GRPO advantage construction, dense mode centers rewards by turn index within
each rollout group. This means turn-0 clarification attempts are compared
against other turn-0 attempts, final-answer attempts are compared against other
final-answer attempts, and premature short trajectories can be penalized without
assigning the same terminal score to every earlier turn.

This is an RLRF-style reward-shaping implementation, not the full SDPO
self-distillation objective. Use it as the dense-reward baseline for comparison
with the SDPO-style mode below.

Dense Lost Math pilot command:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
RENDERER_NAME=qwen3_disable_thinking \
SHARDED_REWARD_MODE=dense \
SHUFFLE_SEED=7 \
MAX_STEPS=30 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-103-dense-rlrf-shuffle7-30
```

Evaluate dense checkpoints with the normal sparse held-out accuracy metric:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
SPLIT=test \
MODEL_PATH='tinker://.../sampler_weights/lost-math-103-dense-rlrf-shuffle7-30-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-103-dense-rlrf-shuffle7-30-eval
```

## Tinker SDPO-Style Feedback Distillation

The proposal's method is stronger than dense reward shaping: it calls for an
on-policy self-teacher with access to the fully specified task to provide dense
token-level learning signal to the student. The local/JHU `verl` path already
contains the full SDPO machinery with feedback-conditioned teacher prompts and
logit KL distillation under `actor.policy_loss.loss_mode=sdpo`.

Tinker does not expose that exact `verl` loss, so the Tinker implementation uses
the closest supported objective:

- `SHARDED_REWARD_MODE=sdpo` keeps the dense turn rewards and centered
  turn-level GRPO advantages.
- For failed sharded rollouts, it builds a feedback-conditioned self-teacher
  prompt with the fully specified task, hidden shard schedule, rubric feedback,
  and any successful peer rollout from the same on-policy group.
- By default, it teacher-forces the student's sampled turn through that teacher
  prompt, requests top-k prompt logprobs from Tinker, and trains the student on
  the original conversation state with `cross_entropy` soft targets.
- `SDPO_TOPK=20`, `SDPO_DISTILL_WEIGHT=0.1`, and
  `SDPO_SKIP_FIRST_N_TOKENS=3` are the current conservative defaults. Set
  `SDPO_TOPK=0` to use a cheaper generated-target CE fallback instead of top-k
  soft targets.

This is SDPO-style feedback distillation for Tinker, not a byte-for-byte copy of
the local `verl` full-logit KL implementation. The important project distinction
is preserved: dense/RLRF mode is scalar turn reward shaping; SDPO mode adds a
feedback-conditioned self-teacher distillation term.

Conservative smoke command:

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

Observed SDPO top-k smoke:

- run: `lost-math-103-sdpo-topk-shuffle7-smoke`
- train rows used: `5`
- optimizer steps with trainable signal: `4/5`
- skipped constant-signal steps: `1/5`
- dense/GRPO assistant-turn datums used: `30`
- SDPO top-k distillation datums used: `7`
- SDPO top-k token positions used: `362`
- final sampler checkpoint:
  `tinker://eee8cb04-5b63-54ce-8138-917dac3ea139:train:0/sampler_weights/lost-math-103-sdpo-topk-shuffle7-smoke-final-sampler`

The smoke validates that Tinker can run the combined objective. In top-k mode
the log field `teacher_response` is intentionally empty because the teacher is
not sampled for text; it is teacher-forced over the student's sampled turn and
used for soft next-token targets.

Observed SDPO top-k smoke eval:

- run: `lost-math-103-sdpo-topk-shuffle7-smoke-eval`
- checkpoint:
  `tinker://eee8cb04-5b63-54ce-8138-917dac3ea139:train:0/sampler_weights/lost-math-103-sdpo-topk-shuffle7-smoke-final-sampler`
- reward: `4/10 = 40.0%`
- format errors: `2/10 = 20.0%`

The smoke checkpoint tied the sparse-GRPO 30-step baseline and trailed base
Qwen plus dense 30/60-step checkpoints by `1/10`. This is acceptable for a
smoke because the purpose was to validate the combined objective, not to measure
learning.

Conservative pilot command:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
RENDERER_NAME=qwen3_disable_thinking \
SHARDED_REWARD_MODE=sdpo \
SDPO_TOPK=20 \
SDPO_DISTILL_WEIGHT=0.1 \
SDPO_SKIP_FIRST_N_TOKENS=3 \
SHUFFLE_SEED=7 \
MAX_STEPS=30 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-103-sdpo-topk-w01-skip3-shuffle7-30
```

Observed SDPO top-k 30-step training:

- train rows used: `30`
- optimizer steps with trainable signal: `27/30`
- skipped constant-signal steps: `3/30`
- dense/GRPO assistant-turn datums used: `184`
- SDPO top-k distillation datums used: `66`
- SDPO top-k token positions used: `3487`
- steps with SDPO distillation signal: `22/30`
- mean training reward across steps: `0.2462`
- final sampler checkpoint:
  `tinker://3a2b98dd-8862-5190-934c-66ba227dacc5:train:0/sampler_weights/lost-math-103-sdpo-topk-shuffle7-30-final-sampler`

The 30-step run completed despite one non-fatal telemetry connection warning.
The warning did not stop training, and both final Tinker state and sampler
checkpoints were saved.

Observed SDPO top-k 30-step eval:

- run: `lost-math-103-sdpo-topk-shuffle7-30-eval`
- checkpoint:
  `tinker://3a2b98dd-8862-5190-934c-66ba227dacc5:train:0/sampler_weights/lost-math-103-sdpo-topk-shuffle7-30-final-sampler`
- reward: `4/10 = 40.0%`
- format errors: `2/10 = 20.0%`
- comparison to sparse GRPO: tied
- comparison to base and dense 30/60-step checkpoints: `-1/10`
- regressed example relative to base/dense: `sharded-GSM8K/1027`

Interpretation: the first SDPO top-k configuration produced strong token-level
training signal but did not improve held-out accuracy. It also regressed the
same premature-answer example that dense reward shaping had fixed. Because
top-k CE loss has many more supervised token positions than scalar GRPO, the
next run should use a smaller distillation weight and skip the first few forced
tokens.

Evaluate the resulting sampler checkpoint with the same sparse held-out metric:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
SPLIT=test \
MODEL_PATH='tinker://.../sampler_weights/lost-math-103-sdpo-topk-shuffle7-30-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-103-sdpo-topk-shuffle7-30-eval
```

## Local/JHU Preprocessing

The standard parquet preprocessing path accepts the same JSON schema:

```bash
python data/preprocess.py --data_source datasets/sharded_multiturn/smoke
```

On clusters, set `HF_HOME` or `HF_DATASETS_CACHE` to a writable scratch path if
the default Hugging Face cache is not writable from the job container.

## Lost In Conversation Conversion

The official Laban et al. dataset is `microsoft/lost_in_conversation` on
Hugging Face. It stores rows with `task_id`, `task`, ordered `shards`, and
task-specific evaluation fields. The converter maps this into our
`sharded_multiturn` schema by using shard 1 as the initial user message and
keeping later shards hidden for turn-by-turn revelation.

Convert a small math pilot:

```bash
HF_HOME=/tmp/hf_home \
HF_DATASETS_CACHE=/tmp/hf_datasets \
python scripts/convert_lost_in_conversation.py \
  --source microsoft/lost_in_conversation \
  --task math \
  --output-dir datasets/sharded_multiturn/lost_math \
  --max-records 40 \
  --train-ratio 0.9 \
  --seed 42
```

Observed pilot conversion:

- input rows: `627`
- task filter: `math`
- converted rows: `40`
- train/test: `36/4`
- hidden shards per converted row: `3` to `11`
- local parquet preprocessing: passed

Run a base Tinker eval on the converted pilot:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math \
SPLIT=test \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-base-test
```

Observed base Tinker eval:

- run: `lost-math-base-test`
- reward: `2/4 = 50.0%`
- format errors: `1/4 = 25.0%`
- trajectory lengths: `4`, `4`, `7`, and `8` assistant turns
- all test rows revealed every hidden shard

One failure surfaced a parser bug: a truncated response ending with
`Final answer:\n<final` was treated as a valid prefix answer. The prefix parser
now only accepts `Final answer: ...` when the answer text appears on the same
line and is not a placeholder.

Run a short sparse-GRPO smoke after the base eval:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math \
RENDERER_NAME=qwen3_disable_thinking \
MAX_STEPS=5 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-grpo-smoke
```

Observed sparse-GRPO smoke:

- run: `lost-math-grpo-smoke`
- train rows used: `5`
- rollout groups: `5`
- rollouts: `20`
- rewarded rollouts: `7/20 = 35.0%`
- format errors: `4/20 = 20.0%`
- mixed reward groups: `2/5`
- final sampler checkpoint:
  `tinker://80b32044-f890-542a-acab-d61bdd131d0e:train:0/sampler_weights/lost-math-grpo-smoke-final-sampler`

This is a real sparse-GRPO signal test. Unlike the tiny smoke task, two rollout
groups had both successes and failures, so the trainer had nonzero advantages on
real sharded multi-turn math examples.

Evaluate the smoke checkpoint:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math \
SPLIT=test \
MODEL_PATH='tinker://80b32044-f890-542a-acab-d61bdd131d0e:train:0/sampler_weights/lost-math-grpo-smoke-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-grpo-smoke-test
```

Observed checkpoint eval:

- run: `lost-math-grpo-smoke-test`
- reward: `2/4 = 50.0%`
- logged format errors before the impersonation guard: `1/4 = 25.0%`
- comparison to base: `1` improved, `1` regressed, `1` unchanged correct, `1`
  unchanged wrong
- important failure mode: one regressed sample invented an
  `Additional information 3/3:` shard and answered from that hallucinated
  hidden fact before the environment revealed it

The aggregate score matches the base pilot, so this smoke should not be treated
as a learning win. It does show that the remote sparse-GRPO loop can train and
evaluate sharded multi-turn trajectories. The environment now rejects
assistant-authored `Additional information ...` messages as invalid format
before larger runs, so rerunning this same sample would count that regression as
a format error rather than a normal wrong final answer. A local rescore of the
saved JSONL samples with the patched scorer keeps reward at `2/4` and increases
format errors from `1/4` to `2/4`, with `1` explicit environment-impersonation
failure.

## Next Implementation Step

The sparse multi-turn path and the first Laban math pilot conversion are now in
place. The immediate next step is to evaluate the guarded Lost Math checkpoint
below, then scale the conversion beyond `--max-records 40` if the held-out
result and reward signal are still usable.

The dense RLRF/self-critic objective should use `full_prompt` as privileged
teacher context once sparse GRPO has a real multi-turn baseline.

## Guarded Lost Math Smoke

After adding the environment-impersonation guard, rerun the short sparse-GRPO
smoke:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math \
RENDERER_NAME=qwen3_disable_thinking \
MAX_STEPS=5 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-grpo-guard-smoke
```

Observed guarded smoke:

- run: `lost-math-grpo-guard-smoke`
- train rows used: `5`
- rollouts: `20`
- rewarded rollouts: `11/20 = 55.0%`
- format errors: `7/20 = 35.0%`
- environment-impersonation failures: `0/20`
- mixed reward groups: `2/5`
- all-success groups: `2/5`
- all-fail groups: `1/5`
- final sampler checkpoint:
  `tinker://b1a02fc0-91ed-508d-82bc-d0a0f42d08e4:train:0/sampler_weights/lost-math-grpo-guard-smoke-final-sampler`

The guard did not fire on this small training run, and the run still produced
mixed reward groups. Evaluate the guarded checkpoint before scaling.

Guarded checkpoint eval:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math \
SPLIT=test \
MODEL_PATH='tinker://b1a02fc0-91ed-508d-82bc-d0a0f42d08e4:train:0/sampler_weights/lost-math-grpo-guard-smoke-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-grpo-guard-smoke-test
```

Observed guarded checkpoint eval:

- logged reward before accepting `<answer>...</answer>`: `1/4 = 25.0%`
- logged format errors before accepting `<answer>...</answer>`: `3/4 = 75.0%`
- local rescore with the corrected parser: `2/4 = 50.0%`
- local rescored format errors: `2/4 = 50.0%`
- environment-impersonation trajectories: `1/4`

This still does not beat the base pilot. The next run should focus on reducing
format and shard-impersonation failures before scaling to a larger split.

## User-Detail Prompt Smoke

Changing the shard label from `Additional information i/n:` to
`User-provided detail i of n:` improved the base format behavior but did not
fully solve copying.

Observed base eval:

- run: `lost-math-base-user-detail-test`
- reward: `2/4 = 50.0%`
- format errors: `0/4`
- failures: two early final answers before enough hidden shards were revealed

Observed GRPO smoke:

- run: `lost-math-grpo-user-detail-smoke`
- train rows used: `5`
- rollouts: `20`
- rewarded rollouts: `9/20 = 45.0%`
- format errors: `7/20 = 35.0%`
- environment-impersonation failures: `4/20`
- mixed reward groups: `2/5`
- all-success groups: `1/5`
- all-fail groups: `2/5`
- final sampler checkpoint:
  `tinker://00149cfc-aace-5243-9622-99496d3d317b:train:0/sampler_weights/lost-math-grpo-user-detail-smoke-final-sampler`

The impersonation failures all came from one training group where the assistant
copied the `User-provided detail ...` label and wrote the next hidden fact
itself. The environment now uses unlabeled shard messages to avoid giving the
model a copyable user-message template.

## Unlabeled Shard Smoke

Removing labels from hidden-shard messages kept the held-out base behavior
clean and removed shard impersonation from the smoke train.

Observed base eval:

- run: `lost-math-base-unlabeled-test`
- reward: `2/4 = 50.0%`
- format errors: `0/4`
- failures: two early final answers before enough hidden shards were revealed

Observed GRPO smoke:

- run: `lost-math-grpo-unlabeled-smoke`
- train rows used: `5`
- rollouts: `20`
- rewarded rollouts: `12/20 = 60.0%`
- format errors: `4/20 = 20.0%`
- environment-impersonation failures: `0/20`
- mixed reward groups: `0/5`
- all-success groups: `3/5`
- all-fail groups: `2/5`
- final sampler checkpoint:
  `tinker://72709eee-9d32-57a3-8701-327e9c143824:train:0/sampler_weights/lost-math-grpo-unlabeled-smoke-final-sampler`

This is the cleanest sharded environment so far, but it produced no mixed
reward groups. The Tinker losses were all zero, so this checkpoint should be
treated as a no-op training run rather than a learned policy. The runner now
skips zero-advantage samples so future no-op runs report `samples_used=0` and
skip optimizer work.

To search for real sparse-GRPO signal without always using the same first five
train rows, the Tinker trainer supports `SHUFFLE_SEED`. Negative values keep the
original deterministic order.

## Shuffled Unlabeled Smoke

Using the unlabeled shard environment with a shuffled 10-row slice produced the
first clean sharded Lost Math run with real sparse-GRPO signal.

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math \
RENDERER_NAME=qwen3_disable_thinking \
SHUFFLE_SEED=7 \
MAX_STEPS=10 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-grpo-unlabeled-shuffle7-10
```

Observed train run:

- run: `lost-math-grpo-unlabeled-shuffle7-10`
- train rows used: `10`
- rollouts: `40`
- rewarded rollouts: `16/40 = 40.0%`
- format errors: `12/40 = 30.0%`
- environment-impersonation failures: `0/40`
- mixed reward groups: `6/10`
- all-success groups: `1/10`
- all-fail groups: `3/10`
- optimizer steps with nonzero advantages: `6/10`
- skipped zero-advantage steps: `4/10`
- final sampler checkpoint:
  `tinker://b7977a46-7133-5ee8-ab83-d5cc679f5754:train:0/sampler_weights/lost-math-grpo-unlabeled-shuffle7-10-final-sampler`

This is the first sharded Lost Math run worth evaluating as a sparse-GRPO
checkpoint. Unlike the prior unlabeled smoke, it has mixed reward groups and
nonzero Tinker losses while preserving `0` environment-impersonation failures.

Checkpoint eval:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math \
SPLIT=test \
MODEL_PATH='tinker://b7977a46-7133-5ee8-ab83-d5cc679f5754:train:0/sampler_weights/lost-math-grpo-unlabeled-shuffle7-10-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-grpo-unlabeled-shuffle7-10-test
```

Observed checkpoint eval:

- reward: `2/4 = 50.0%`
- format errors: `0/4`
- comparison to base: same `2` successes and same `2` failures
- main remaining failure mode: premature final answers before enough hidden
  shards are revealed

The sparse-GRPO checkpoint ties the clean base on this tiny held-out split. It
does not yet show held-out improvement, but the environment and runner are now
stable enough to scale the pilot.

## Scaled Lost Math Split

The first scaled split keeps the same `math` task filter but raises the
conversion cap to `--max-records 200`:

```bash
HF_HOME=/tmp/hf_home \
HF_DATASETS_CACHE=/tmp/hf_datasets \
python scripts/convert_lost_in_conversation.py \
  --source microsoft/lost_in_conversation \
  --task math \
  --output-dir datasets/sharded_multiturn/lost_math_200 \
  --max-records 200 \
  --train-ratio 0.9 \
  --seed 42
```

The cached upstream source yielded `103` convertible math rows:

- output directory: `datasets/sharded_multiturn/lost_math_200`
- train/test: `93/10`
- reward kind: `number`
- hidden shards per row: `3` to `11`
- mean hidden shards: `4.68`
- local parquet preprocessing: passed

Run the scaled base eval:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
SPLIT=test \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-103-base-unlabeled-test
```

Observed scaled base eval:

- run: `lost-math-103-base-unlabeled-test`
- reward: `5/10 = 50.0%`
- format errors: `2/10 = 20.0%`

Then run a scaled sparse-GRPO pilot:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
RENDERER_NAME=qwen3_disable_thinking \
SHUFFLE_SEED=7 \
MAX_STEPS=30 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-103-grpo-unlabeled-shuffle7-30
```

If the 30-step run has enough mixed reward groups and nonzero updates, evaluate
its final sampler checkpoint on the same 10-row test split.

Observed clean 30-step scaled sparse-GRPO pilot from terminal output:

- run: `lost-math-103-grpo-unlabeled-shuffle7-30`
- train rows used: `30`
- terminal reward means indicate `53/120 = 44.2%` rewarded rollouts
- optimizer steps with nonzero advantages: `10/30`
- skipped zero-advantage steps: `20/30`
- assistant-turn datums used for nonzero-advantage updates: `212`
- final sampler checkpoint from the completed run:
  `tinker://0e50e3c3-b7ea-5d28-9016-194b29aedb57:train:0/sampler_weights/lost-math-103-grpo-unlabeled-shuffle7-30-final-sampler`

A second run was accidentally started with the same run name before the first
finished. Same-run local JSONL files are therefore interleaved and should not be
used for aggregate analysis. Future Tinker GRPO launches create a run lock so a
concurrent duplicate run name fails fast instead of corrupting logs.

Scaled checkpoint eval:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
SPLIT=test \
MODEL_PATH='tinker://0e50e3c3-b7ea-5d28-9016-194b29aedb57:train:0/sampler_weights/lost-math-103-grpo-unlabeled-shuffle7-30-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-103-grpo-unlabeled-shuffle7-30-eval
```

Observed eval:

| Run | Correct | Accuracy | Format Errors |
| --- | ---: | ---: | ---: |
| Base Qwen3 unlabeled | `5/10` | `50.0%` | `2/10` |
| Sparse GRPO 30-step | `4/10` | `40.0%` | `2/10` |
| Dense RLRF 30-step | `5/10` | `50.0%` | `2/10` |
| Dense RLRF 60-step | `5/10` | `50.0%` | `2/10` |

Example-level comparison:

- improved examples: `0`
- regressed examples: `1`
- unchanged correct examples: `4`
- unchanged wrong examples: `5`

The regressed example is `sharded-GSM8K/1027`: base waited for more hidden
shards and answered `$12`, while the sparse-GRPO checkpoint answered too early
with `$8`. This reinforces the main final-project hypothesis: sparse terminal
reward alone is not enough to reliably teach multi-turn information gathering
and can worsen premature final-answer behavior.

Observed dense/RLRF 30-step pilot:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
RENDERER_NAME=qwen3_disable_thinking \
SHARDED_REWARD_MODE=dense \
SHUFFLE_SEED=7 \
MAX_STEPS=30 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-103-dense-rlrf-shuffle7-30
```

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

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
SPLIT=test \
MODEL_PATH='tinker://c5134071-73d9-57e7-873b-12cb184f0d7f:train:0/sampler_weights/lost-math-103-dense-rlrf-shuffle7-30-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-103-dense-rlrf-shuffle7-30-eval
```

- reward: `5/10 = 50.0%`
- format errors: `2/10 = 20.0%`
- comparison to base: same `5` successes and same `5` failures
- comparison to sparse GRPO: fixes the sparse regression on
  `sharded-GSM8K/1027`

Interpretation: dense/RLRF-style shaping gives much denser update signal than
sparse GRPO (`24/30` nonzero steps versus `10/30`) and removes the one held-out
regression caused by sparse GRPO's premature answer. It does not yet improve
held-out accuracy over base on this small 10-row test split, so it should be
treated as a promising but not yet successful final-project method.

Observed dense/RLRF 60-step pilot:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
RENDERER_NAME=qwen3_disable_thinking \
SHARDED_REWARD_MODE=dense \
SHUFFLE_SEED=7 \
MAX_STEPS=60 \
BATCH_SIZE=1 \
ROLLOUT_N=4 \
MAX_TURNS=0 \
MAX_TOKENS=256 \
./run_tinker_grpo.sh lost-math-103-dense-rlrf-shuffle7-60
```

- train rows used: `60`
- optimizer steps with nonzero advantages: `40/60`
- skipped zero-advantage steps: `20/60`
- assistant-turn datums used for nonzero-advantage updates: `268`
- mean dense training reward across steps: `0.2446`
- first-half mean dense training reward: `0.2443`
- second-half mean dense training reward: `0.2449`
- final sampler checkpoint:
  `tinker://a73b0798-2115-5250-b8b2-b92c0145eaab:train:0/sampler_weights/lost-math-103-dense-rlrf-shuffle7-60-final-sampler`

Dense 60-step checkpoint eval:

```bash
PYTHON_BIN=.venv/bin/python \
DATA_PATH=datasets/sharded_multiturn/lost_math_200 \
SPLIT=test \
MODEL_PATH='tinker://a73b0798-2115-5250-b8b2-b92c0145eaab:train:0/sampler_weights/lost-math-103-dense-rlrf-shuffle7-60-final-sampler' \
MODEL_NAME=Qwen/Qwen3-8B \
RENDERER_NAME=qwen3_disable_thinking \
MAX_TURNS=0 \
MAX_TOKENS=256 \
TEMPERATURE=0.0 \
BATCH_SIZE=1 \
NUM_SAMPLES=1 \
./run_tinker_eval.sh lost-math-103-dense-rlrf-shuffle7-60-eval
```

- reward: `5/10 = 50.0%`
- format errors: `2/10 = 20.0%`
- comparison to dense 30-step: same `5` successes and same `5` failures
- comparison to base: same `5` successes and same `5` failures
- comparison to sparse GRPO: still fixes the sparse regression on
  `sharded-GSM8K/1027`

Interpretation: scaling the same dense rubric from 30 to 60 steps preserved the
sparse-regression fix but did not improve held-out accuracy. The training reward
was flat across the first and second halves, so the next project step should not
be merely running more of this exact objective.
