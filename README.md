# quodet

Quodet was created at [Sundai Hack 138](https://sundai.club). Sundai Club is a
community for building and launching AI prototypes every Sunday.

Quodet's goal is to integrate directly with coding-agent workflows and provide
extremely low-latency code-quality feedback while an agent is still working.
The independent watcher batches related file changes, reviews the current code,
and is designed to return high-confidence defects quickly enough for the coding
agent to verify and address them in the same development loop.

See [EXAMPLES.md](EXAMPLES.md) for complete buggy evaluation snippets and the
focused fixes Quodet recommended during a live review.

`quodet` recursively watches a directory. After file writes settle for three
seconds, it sends all changed files in that batch to Codex Luna through Simon
Willison's [`llm`](https://llm.datasette.io/) CLI. The default review requests
only negative findings that Luna is at least 95% confident are real defects.
It asks the model to trace a concrete execution path to an observable failure
and to discard findings that rely on speculation about missing code. Severity
is calibrated only from demonstrated impact, without assuming an unknown blast
radius.
Responses are constrained to this JSON shape:

```json
{
  "findings": [
    {
      "file": "relative/path.py",
      "line": 42,
      "severity": "medium",
      "confidence": 0.99,
      "title": "Expired cache entries remain visible",
      "explanation": "Cache.get() removes an expired entry but falls through and returns entry.value, so the triggering caller still receives stale data.",
      "suggested_fix": "In Cache.get(), return None immediately after removing the expired entry, before returning entry.value. This prevents the expired branch from exposing stale data. Add a regression test that advances the injected clock past expires_at and asserts get() returns None."
    }
  ]
}
```

When no confident negative findings exist, Luna returns `{"findings": []}`.

## Actionable recommended fixes

[PR #3](https://github.com/qfennessy/quodet/pull/3) strengthened each finding's
`suggested_fix` from an unconstrained string into a focused repair contract:

- Recommendations are bounded to 2,000 characters and name the relevant code
  element when the supplied evidence supports one.
- They connect the smallest practical behavior change to the demonstrated
  failure path and include a narrow regression test or validation step.
- When the supplied files are insufficient for a safe repair, the reviewer
  identifies the missing evidence instead of inventing architecture.
- Recommendations remain untrusted review data. Coding agents and developers
  must independently verify the diagnosis and fix; Quodet never auto-applies
  the proposed change.

PR #3 also added a frozen deterministic recommendation fixture and records the
model, prompt SHA-256, fixture revision, and case IDs for separate live-model
evaluations.

It uses `gpt-5.6-luna` with `reasoning_effort=high` by default. The installed
`llm` provider calls `high` its maximum supported reasoning level.

## Setup

Install and configure `llm` with an OpenAI key first:

```sh
uv tool install llm
llm keys set openai
```

Then install this program's Python dependency:

```sh
uv sync
```

## Run

Watch the current directory:

```sh
uv run quodet .
```

Or watch another directory and wait five seconds after the last write:

```sh
uv run quodet /path/to/project --debounce 5
```

Stop the watcher with Ctrl-C. Useful options include:

```text
--model MODEL       Select another llm model or alias
--reasoning-effort  auto, low, medium, or high
--prompt TEXT       Override the review prompt
--exclude GLOB      Ignore matching paths; may be repeated
--max-bytes BYTES   Skip larger individual files
--review-timeout S  Stop a stalled provider review (default: 60 seconds)
--poll              Poll when native filesystem events are unavailable
--log               Save requests and responses in llm's local history
```

By default, requests are not saved to the local `llm` history. Common VCS,
dependency, cache, virtual-environment, and build directories are ignored.
Binary, non-UTF-8, unreadable, and files larger than 2 MB are skipped. Eligible
text files are sanitized into private temporary fragments before being sent;
the original file and its original filename are never passed to `llm`.

Ignored environments and installed dependencies include `.venv`, `venv`,
`.tox`, `.nox`, `.direnv`, Python `site-packages`/`dist-packages`,
`__pypackages__`, `node_modules`, package-manager caches, and bundled dependency
directories. Arbitrarily named Python environments are recognized by their
`pyvenv.cfg` marker. Dependency manifests and lockfiles remain eligible because
they are development inputs. Excluded events are discarded before debouncing,
so dependency installation churn cannot trigger or delay a review batch.

## Agent-change evaluation

The frozen synthetic corpus under `evals/agent_changes` models coding-agent
changes from obvious single-file mistakes through subtle lifecycle, external
contract, authorization, concurrency, cache, tooling, atomicity, and cost
failures. [The coverage map](evals/agent_changes/COVERAGE.md) records how every
case maps to those families, its scope, evidence depth, and evaluation split.

The taxonomy comes from an independently verified, non-public review corpus.
Its source location and answer-bearing records are intentionally not published
here. Only its 12 designated calibration answers influenced answer-bearing
fixture design. Holdout, temporal, clean-control, and confirmation answers
remain sealed. Quodet contains repository-neutral synthetic scenarios, not
source application code, identifiers, secrets, or sealed answer text.

Related source and test files are replayed within the three-second quiet window:

```sh
uv run quodet prompt_eval_workspace/agent_replay --log
uv run python -m evals.agent_changes.replay 03_cross_file_units
```

Run the watcher in one terminal and the replay command in another. Use `all`
instead of a case ID to replay every scenario sequentially.
Add `--poll` when running across a container, sandbox, network mount, or another
environment where native filesystem events do not cross process boundaries.

For prompt development, run only the calibration split:

```sh
uv run python -m evals.agent_changes.live_eval calibration --log
```

Freeze the prompt and fixture revision before a one-shot evaluation. Then run
the pre-existing holdout and matched clean controls without iterating on their
answers:

```sh
uv run python -m evals.agent_changes.live_eval holdout --log
uv run python -m evals.agent_changes.live_eval clean-control --log
```

`temporal` and `confirmation` are explicit reserved/sealed boundaries with no
answer-bearing fixtures in this repository. Do not convert their source answers
into fixtures after seeing results. A full `all` run is useful for a frozen
comparison, but it is not permission to tune against non-calibration cases.

The live runner starts the watcher and replay in one process tree to avoid
cross-sandbox event delivery problems. Every raw artifact under the ignored
`eval-results/` directory retains the exact model and options, complete prompt
and schema with revisions and hashes, fixture revision and manifest hash, raw
provider response, parsed response, transcript, schema/provider/timeout state,
and per-case latency. Malformed output and timeouts remain failed samples; the
runner neither repairs them nor silently substitutes an unrecorded retry.

Raw findings require independent semantic adjudication. Generate a template,
judge every finding's explanation and demonstrated failure path, then score it:

```sh
uv run python -m evals.agent_changes.scoring \
  eval-results/RUN.raw.json --write-template /tmp/RUN.adjudication.json
# Edit every REPLACE_ME verdict and explain the rationale.
uv run python -m evals.agent_changes.scoring \
  eval-results/RUN.raw.json \
  --adjudication /tmp/RUN.adjudication.json \
  --output eval-results/RUN.scored.json
```

The scored report includes TP, FP, FN, schema-valid rate, split and family
breakdowns, and clean-control false-positive rate by family. Filename equality
is diagnostic only: a finding on the expected file with the wrong explanation
is an FP and leaves the expected defect as an FN.

The watcher sees changes made by every process, not only a coding agent. Add
generated output paths with `--exclude` to prevent noisy reviews or feedback
loops. Files are read by `llm` when a review starts, so the review reflects the
latest contents after the quiet period rather than every intermediate write.

The frozen fixture under `evals/recommended_fixes` records a known cache defect,
a complete expected finding, and the code-element, failure-path, and validation
characteristics its recommendation must contain. Unit tests check this fixture
deterministically without contacting a provider. Live-model runs remain an
explicit, separate evaluation through `evals.agent_changes.live_eval`, which
records exact run provenance and raw outcomes. Normal unit tests are
deterministic and never contact a provider.

## Privacy

Before upload, quodet redacts private-key blocks, common provider token formats,
authorization headers, credentials in URLs, sensitive assignments such as
`API_KEY=...` and `password: ...`, and generic high-entropy token-like values.
Temporary sanitized files use owner-only permissions and are deleted after the
request. Attachment filenames are generic, and both relative-path metadata and
custom prompt text are sanitized. Binary files are not uploaded because they
cannot be safely screened.

Secret detection is defense in depth, not a proof that arbitrary sensitive data
cannot pass. Do not watch a sensitive directory. Exclude secrets and private
data explicitly, for example:

```sh
uv run quodet . --exclude '.env*' --exclude '*.pem' --exclude 'private/**'
```
