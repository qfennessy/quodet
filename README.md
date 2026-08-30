# quodet

Quodet was created at [Sundai Hack 138](https://sundai.club). Sundai Club is a
community for building and launching AI prototypes every Sunday.

Quodet's goal is to integrate directly with coding-agent workflows and provide
extremely low-latency code-quality feedback while an agent is still working.
The independent watcher batches related file changes, reviews the current code,
and is designed to return high-confidence defects quickly enough for the coding
agent to verify and address them in the same development loop.

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
      "severity": "high",
      "confidence": 0.96,
      "title": "Short defect title",
      "explanation": "Why this is a concrete defect",
      "suggested_fix": "A focused correction"
    }
  ]
}
```

When no confident negative findings exist, Luna returns `{"findings": []}`.

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

The evaluation corpus under `evals/agent_changes` models coding-agent changes
from obvious single-file mistakes through subtle multi-file authorization,
concurrency, cleanup, unit-contract, and deduplication defects. Each scenario
declares its expected findings, and related source/test files are replayed
within the three-second quiet window:

```sh
uv run quodet prompt_eval_workspace/agent_replay --log
uv run python -m evals.agent_changes.replay 03_cross_file_units
```

Run the watcher in one terminal and the replay command in another. Use `all`
instead of a case ID to replay every scenario sequentially.
Add `--poll` when running across a container, sandbox, network mount, or another
environment where native filesystem events do not cross process boundaries.

For a scored live run, start the watcher and replay in one process tree. This
avoids cross-sandbox event delivery problems and checks the returned finding
files against the manifest, including the clean control:

```sh
uv run python -m evals.agent_changes.live_eval all --log
```

The watcher sees changes made by every process, not only a coding agent. Add
generated output paths with `--exclude` to prevent noisy reviews or feedback
loops. Files are read by `llm` when a review starts, so the review reflects the
latest contents after the quiet period rather than every intermediate write.

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
