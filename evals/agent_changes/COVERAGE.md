# Evaluation coverage map

This map was derived from an independently verified, non-public review corpus.
The source location and answer-bearing records are intentionally not published
in this repository. The source corpus reports 56 verified positives across
state/lifecycle (13), external API contract (9), privacy/authorization (9),
retry/concurrency (8), UI/cache (6), CI/tooling (6), persistence/atomicity (3),
and performance/cost (2), plus 16 matched clean controls.

Only the 12 primary calibration answers were read as answer-bearing design
material. Holdout, temporal, clean-control, and confirmation answers were not
read or copied. Their aggregate taxonomy and non-answer metadata establish
priorities and split boundaries only. Every fixture here is repository-neutral
synthetic code; none is source application code.

## Case mapping

| Quodet case | Family | Scope | Evidence depth | Split |
| --- | --- | --- | --- | --- |
| `01_obvious_runtime` | state/lifecycle | narrow | local execution path | holdout |
| `02_source_and_test_boundary` | UI/cache | narrow | language boundary plus weak test | holdout |
| `03_cross_file_units` | external API contract | cross-file | producer-consumer units | holdout |
| `04_tenant_cache_scope` | privacy/authorization | cross-file | tenant identity and cache key | holdout |
| `05_async_agent_patch` | retry/concurrency | cross-file | await interleaving | holdout |
| `06_exception_cleanup` | state/lifecycle; retry/concurrency | cross-file | exceptional resource lifecycle | holdout |
| `07_batch_deduplication` | persistence/atomicity; retry/concurrency | narrow | in-batch state update | holdout |
| `08_clean_related_change` | UI/cache; retry/concurrency | cross-file | related clean behavior | clean-control |
| `09_stale_undo_identity` | state/lifecycle | narrow | stale revision identity | calibration |
| `10_aggregate_cache_fingerprint` | UI/cache | cross-file | child state under stable aggregate | calibration |
| `11_partial_async_rollback` | persistence/atomicity; retry/concurrency | cross-file | sibling task rollback | calibration |
| `12_semantically_invalid_external_value` | external API contract | narrow | semantic domain validation | calibration |
| `13_swallowed_retry_signal` | retry/concurrency | narrow | exception taxonomy | calibration |
| `14_stale_lease_restart` | state/lifecycle | cross-file | persisted process marker | calibration |
| `15_unknown_timeout_spend` | performance/cost | cross-file | timeout accounting state | calibration |
| `16_tooling_dependency_filter` | CI/tooling | cross-file | job dependency graph | calibration |
| `17_partial_persistence_narrow` | persistence/atomicity | narrow | mutation ordering | calibration |
| `18_cost_retry_narrow` | performance/cost | narrow | ambiguous timeout/idempotency | calibration |
| `19_private_exception_logging` | privacy/authorization | narrow | exception payload to log sink | calibration |
| `20_tool_exit_status` | CI/tooling | narrow | process exit contract | calibration |
| `21_clean_identity_guard` | state/lifecycle | narrow | matched stale-identity control | clean-control |
| `22_clean_cache_fingerprint` | UI/cache | cross-file | matched aggregate-cache control | clean-control |
| `23_clean_async_rollback` | persistence/atomicity; retry/concurrency | cross-file | matched atomicity control | clean-control |
| `24_clean_external_validation` | external API contract | narrow | matched semantic-contract control | clean-control |
| `25_clean_tooling_dependencies` | CI/tooling | cross-file | matched tooling-dependency control | clean-control |
| `26_clean_settled_async_rollback` | persistence/atomicity; retry/concurrency | cross-file | reachable cancellation and settled sibling schedule | clean-control |

All eight corpus families now have positive coverage. Each has both narrow and
cross-file evidence, including multi-family cases where one execution path
crosses two failure modes. Six new matched controls
span one- and two-file cases and basic through sophisticated reasoning, in
addition to the original two-file control.

## Deliberate gaps and leakage boundary

- The original seven Quodet positives are a frozen holdout because they existed
  before this corpus-derived expansion. Do not tune prompts from their answers.
- No answer-bearing temporal fixture was constructed: temporal source answers
  remain sealed. The runner keeps a separate zero-sized temporal metric bucket.
- No confirmation fixture or answer is included. Confirmation remains a sealed,
  separately reported boundary for later one-shot work.
- Filenames are retained only as diagnostics. A finding becomes a true positive
  only after a reviewer adjudicates its explanation and concrete failure path
  against the expected behavior; the right filename with the wrong diagnosis
  is a false positive and leaves the expected defect as a false negative.
