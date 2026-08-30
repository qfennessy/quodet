from pathlib import PurePosixPath


LINT_DEPENDENCIES = {
    ".github/actionlint.yaml",
    "scripts/run-actionlint.sh",
}


def should_run_workflow_lint(changed_paths: list[str]) -> bool:
    return any(
        path in LINT_DEPENDENCIES
        or PurePosixPath(path).match(".github/workflows/*.yml")
        for path in changed_paths
    )
