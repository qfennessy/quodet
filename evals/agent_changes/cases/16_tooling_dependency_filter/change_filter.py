from pathlib import PurePosixPath


def should_run_workflow_lint(changed_paths: list[str]) -> bool:
    return any(
        PurePosixPath(path).match(".github/workflows/*.yml")
        for path in changed_paths
    )
