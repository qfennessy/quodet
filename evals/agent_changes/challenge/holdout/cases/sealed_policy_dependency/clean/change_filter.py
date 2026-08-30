def should_run_policy(changed_paths: list[str]) -> bool:
    return any(
        path.startswith("policies/") or path == "scripts/run_policy.py"
        for path in changed_paths
    )
