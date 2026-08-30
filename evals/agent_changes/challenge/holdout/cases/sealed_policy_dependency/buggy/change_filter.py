def should_run_policy(changed_paths: list[str]) -> bool:
    return any(path.startswith("policies/") for path in changed_paths)
