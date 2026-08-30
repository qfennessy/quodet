POLICY_JOB_INPUTS = ("policies/", "scripts/run_policy.py")


def command() -> list[str]:
    return ["python", "scripts/run_policy.py", "policies/"]
