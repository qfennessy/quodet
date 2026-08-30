from change_filter import should_run_policy
from policy_job import POLICY_JOB_INPUTS


assert "scripts/run_policy.py" in POLICY_JOB_INPUTS
assert should_run_policy(["scripts/run_policy.py"]) is True
