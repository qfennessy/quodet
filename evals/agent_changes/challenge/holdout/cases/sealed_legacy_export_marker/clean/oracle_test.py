from export_record import load_record
from export_service import should_export


legacy = load_record({
    "record_id": "r1",
    "export_job_id": "job-7",
    "export_job_state": "completed",
})
assert should_export(legacy) is False
