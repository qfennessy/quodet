from dataclasses import dataclass


@dataclass
class ExportRecord:
    record_id: str
    exported_at: str | None
    export_job_id: str | None
    export_job_state: str | None


def load_record(payload: dict[str, str]) -> ExportRecord:
    return ExportRecord(
        record_id=payload["record_id"],
        exported_at=payload.get("exported_at"),
        export_job_id=payload.get("export_job_id"),
        export_job_state=payload.get("export_job_state"),
    )
