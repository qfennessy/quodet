from export_record import ExportRecord


def should_export(record: ExportRecord) -> bool:
    return record.exported_at is None
