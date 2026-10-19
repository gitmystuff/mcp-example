# parquet_store.py
"""
Audit log for MCP tool calls, stored as a Parquet file.

Every tool call the server handles is recorded as one row (request id, tool
name, status, and tool-specific fields), with a UTC timestamp added here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Anchor to the project folder so the log is always written in the same place,
# whatever directory the server was launched from.
PROJECT_DIR = Path(__file__).resolve().parent
PARQUET_DIR = PROJECT_DIR / "data_parquet"
PARQUET_FILE = PARQUET_DIR / "tool_audit.parquet"


def _ensure_dir() -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def append_audit_row(row: Dict[str, Any]) -> str:
    """
    Append a single tool call row to the audit Parquet file.

    Parquet files can't be appended to in place, so for this class demo we
    read the existing file, add the row, and rewrite it (simple and reliable
    for small logs). Rows from different tools have different fields; missing
    fields are stored as nulls.
    """
    _ensure_dir()

    row = dict(row)
    row["server_utc_ts"] = datetime.now(timezone.utc).isoformat()

    new_df = pd.DataFrame([row])

    if PARQUET_FILE.exists():
        old_df = pd.read_parquet(PARQUET_FILE)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    # Write to a temporary file first, then swap it in, so an interrupted
    # write can't leave a corrupted log behind.
    tmp_path = PARQUET_FILE.with_suffix(".parquet.tmp")
    out_df.to_parquet(tmp_path, index=False)
    tmp_path.replace(PARQUET_FILE)
    return str(PARQUET_FILE)


def export_audit_copy(filename: str = "audit_export.parquet") -> str:
    """
    Export a copy of the audit Parquet file for submission.

    Only a plain file name is accepted; the copy is always written inside
    data_parquet/.
    """
    _ensure_dir()

    name = Path(filename).name
    if not name or name != filename or name in (".", ".."):
        raise ValueError(
            f"filename must be a plain file name such as 'audit_export.parquet', not a path: {filename!r}"
        )
    if not name.endswith(".parquet"):
        name += ".parquet"
    if name == PARQUET_FILE.name:
        raise ValueError(f"filename can't be '{PARQUET_FILE.name}'; that is the live audit log.")

    out_path = PARQUET_DIR / name

    if PARQUET_FILE.exists():
        df = pd.read_parquet(PARQUET_FILE)
    else:
        df = pd.DataFrame()
    df.to_parquet(out_path, index=False)

    return str(out_path)
