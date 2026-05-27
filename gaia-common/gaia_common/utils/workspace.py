"""Tabular workspace store (GAIA_Project-19g Phase 1).

GAIA's existing state stores — markdown, JSON, SQLite, vector embeddings —
don't fit 2D tabular data she manipulates programmatically (training
metrics across steps, decision matrices, session summaries, curriculum
inspection).

This module is the storage + manipulation layer. Workbooks live as JSON
files at /shared/workspaces/<name>.json. Each workbook has one or more
sheets; each sheet is a list of row dicts (homogeneous within a sheet
but not across sheets).

Row-of-dicts model is chosen over cell-address (A1, B2, ...) for Phase 1
because:
  - matches the sortable/filterable HTML table the UI will render
  - simpler API surface for the common case ("append a row of metrics")
  - cell addressing can be layered on top in Phase 2 when formulas land

API:

  create_workbook(name, sheet_name="main") -> Workbook
      Create + persist an empty workbook with one (empty) sheet.

  load_workbook(name) -> Workbook | None
      Load by name; None if absent.

  list_workbooks() -> list[str]
      Names of all workbooks in the store.

  read_rows(name, sheet="main", *, filter_=None, sort_by=None,
            descending=False) -> list[dict]
      Read rows, optionally filtered + sorted. filter_ is a dict of
      column→expected_value (exact match) OR a callable(row)->bool.

  append_row(name, sheet, row_dict) -> int
      Append a row; return its 0-based index.

  write_cell(name, sheet, row_index, column, value) -> bool
      Update a single cell; return True on success.

  create_sheet(name, sheet_name) -> bool
      Add a new sheet to an existing workbook.

  delete_sheet(name, sheet_name) -> bool
      Remove a sheet (refuses to delete the last sheet).

Atomic writes via tmp + rename. File-locking is intentionally NOT
implemented in Phase 1 — last-write-wins is the documented policy
(see bd issue's open design questions).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

logger = logging.getLogger("GAIA.Workspace")


DEFAULT_STORE_DIR = Path(os.environ.get(
    "GAIA_WORKSPACES_DIR", "/shared/workspaces",
))

# Bounded: a workbook over this many cells in a single sheet probably
# means the wrong storage layer. The 100K cap matches the bd issue's
# Phase 1 size limit.
MAX_CELLS_PER_SHEET = 100_000
MAX_SHEETS_PER_WORKBOOK = 50

_WORKBOOK_NAME_RE = re.compile(r"\A[a-zA-Z0-9_-]+\Z")
_lock = threading.Lock()


# ── Dataclass ───────────────────────────────────────────────────────


@dataclass
class Workbook:
    """In-memory representation of one workspace file."""
    name: str
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    sheets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sheets": self.sheets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Workbook":
        return cls(
            name=data.get("name", "unnamed"),
            version=int(data.get("version", 1)),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            sheets=data.get("sheets", {}),
        )


# ── Helpers ─────────────────────────────────────────────────────────


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_name(name: str, *, what: str = "workbook") -> None:
    if not name or not _WORKBOOK_NAME_RE.match(name):
        raise ValueError(
            f"Invalid {what} name {name!r}: must match [a-zA-Z0-9_-]+ "
            "(no spaces, slashes, or dots)"
        )


def _workbook_path(name: str, *, store_dir: Optional[Path] = None) -> Path:
    base = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
    return base / f"{name}.json"


def _atomic_write(path: Path, payload: dict) -> bool:
    """tmp + rename write. Returns True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(path)
        return True
    except OSError as e:
        logger.warning("Workbook write failed for %s: %s", path, e)
        return False


def _count_cells(sheet: dict) -> int:
    """Sheet cell count = rows × distinct columns observed."""
    rows = sheet.get("rows") or []
    if not rows:
        return 0
    cols = set()
    for r in rows:
        if isinstance(r, dict):
            cols.update(r.keys())
    return len(rows) * max(1, len(cols))


# ── CRUD ────────────────────────────────────────────────────────────


def create_workbook(
    name: str,
    *,
    sheet_name: str = "main",
    store_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Workbook:
    """Create and persist an empty workbook.

    Raises FileExistsError if the workbook already exists and
    overwrite=False. ValueError on invalid name.
    """
    _validate_name(name)
    _validate_name(sheet_name, what="sheet")
    path = _workbook_path(name, store_dir=store_dir)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Workbook already exists: {name}")
    now = _utcnow()
    wb = Workbook(
        name=name, version=1, created_at=now, updated_at=now,
        sheets={sheet_name: {"rows": [], "columns": []}},
    )
    with _lock:
        if not _atomic_write(path, wb.to_dict()):
            raise OSError(f"Failed to persist workbook {name}")
    return wb


def load_workbook(
    name: str, *, store_dir: Optional[Path] = None,
) -> Optional[Workbook]:
    _validate_name(name)
    path = _workbook_path(name, store_dir=store_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Workbook load failed for %s: %s", name, e)
        return None
    return Workbook.from_dict(data)


def save_workbook(
    wb: Workbook, *, store_dir: Optional[Path] = None,
) -> bool:
    _validate_name(wb.name)
    wb.updated_at = _utcnow()
    path = _workbook_path(wb.name, store_dir=store_dir)
    with _lock:
        return _atomic_write(path, wb.to_dict())


def list_workbooks(*, store_dir: Optional[Path] = None) -> list[str]:
    base = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
    if not base.exists():
        return []
    return sorted(
        p.stem for p in base.glob("*.json")
        if _WORKBOOK_NAME_RE.match(p.stem)
    )


def delete_workbook(
    name: str, *, store_dir: Optional[Path] = None,
) -> bool:
    _validate_name(name)
    path = _workbook_path(name, store_dir=store_dir)
    if not path.exists():
        return False
    try:
        with _lock:
            path.unlink()
        return True
    except OSError as e:
        logger.warning("Workbook delete failed for %s: %s", name, e)
        return False


# ── Row operations ──────────────────────────────────────────────────


def _ensure_sheet(wb: Workbook, sheet_name: str) -> dict:
    if sheet_name not in wb.sheets:
        raise KeyError(
            f"Sheet {sheet_name!r} not in workbook {wb.name!r}"
            f" (sheets: {list(wb.sheets.keys())})"
        )
    return wb.sheets[sheet_name]


def read_rows(
    name: str,
    sheet: str = "main",
    *,
    filter_: Optional[Union[dict, Callable[[dict], bool]]] = None,
    sort_by: Optional[str] = None,
    descending: bool = False,
    limit: Optional[int] = None,
    store_dir: Optional[Path] = None,
) -> list[dict]:
    """Read rows with optional filter + sort.

    filter_ can be either a dict (exact column→value match) or a
    callable (row→bool). sort_by is a column name; rows without that
    column are sorted to the end.
    """
    wb = load_workbook(name, store_dir=store_dir)
    if wb is None:
        return []
    sheet_data = _ensure_sheet(wb, sheet)
    rows = list(sheet_data.get("rows") or [])

    if filter_ is not None:
        if callable(filter_):
            rows = [r for r in rows if filter_(r)]
        elif isinstance(filter_, dict):
            rows = [
                r for r in rows
                if all(r.get(k) == v for k, v in filter_.items())
            ]

    if sort_by:
        # Rows missing the sort column go LAST (stable sort, "missing"
        # treated as max so they don't shadow real data).
        sentinel = object()

        def _key(r):
            v = r.get(sort_by, sentinel)
            return (v is sentinel, v if v is not sentinel else "")

        rows.sort(key=_key, reverse=descending)

    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def append_row(
    name: str,
    sheet: str,
    row: dict,
    *,
    store_dir: Optional[Path] = None,
) -> int:
    """Append a row dict; return its 0-based index."""
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")
    wb = load_workbook(name, store_dir=store_dir)
    if wb is None:
        raise FileNotFoundError(f"Workbook not found: {name}")
    sheet_data = _ensure_sheet(wb, sheet)
    rows = sheet_data.get("rows") or []
    if (len(rows) + 1) * max(1, len(row)) > MAX_CELLS_PER_SHEET:
        raise ValueError(
            f"Sheet {sheet!r} would exceed MAX_CELLS_PER_SHEET="
            f"{MAX_CELLS_PER_SHEET}"
        )
    rows.append(dict(row))
    sheet_data["rows"] = rows
    # Update columns set
    cols = list(sheet_data.get("columns") or [])
    for k in row.keys():
        if k not in cols:
            cols.append(k)
    sheet_data["columns"] = cols
    if not save_workbook(wb, store_dir=store_dir):
        raise OSError(f"Failed to persist append to {name}/{sheet}")
    return len(rows) - 1


def write_cell(
    name: str,
    sheet: str,
    row_index: int,
    column: str,
    value: Any,
    *,
    store_dir: Optional[Path] = None,
) -> bool:
    """Update a single cell. Returns True on success, False if row
    index is out of range. Adds the column if it didn't exist."""
    wb = load_workbook(name, store_dir=store_dir)
    if wb is None:
        return False
    sheet_data = _ensure_sheet(wb, sheet)
    rows = sheet_data.get("rows") or []
    if row_index < 0 or row_index >= len(rows):
        return False
    rows[row_index][column] = value
    sheet_data["rows"] = rows
    cols = list(sheet_data.get("columns") or [])
    if column not in cols:
        cols.append(column)
        sheet_data["columns"] = cols
    return save_workbook(wb, store_dir=store_dir)


# ── Sheet operations ────────────────────────────────────────────────


def create_sheet(
    name: str,
    sheet_name: str,
    *,
    store_dir: Optional[Path] = None,
) -> bool:
    """Add a new (empty) sheet to a workbook. Refuses if the sheet
    already exists or if we'd exceed MAX_SHEETS_PER_WORKBOOK."""
    _validate_name(sheet_name, what="sheet")
    wb = load_workbook(name, store_dir=store_dir)
    if wb is None:
        return False
    if sheet_name in wb.sheets:
        return False
    if len(wb.sheets) >= MAX_SHEETS_PER_WORKBOOK:
        raise ValueError(
            f"Workbook {name!r} already has "
            f"{MAX_SHEETS_PER_WORKBOOK} sheets (max)"
        )
    wb.sheets[sheet_name] = {"rows": [], "columns": []}
    return save_workbook(wb, store_dir=store_dir)


def delete_sheet(
    name: str,
    sheet_name: str,
    *,
    store_dir: Optional[Path] = None,
) -> bool:
    """Remove a sheet. Refuses to delete the last remaining sheet
    (workbooks must have at least one)."""
    wb = load_workbook(name, store_dir=store_dir)
    if wb is None or sheet_name not in wb.sheets:
        return False
    if len(wb.sheets) <= 1:
        return False
    del wb.sheets[sheet_name]
    return save_workbook(wb, store_dir=store_dir)


def list_sheets(
    name: str, *, store_dir: Optional[Path] = None,
) -> list[str]:
    wb = load_workbook(name, store_dir=store_dir)
    if wb is None:
        return []
    return list(wb.sheets.keys())


def sheet_info(
    name: str,
    sheet: str = "main",
    *,
    store_dir: Optional[Path] = None,
) -> dict:
    """Return summary stats for one sheet: row count, columns, cell count."""
    wb = load_workbook(name, store_dir=store_dir)
    if wb is None or sheet not in wb.sheets:
        return {}
    sheet_data = wb.sheets[sheet]
    rows = sheet_data.get("rows") or []
    cols = sheet_data.get("columns") or []
    return {
        "name": sheet,
        "rows": len(rows),
        "columns": list(cols),
        "cells": _count_cells(sheet_data),
    }
