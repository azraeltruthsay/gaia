"""Tests for the tabular workspace store (GAIA_Project-19g Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaia_common.utils.workspace import (
    MAX_CELLS_PER_SHEET,
    MAX_SHEETS_PER_WORKBOOK,
    Workbook,
    append_row,
    create_sheet,
    create_workbook,
    delete_sheet,
    delete_workbook,
    list_sheets,
    list_workbooks,
    load_workbook,
    read_rows,
    save_workbook,
    sheet_info,
    write_cell,
)


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "workspaces"


# ── Validation ──────────────────────────────────────────────────────


class TestNameValidation:
    @pytest.mark.parametrize("bad", [
        "", "has space", "has/slash", "has.dot", "../escape", "name\n",
    ])
    def test_invalid_names_rejected(self, bad, store):
        with pytest.raises(ValueError):
            create_workbook(bad, store_dir=store)

    @pytest.mark.parametrize("good", [
        "training_runs", "ABC123", "with-dashes", "u_n_d_e_r_s",
    ])
    def test_valid_names_accepted(self, good, store):
        wb = create_workbook(good, store_dir=store)
        assert wb.name == good

    def test_invalid_sheet_name_rejected(self, store):
        with pytest.raises(ValueError):
            create_workbook("ok", sheet_name="bad name", store_dir=store)


# ── Create / load / list / delete ───────────────────────────────────


class TestLifecycle:
    def test_create_persists(self, store):
        wb = create_workbook("foo", store_dir=store)
        assert (store / "foo.json").exists()
        data = json.loads((store / "foo.json").read_text())
        assert data["name"] == "foo"
        assert "main" in data["sheets"]

    def test_create_with_custom_sheet(self, store):
        wb = create_workbook("foo", sheet_name="metrics", store_dir=store)
        assert "metrics" in wb.sheets
        assert "main" not in wb.sheets

    def test_create_fails_if_exists(self, store):
        create_workbook("foo", store_dir=store)
        with pytest.raises(FileExistsError):
            create_workbook("foo", store_dir=store)

    def test_create_overwrite_flag(self, store):
        create_workbook("foo", store_dir=store)
        # Should not raise
        wb = create_workbook("foo", store_dir=store, overwrite=True)
        assert wb.sheets["main"]["rows"] == []

    def test_load_missing_returns_none(self, store):
        assert load_workbook("nope", store_dir=store) is None

    def test_round_trip(self, store):
        wb = create_workbook("foo", store_dir=store)
        wb.sheets["main"]["rows"].append({"a": 1, "b": "two"})
        wb.sheets["main"]["columns"] = ["a", "b"]
        assert save_workbook(wb, store_dir=store) is True
        wb2 = load_workbook("foo", store_dir=store)
        assert wb2.sheets["main"]["rows"][0] == {"a": 1, "b": "two"}

    def test_list_workbooks(self, store):
        assert list_workbooks(store_dir=store) == []
        create_workbook("alpha", store_dir=store)
        create_workbook("beta", store_dir=store)
        names = list_workbooks(store_dir=store)
        assert names == ["alpha", "beta"]  # sorted

    def test_list_filters_malformed_names(self, store):
        store.mkdir()
        # Drop a file with a "bad" name into the store dir; list should ignore it
        (store / "valid.json").write_text("{}")
        (store / ".hidden.json").write_text("{}")
        (store / "with space.json").write_text("{}")
        names = list_workbooks(store_dir=store)
        # ".hidden" and "with space" should both be excluded by the regex
        assert "valid" in names
        assert ".hidden" not in names
        assert "with space" not in names

    def test_delete(self, store):
        create_workbook("foo", store_dir=store)
        assert delete_workbook("foo", store_dir=store) is True
        assert not (store / "foo.json").exists()

    def test_delete_missing_returns_false(self, store):
        assert delete_workbook("foo", store_dir=store) is False


# ── Row operations ──────────────────────────────────────────────────


class TestAppendRow:
    def test_returns_index(self, store):
        create_workbook("foo", store_dir=store)
        assert append_row("foo", "main", {"a": 1}, store_dir=store) == 0
        assert append_row("foo", "main", {"a": 2}, store_dir=store) == 1
        assert append_row("foo", "main", {"a": 3}, store_dir=store) == 2

    def test_persists_to_disk(self, store):
        create_workbook("foo", store_dir=store)
        append_row("foo", "main", {"step": 0, "loss": 2.5}, store_dir=store)
        wb = load_workbook("foo", store_dir=store)
        assert wb.sheets["main"]["rows"] == [{"step": 0, "loss": 2.5}]

    def test_tracks_columns(self, store):
        create_workbook("foo", store_dir=store)
        append_row("foo", "main", {"a": 1, "b": 2}, store_dir=store)
        append_row("foo", "main", {"a": 3, "c": 4}, store_dir=store)
        wb = load_workbook("foo", store_dir=store)
        # Columns accumulate in insertion order
        assert wb.sheets["main"]["columns"] == ["a", "b", "c"]

    def test_missing_workbook_raises(self, store):
        with pytest.raises(FileNotFoundError):
            append_row("nope", "main", {"a": 1}, store_dir=store)

    def test_missing_sheet_raises(self, store):
        create_workbook("foo", store_dir=store)
        with pytest.raises(KeyError):
            append_row("foo", "missing", {"a": 1}, store_dir=store)

    def test_non_dict_row_rejected(self, store):
        create_workbook("foo", store_dir=store)
        with pytest.raises(TypeError):
            append_row("foo", "main", [1, 2, 3], store_dir=store)


class TestWriteCell:
    def test_updates_existing(self, store):
        create_workbook("foo", store_dir=store)
        append_row("foo", "main", {"a": 1}, store_dir=store)
        assert write_cell("foo", "main", 0, "a", 99, store_dir=store) is True
        wb = load_workbook("foo", store_dir=store)
        assert wb.sheets["main"]["rows"][0]["a"] == 99

    def test_adds_new_column(self, store):
        create_workbook("foo", store_dir=store)
        append_row("foo", "main", {"a": 1}, store_dir=store)
        write_cell("foo", "main", 0, "b", "new", store_dir=store)
        wb = load_workbook("foo", store_dir=store)
        assert wb.sheets["main"]["rows"][0]["b"] == "new"
        assert "b" in wb.sheets["main"]["columns"]

    def test_out_of_range_returns_false(self, store):
        create_workbook("foo", store_dir=store)
        append_row("foo", "main", {"a": 1}, store_dir=store)
        assert write_cell("foo", "main", 5, "a", 99, store_dir=store) is False
        assert write_cell("foo", "main", -1, "a", 99, store_dir=store) is False

    def test_missing_workbook_returns_false(self, store):
        assert write_cell("nope", "main", 0, "a", 1, store_dir=store) is False


# ── Read with filter + sort ─────────────────────────────────────────


@pytest.fixture
def populated(store):
    """Workbook with a 'main' sheet of 5 rows for filter/sort tests."""
    create_workbook("wb", store_dir=store)
    rows = [
        {"name": "alice", "score": 92, "tier": "A"},
        {"name": "bob", "score": 78, "tier": "B"},
        {"name": "carol", "score": 85, "tier": "A"},
        {"name": "dave", "score": 60, "tier": "C"},
        {"name": "eve", "score": 91, "tier": "A"},
    ]
    for r in rows:
        append_row("wb", "main", r, store_dir=store)
    return store


class TestReadRows:
    def test_no_filter_returns_all(self, populated):
        rows = read_rows("wb", store_dir=populated)
        assert len(rows) == 5

    def test_dict_filter_exact_match(self, populated):
        rows = read_rows("wb", filter_={"tier": "A"}, store_dir=populated)
        names = {r["name"] for r in rows}
        assert names == {"alice", "carol", "eve"}

    def test_callable_filter(self, populated):
        rows = read_rows(
            "wb",
            filter_=lambda r: r["score"] > 80,
            store_dir=populated,
        )
        names = {r["name"] for r in rows}
        assert names == {"alice", "carol", "eve"}

    def test_sort_by_column_ascending(self, populated):
        rows = read_rows("wb", sort_by="score", store_dir=populated)
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores)

    def test_sort_by_column_descending(self, populated):
        rows = read_rows(
            "wb", sort_by="score", descending=True, store_dir=populated,
        )
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_filter_plus_sort(self, populated):
        rows = read_rows(
            "wb", filter_={"tier": "A"}, sort_by="score",
            store_dir=populated,
        )
        names = [r["name"] for r in rows]
        assert names == ["carol", "eve", "alice"]  # 85, 91, 92

    def test_limit(self, populated):
        rows = read_rows("wb", limit=2, store_dir=populated)
        assert len(rows) == 2

    def test_rows_missing_sort_column_go_last(self, populated):
        # Append a row missing the 'score' column
        append_row("wb", "main", {"name": "frank", "tier": "D"}, store_dir=populated)
        rows = read_rows("wb", sort_by="score", store_dir=populated)
        # frank should be LAST (missing values sort after present ones)
        assert rows[-1]["name"] == "frank"

    def test_missing_workbook_empty_list(self, store):
        assert read_rows("nope", store_dir=store) == []


# ── Sheet operations ────────────────────────────────────────────────


class TestSheetOperations:
    def test_create_sheet(self, store):
        create_workbook("wb", store_dir=store)
        assert create_sheet("wb", "metrics", store_dir=store) is True
        assert list_sheets("wb", store_dir=store) == ["main", "metrics"]

    def test_create_existing_sheet_returns_false(self, store):
        create_workbook("wb", store_dir=store)
        assert create_sheet("wb", "main", store_dir=store) is False

    def test_create_sheet_invalid_name(self, store):
        create_workbook("wb", store_dir=store)
        with pytest.raises(ValueError):
            create_sheet("wb", "bad name", store_dir=store)

    def test_delete_sheet(self, store):
        create_workbook("wb", store_dir=store)
        create_sheet("wb", "extra", store_dir=store)
        assert delete_sheet("wb", "extra", store_dir=store) is True
        assert "extra" not in list_sheets("wb", store_dir=store)

    def test_cannot_delete_last_sheet(self, store):
        create_workbook("wb", store_dir=store)
        assert delete_sheet("wb", "main", store_dir=store) is False

    def test_delete_missing_sheet_returns_false(self, store):
        create_workbook("wb", store_dir=store)
        assert delete_sheet("wb", "nope", store_dir=store) is False

    def test_list_sheets_missing_workbook(self, store):
        assert list_sheets("nope", store_dir=store) == []

    def test_max_sheets_enforced(self, store):
        create_workbook("wb", store_dir=store)
        # Already has "main"; add up to MAX_SHEETS_PER_WORKBOOK total
        for i in range(MAX_SHEETS_PER_WORKBOOK - 1):
            assert create_sheet("wb", f"sheet_{i}", store_dir=store) is True
        with pytest.raises(ValueError):
            create_sheet("wb", "one_too_many", store_dir=store)


class TestSheetInfo:
    def test_basic_stats(self, populated):
        info = sheet_info("wb", store_dir=populated)
        assert info["name"] == "main"
        assert info["rows"] == 5
        assert set(info["columns"]) == {"name", "score", "tier"}
        # 5 rows × 3 cols = 15 cells
        assert info["cells"] == 15

    def test_missing_workbook(self, store):
        assert sheet_info("nope", store_dir=store) == {}

    def test_missing_sheet(self, store):
        create_workbook("wb", store_dir=store)
        assert sheet_info("wb", sheet="missing", store_dir=store) == {}


# ── Atomic write smoke ──────────────────────────────────────────────


class TestAtomicWrite:
    def test_no_tmp_file_after_write(self, store):
        create_workbook("wb", store_dir=store)
        append_row("wb", "main", {"a": 1}, store_dir=store)
        # No leftover .tmp files
        tmps = list(store.glob("*.tmp"))
        assert tmps == []

    def test_concurrent_writes_dont_corrupt(self, store):
        """Best-effort: a few quick sequential appends from the same
        thread shouldn't ever leave the file truncated. This isn't a
        real concurrency test — actual concurrent access is a
        last-write-wins policy per the bd issue's Phase 1 spec."""
        create_workbook("wb", store_dir=store)
        for i in range(20):
            append_row("wb", "main", {"i": i}, store_dir=store)
        wb = load_workbook("wb", store_dir=store)
        assert len(wb.sheets["main"]["rows"]) == 20
