"""Offline tests for the fetch profiling helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile_fetch.py"
_SPEC = importlib.util.spec_from_file_location("_pycubrid_test_profile_fetch", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
profile_fetch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profile_fetch)

_SQL = "SELECT id, val FROM scratch WHERE id > ?"


class FakeCursor:
    """Replays a bounded sequence of rows or batches for every ``execute()``."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows
        self._pending: list[Any] = []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchone_calls = 0
        self.fetchall_calls = 0
        self.fetchmany_sizes: list[int] = []

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self.executed.append((sql, params))
        self._pending = list(self._rows)

    def fetchone(self) -> Any:
        self.fetchone_calls += 1
        return self._pending.pop(0) if self._pending else None

    def fetchall(self) -> list[Any]:
        self.fetchall_calls += 1
        rows, self._pending = self._pending, []
        return rows

    def fetchmany(self, size: int) -> list[Any]:
        self.fetchmany_sizes.append(size)
        batch, self._pending = self._pending[:size], self._pending[size:]
        return batch


def test_run_fetchone_stops_at_none_once_per_iteration() -> None:
    cursor = FakeCursor([(1, "a"), (2, "b"), (3, "c")])

    profile_fetch.run_fetchone(cursor, 2, sql=_SQL)

    assert cursor.executed == [(_SQL, (0,)), (_SQL, (0,))]
    # Three rows plus the terminating ``None`` for each of the two iterations.
    assert cursor.fetchone_calls == 2 * (3 + 1)
    assert cursor.fetchall_calls == 0
    assert cursor.fetchmany_sizes == []


def test_run_fetchone_handles_empty_result() -> None:
    cursor = FakeCursor([])

    profile_fetch.run_fetchone(cursor, 2, sql=_SQL)

    assert len(cursor.executed) == 2
    assert cursor.fetchone_calls == 2


def test_run_fetchall_calls_fetchall_exactly_once_per_iteration() -> None:
    cursor = FakeCursor([(1, "a"), (2, "b")])

    profile_fetch.run_fetchall(cursor, 2, sql=_SQL)

    assert cursor.executed == [(_SQL, (0,)), (_SQL, (0,))]
    assert cursor.fetchall_calls == 2
    assert cursor.fetchone_calls == 0
    assert cursor.fetchmany_sizes == []


def test_run_fetchmany_stops_at_empty_batch_and_passes_batch_size() -> None:
    cursor = FakeCursor([(1, "a"), (2, "b"), (3, "c")])

    profile_fetch.run_fetchmany(cursor, 2, 2, sql=_SQL)

    assert cursor.executed == [(_SQL, (0,)), (_SQL, (0,))]
    # Batches of 2 and 1, then the empty batch that ends each iteration.
    assert cursor.fetchmany_sizes == [2, 2, 2] * 2
    assert cursor.fetchone_calls == 0
    assert cursor.fetchall_calls == 0


def test_run_fetchmany_issues_a_single_fetch_when_result_is_empty() -> None:
    cursor = FakeCursor([])

    profile_fetch.run_fetchmany(cursor, 2, 5, sql=_SQL)

    assert len(cursor.executed) == 2
    assert cursor.fetchmany_sizes == [5, 5]


def test_helpers_issue_no_statements_for_zero_iterations() -> None:
    cursor = FakeCursor([(1, "a")])

    profile_fetch.run_fetchone(cursor, 0, sql=_SQL)
    profile_fetch.run_fetchall(cursor, 0, sql=_SQL)
    profile_fetch.run_fetchmany(cursor, 0, 3, sql=_SQL)

    assert cursor.executed == []
    assert cursor.fetchone_calls == 0
    assert cursor.fetchall_calls == 0
    assert cursor.fetchmany_sizes == []
