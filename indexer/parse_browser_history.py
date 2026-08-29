"""Parse browser history (Safari, Chrome, Arc, Opera GX) into daily browsing chunks."""
from __future__ import annotations

import datetime
import glob
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlsplit

from chunker import Chunk, _new_id

# Safari stores visit_time as seconds since the Core Data epoch (2001-01-01 UTC).
CORE_DATA_EPOCH_OFFSET = 978307200
# Chromium stores last_visit_time as microseconds since 1601-01-01 UTC.
WEBKIT_EPOCH_OFFSET = 11644473600

SAFARI_HISTORY_PATH = Path.home() / "Library" / "Safari" / "History.db"

CHROMIUM_HISTORY_GLOBS: tuple[tuple[str, str], ...] = (
    ("Chrome", str(Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "*" / "History")),
    ("Arc", str(Path.home() / "Library" / "Application Support" / "Arc" / "User Data" / "*" / "History")),
    ("Opera GX", str(Path.home() / "Library" / "Application Support" / "com.operasoftware.OperaGX" / "*" / "History")),
)

VISITS_PER_CHUNK = 25
MAX_URL_LEN = 500

_SKIP_SCHEMES = ("data:", "file:", "javascript:", "about:", "chrome:", "chrome-extension:", "arc:", "opera:")
_NOISE_HOSTS = {
    "accounts.google.com",
    "accounts.youtube.com",
    "appleid.apple.com",
    "login.microsoftonline.com",
}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]"}


@dataclass
class BrowserVisit:
    row_id: int
    title: str
    url: str          # domain + path, query string stripped entirely
    domain: str
    visit_iso: str    # naive-UTC ISO
    browser: str


def core_data_time_to_iso(seconds: float | None) -> str:
    """Core Data epoch seconds (2001-01-01 UTC) → naive-UTC ISO string."""
    if seconds is None:
        return ""
    try:
        return (
            datetime.datetime.fromtimestamp(seconds + CORE_DATA_EPOCH_OFFSET, tz=datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat(timespec="seconds")
        )
    except (OverflowError, OSError, ValueError):
        return ""


def webkit_time_to_iso(microseconds: int | None) -> str:
    """WebKit epoch microseconds (1601-01-01 UTC) → naive-UTC ISO string."""
    if not microseconds:
        return ""
    try:
        unix_ts = microseconds / 1_000_000 - WEBKIT_EPOCH_OFFSET
        return (
            datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat(timespec="seconds")
        )
    except (OverflowError, OSError, ValueError):
        return ""


def clean_url(raw: str | None) -> tuple[str, str] | None:
    """Return (domain+path, domain) with the query/fragment stripped, or None to skip.

    Skips non-http(s) schemes, localhost, known auth endpoints, and absurdly
    long URLs (session tokens, redirect chains).
    """
    if not raw:
        return None
    lowered = raw.lower()
    if any(lowered.startswith(s) for s in _SKIP_SCHEMES):
        return None
    if len(raw) > MAX_URL_LEN:
        return None
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower()
    if not host or host in _LOCAL_HOSTS or host in _NOISE_HOSTS:
        return None
    path = parts.path.rstrip("/")
    return (f"{host}{path}", host)


def _copy_sqlite_to_temp(src: Path) -> tuple[str, str]:
    """Copy a sqlite DB (+ -wal/-shm sidecars) to a fresh temp dir; never touch the original.

    Returns (db_path, tmp_dir); caller must shutil.rmtree(tmp_dir).
    """
    tmp_dir = tempfile.mkdtemp(prefix="semse_history_")
    try:
        dest = Path(tmp_dir) / src.name
        shutil.copy2(src, dest)
        for suffix in ("-wal", "-shm"):
            sidecar = src.parent / (src.name + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, Path(tmp_dir) / (src.name + suffix))
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return str(dest), tmp_dir


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def _iter_safari(db_path: Path = SAFARI_HISTORY_PATH) -> Iterator[BrowserVisit]:
    if not db_path.exists():
        print(f"  browsing: Safari history not found at {db_path}, skipping")
        return
    path, tmp_dir = _copy_sqlite_to_temp(db_path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if not (_table_exists(conn, "history_items") and _table_exists(conn, "history_visits")):
            print("  browsing: Safari History.db missing expected tables, skipping")
            return
        cur = conn.execute(
            """
            SELECT v.id, v.visit_time, v.title, i.url
            FROM history_visits v
            JOIN history_items i ON v.history_item = i.id
            WHERE v.redirect_destination IS NULL
            ORDER BY v.visit_time ASC
            """
        )
        for row_id, visit_time, title, url in cur:
            cleaned = clean_url(url)
            if not cleaned:
                continue
            visit_iso = core_data_time_to_iso(visit_time)
            if not visit_iso:
                continue
            yield BrowserVisit(
                row_id=int(row_id),
                title=(title or "").strip() or cleaned[1],
                url=cleaned[0],
                domain=cleaned[1],
                visit_iso=visit_iso,
                browser="Safari",
            )
    finally:
        conn.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _iter_chromium(browser: str, db_path: Path) -> Iterator[BrowserVisit]:
    # Temp-copy handles the lock Chromium holds on History while running.
    path, tmp_dir = _copy_sqlite_to_temp(db_path)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if not _table_exists(conn, "urls"):
            print(f"  browsing: {browser} History at {db_path} missing urls table, skipping")
            return
        cur = conn.execute(
            """
            SELECT id, url, title, last_visit_time
            FROM urls
            WHERE last_visit_time > 0 AND hidden = 0
            ORDER BY last_visit_time ASC
            """
        )
        for row_id, url, title, last_visit_time in cur:
            cleaned = clean_url(url)
            if not cleaned:
                continue
            visit_iso = webkit_time_to_iso(last_visit_time)
            if not visit_iso:
                continue
            yield BrowserVisit(
                row_id=int(row_id),
                title=(title or "").strip() or cleaned[1],
                url=cleaned[0],
                domain=cleaned[1],
                visit_iso=visit_iso,
                browser=browser,
            )
    finally:
        conn.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def iter_browser_visits(since_iso: str | None = None) -> Iterator[BrowserVisit]:
    """Yield visits from every installed browser, deduped by (url, day).

    Missing browsers are skipped with a notice; source DBs are never written.
    `since_iso` (when set) drops visits at or before the cutoff — used by
    additive re-index runs.
    """
    seen: set[tuple[str, str]] = set()

    def _dedup(visits: Iterable[BrowserVisit]) -> Iterator[BrowserVisit]:
        for v in visits:
            if since_iso and v.visit_iso <= since_iso:
                continue
            key = (v.url, v.visit_iso[:10])
            if key in seen:
                continue
            seen.add(key)
            yield v

    yield from _dedup(_iter_safari())

    for browser, pattern in CHROMIUM_HISTORY_GLOBS:
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"  browsing: no {browser} profiles found, skipping")
            continue
        for p in paths:
            try:
                yield from _dedup(_iter_chromium(browser, Path(p)))
            except (sqlite3.Error, OSError) as e:
                print(f"  browsing: {browser} profile {p} unreadable ({e}), skipping")


def _format_visit_line(v: BrowserVisit) -> str:
    return f"{v.visit_iso[:10]} · {v.title} — {v.url} ({v.browser})"


def chunk_browser_visits(
    visits: Iterable[BrowserVisit],
    per_chunk: int = VISITS_PER_CHUNK,
) -> Iterator[Chunk]:
    """Group visits by day, then split each day into chunks of ~per_chunk.

    Snippet-style chunks: empty `messages`, empty `contact_names`, `subject`
    carries the day label.
    """
    by_day: dict[str, list[BrowserVisit]] = {}
    for v in visits:
        by_day.setdefault(v.visit_iso[:10], []).append(v)

    for day in sorted(by_day):
        group = sorted(by_day[day], key=lambda v: v.visit_iso)
        for start in range(0, len(group), per_chunk):
            slab = group[start : start + per_chunk]
            yield Chunk(
                chunk_id=_new_id(),
                source="browsing",
                contact_names=[],
                date_start=min(v.visit_iso for v in slab),
                date_end=max(v.visit_iso for v in slab),
                text=f"Web browsing on {day}:\n"
                + "\n".join(_format_visit_line(v) for v in slab),
                row_ids=[v.row_id for v in slab],
                messages=[],
                subject=f"Browsing · {day}",
            )


if __name__ == "__main__":
    per_browser: dict[str, int] = {}
    all_visits: list[BrowserVisit] = []
    for visit in iter_browser_visits():
        per_browser[visit.browser] = per_browser.get(visit.browser, 0) + 1
        all_visits.append(visit)
    chunks = list(chunk_browser_visits(all_visits))
    print(f"\n{len(all_visits)} visits after dedup → {len(chunks)} chunks")
    for name, count in sorted(per_browser.items()):
        print(f"  {name}: {count}")
    for c in chunks[-3:]:
        print(f"\n--- {c.subject} ({len(c.row_ids)} visits) ---")
        print("\n".join(c.text.splitlines()[:6]))
