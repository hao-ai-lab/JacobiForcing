#!/usr/bin/env python3
"""
Merge JSON files (each containing a top-level list[dict]) into one JSONL file, RAM-safe.

- Streams with ijson (no giant lists in RAM)
- tqdm progress bars
- Skips corrupted/partial JSON files (e.g., IncompleteJSONError) and continues

Install:
  pip install ijson tqdm
"""

from __future__ import annotations
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterator, Dict, Any, Optional

from tqdm import tqdm

import ijson
from ijson import common as ijson_common


def iter_json_array_items(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield dict items from a top-level JSON array without loading the whole file."""
    with path.open("rb") as f:
        for obj in ijson.items(f, "item"):
            if not isinstance(obj, dict):
                raise TypeError(f"{path} contains a non-dict item in its list.")
            yield obj


def key_to_text(val: Any) -> str:
    if val is None:
        return "N:null"
    if isinstance(val, bool):
        return f"B:{int(val)}"
    if isinstance(val, int):
        return f"I:{val}"
    if isinstance(val, float):
        return f"F:{repr(val)}"
    if isinstance(val, str):
        return "S:" + val
    return "J:" + json.dumps(val, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Deduper:
    def __init__(self, mode: str, db_path: Optional[Path] = None):
        mode = (mode or "sqlite").lower()
        if mode not in {"memory", "sqlite"}:
            raise ValueError("--dedup-store must be 'memory' or 'sqlite'")
        self.mode = mode
        self.seen = set() if mode == "memory" else None
        self.conn = None
        self.cur = None
        self._batch = 0
        if mode == "sqlite":
            if db_path is None:
                raise ValueError("db_path must be provided for sqlite dedup store")
            self.conn = sqlite3.connect(str(db_path))
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("CREATE TABLE IF NOT EXISTS seen (k TEXT PRIMARY KEY)")
            self.cur = self.conn.cursor()

    def check_and_mark(self, key_text: str) -> bool:
        if self.mode == "memory":
            if key_text in self.seen:
                return False
            self.seen.add(key_text)
            return True
        else:
            self.cur.execute("INSERT OR IGNORE INTO seen(k) VALUES (?)", (key_text,))
            inserted = self.cur.rowcount == 1
            self._batch += 1
            if self._batch >= 10000:
                self.conn.commit()
                self._batch = 0
            return inserted

    def close(self):
        if self.conn is not None:
            self.conn.commit()
            self.cur.close()
            self.conn.close()


def main():
    ap = argparse.ArgumentParser(
        description="Merge JSON files (each a list of dicts) into one JSONL file (streaming)."
    )

    # Support --input_dir.
    ap.add_argument("--input_dir", dest="input_dir", type=Path, help="same as positional input_dir")

    ap.add_argument(
        "-o", "--output", "--output_file", dest="output", type=Path, default=Path("merged.jsonl"),
        help="Output JSONL filepath (default: merged.jsonl)"
    )
    ap.add_argument("--pattern", default="*.json", help='Glob pattern to match files (default: "*.json")')
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--unique-key", default=None, help="Field name to de-duplicate records by key")
    ap.add_argument(
        "--last-wins", action="store_true",
        help="When using --unique-key, keep the last occurrence instead of the first (two-pass, disk-safe)."
    )
    ap.add_argument(
        "--dedup-store", choices=["memory", "sqlite"], default="sqlite",
        help="Where to store seen keys (default: sqlite)"
    )
    ap.add_argument(
        "--dedup-db", type=Path, default=None,
        help="Path to the SQLite file used for dedup keys (default: <output>.dedup.sqlite)"
    )
    ap.add_argument(
        "--skip-bad-files", action="store_true", default=True,
        help="Skip files that fail to parse (default: enabled)"
    )

    args = ap.parse_args()
    input_dir = args.input_dir
    if input_dir is None:
        ap.error("--input_dir is required.")
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    # Gather files
    globber = input_dir.rglob if args.recursive else input_dir.glob
    files = sorted(p for p in globber(args.pattern) if p.is_file())

    # Avoid reading/writing the output if it sits inside input_dir and matches pattern
    files = [p for p in files if p.resolve() != args.output.resolve()]

    if not files:
        raise SystemExit("No matching JSON files found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # De-dup setup
    deduper: Optional[Deduper] = None
    if args.unique_key:
        db_path = args.dedup_db or (args.output.parent / (args.output.name + ".dedup.sqlite"))
        deduper = Deduper(args.dedup_store, db_path if args.dedup_store == "sqlite" else None)

    # LAST-WINS: two-pass, disk-safe. Both passes skip bad files.
    if args.unique_key and args.last_wins:
        db_path = (args.dedup_db or (args.output.parent / (args.output.name + ".dedup.sqlite")))
        idx_db = sqlite3.connect(str(db_path) + ".lastwins")
        idx_db.execute("PRAGMA journal_mode=WAL;")
        idx_db.execute("PRAGMA synchronous=NORMAL;")
        idx_db.execute("CREATE TABLE IF NOT EXISTS lastpos (k TEXT PRIMARY KEY, pos INTEGER)")
        cur = idx_db.cursor()

        total_objs = 0
        bad_files = 0
        with tqdm(total=len(files), desc="Pass 1/2 (files)", unit="file") as p_files:
            for path in files:
                try:
                    for rec in iter_json_array_items(path):
                        total_objs += 1
                        key_val = rec.get(args.unique_key, None)
                        if key_val is not None:
                            ktext = key_to_text(key_val)
                            cur.execute(
                                "INSERT INTO lastpos(k, pos) VALUES (?, ?) "
                                "ON CONFLICT(k) DO UPDATE SET pos=excluded.pos",
                                (ktext, total_objs),
                            )
                except (ijson_common.IncompleteJSONError, ijson_common.JSONError, ValueError, TypeError) as e:
                    bad_files += 1
                    tqdm.write(f"[skip] {path} parse error: {e.__class__.__name__}: {e}")
                except Exception as e:
                    bad_files += 1
                    tqdm.write(f"[skip] {path} unexpected error: {e}")
                finally:
                    p_files.update(1)
        idx_db.commit()

        written = 0
        with args.output.open("w", encoding="utf-8") as out, \
             tqdm(total=len(files), desc="Pass 2/2 (files)", unit="file") as p_files2, \
             tqdm(total=0, desc="Objects written", unit="obj") as p_objs:

            pos = 0
            for path in files:
                try:
                    for rec in iter_json_array_items(path):
                        pos += 1
                        key_val = rec.get(args.unique_key, None)
                        if key_val is None:
                            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            written += 1
                            p_objs.update(1)
                        else:
                            ktext = key_to_text(key_val)
                            row = idx_db.execute("SELECT pos FROM lastpos WHERE k=?", (ktext,)).fetchone()
                            if row and row[0] == pos:
                                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                written += 1
                                p_objs.update(1)
                except (ijson_common.IncompleteJSONError, ijson_common.JSONError, ValueError, TypeError) as e:
                    tqdm.write(f"[skip] {path} parse error in pass 2: {e.__class__.__name__}: {e}")
                except Exception as e:
                    tqdm.write(f"[skip] {path} unexpected error in pass 2: {e}")
                finally:
                    p_files2.update(1)

        cur.close()
        idx_db.close()
        if deduper:
            deduper.close()
        tqdm.write(f"Skipped files (parse errors): {bad_files}")
        print(f"Merged {len(files) - bad_files} file(s) into {args.output} with {written} line(s).")
        return

    # FIRST-WINS / NO DEDUP: single-pass, skip bad files
    written = 0
    bad_files = 0
    with args.output.open("w", encoding="utf-8") as out, \
         tqdm(total=len(files), desc="Files", unit="file") as p_files, \
         tqdm(total=0, desc="Objects processed", unit="obj") as p_objs:

        for path in files:
            try:
                for rec in iter_json_array_items(path):
                    if deduper and args.unique_key is not None:
                        key_val = rec.get(args.unique_key, None)
                        if key_val is not None:
                            ktext = key_to_text(key_val)
                            if not deduper.check_and_mark(ktext):
                                p_objs.update(1)  # processed but skipped
                                continue
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    p_objs.update(1)
            except (ijson_common.IncompleteJSONError, ijson_common.JSONError, ValueError, TypeError) as e:
                bad_files += 1
                tqdm.write(f"[skip] {path} parse error: {e.__class__.__name__}: {e}")
            except Exception as e:
                bad_files += 1
                tqdm.write(f"[skip] {path} unexpected error: {e}")
            finally:
                p_files.update(1)

    if deduper:
        deduper.close()

    tqdm.write(f"Skipped files (parse errors): {bad_files}")
    print(f"Merged {len(files) - bad_files} file(s) into {args.output} with {written} line(s).")


if __name__ == "__main__":
    main()
