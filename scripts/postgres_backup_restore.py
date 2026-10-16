#!/usr/bin/env python3
"""Create a PostgreSQL backup or run a destructive restore only against an explicit drill target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import psycopg2
from psycopg2 import sql


def pg_url(value: str) -> str:
    return value.replace("postgresql+psycopg2://", "postgresql://", 1).replace("postgresql+asyncpg://", "postgresql://", 1)


def database_name(value: str) -> str:
    return urlsplit(pg_url(value)).path.lstrip("/")


def redacted(value: str) -> str:
    parsed = urlsplit(pg_url(value))
    host = parsed.hostname or "unknown"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def subprocess_connection(value: str) -> tuple[str, dict[str, str]]:
    """Keep database passwords out of command arguments and captured logs."""
    parsed = urlsplit(pg_url(value))
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    user = f"{parsed.username}@" if parsed.username else ""
    port = f":{parsed.port}" if parsed.port else ""
    dsn = urlunsplit((parsed.scheme, f"{user}{host}{port}", parsed.path, parsed.query, ""))
    environment = os.environ.copy()
    if parsed.password:
        environment["PGPASSWORD"] = unquote(parsed.password)
    return dsn, environment


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_counts(dsn: str) -> dict[str, int]:
    with psycopg2.connect(pg_url(dsn)) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename")
        tables = [row[0] for row in cursor.fetchall() if row[0] != "alembic_version"]
        result = {}
        for table in tables:
            cursor.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
            result[table] = int(cursor.fetchone()[0])
        if not result:
            raise RuntimeError("No public tables were found; refusing to treat an empty restore as verified")
        return result


def create_backup(source: str, output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_dsn, environment = subprocess_connection(source)
    subprocess.run(
        ["pg_dump", "--format=custom", "--no-owner", "--no-privileges", "--file", str(output), source_dsn],
        check=True,
        env=environment,
    )
    value = checksum(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{value}  {output.name}\n", encoding="utf-8")
    return value


def restore_drill(source: str, target: str, backup: Path, confirmation: str) -> dict:
    source_name = database_name(source)
    target_name = database_name(target)
    if not target_name or confirmation != target_name:
        raise ValueError("--confirm-target-db must exactly match the restore target database")
    if pg_url(source) == pg_url(target) or source_name == target_name:
        raise ValueError("The restore target must be a separate disposable database")
    expected = backup.with_suffix(backup.suffix + ".sha256").read_text(encoding="utf-8").split()[0]
    actual = checksum(backup)
    if actual != expected:
        raise ValueError("Backup checksum verification failed")
    before = table_counts(source)
    target_dsn, environment = subprocess_connection(target)
    subprocess.run([
        "pg_restore", "--clean", "--if-exists", "--no-owner", "--no-privileges",
        "--dbname", target_dsn, str(backup),
    ], check=True, env=environment)
    after = table_counts(target)
    return {
        "restore_passed": before == after,
        "row_count_verified": before == after,
        "checksum_verified": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source": redacted(source),
        "target": redacted(target),
        "backup_sha256": actual,
        "tables_verified": len(before),
        "mismatches": {
            table: {"source": before.get(table), "target": after.get(table)}
            for table in sorted(set(before) | set(after)) if before.get(table) != after.get(table)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("--source", required=True)
    backup_parser.add_argument("--output", required=True, type=Path)
    drill_parser = subparsers.add_parser("drill")
    drill_parser.add_argument("--source", required=True)
    drill_parser.add_argument("--target", required=True)
    drill_parser.add_argument("--backup", required=True, type=Path)
    drill_parser.add_argument("--confirm-target-db", required=True)
    drill_parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.command == "backup":
        value = create_backup(args.source, args.output)
        print(json.dumps({"backup": str(args.output), "sha256": value, "source": redacted(args.source)}, indent=2))
        return 0
    result = restore_drill(args.source, args.target, args.backup, args.confirm_target_db)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["restore_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
