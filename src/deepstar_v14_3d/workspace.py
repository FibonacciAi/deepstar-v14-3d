from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import APP_BUILD, Settings


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Job:
    identifier: str
    root: Path
    source_dir: Path
    output_dir: Path
    log_dir: Path
    manifest_path: Path

    @classmethod
    def create(cls, settings: Settings) -> "Job":
        settings.prepare()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        identifier = f"{stamp}-{uuid.uuid4().hex[:8]}"
        root = settings.home / "jobs" / identifier
        source = root / "source"
        output = root / "outputs"
        logs = root / "logs"
        for directory in (source, output, logs):
            directory.mkdir(parents=True, exist_ok=False)
        job = cls(identifier, root, source, output, logs, root / "manifest.json")
        job.write_manifest({"status": "created"})
        return job

    @classmethod
    def open(cls, settings: Settings, identifier: str) -> "Job":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Invalid job identifier")
        root = settings.home / "jobs" / identifier
        return cls(
            identifier,
            root,
            root / "source",
            root / "outputs",
            root / "logs",
            root / "manifest.json",
        )

    def copy_source(self, source: Path) -> Path:
        clean_name = SAFE_NAME.sub("-", source.name).strip(".-") or "source-image"
        destination = self.source_dir / clean_name
        shutil.copy2(source, destination)
        return destination

    def write_manifest(self, updates: dict[str, Any]) -> dict[str, Any]:
        existing: dict[str, Any] = {}
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
        existing.update(updates)
        existing.setdefault("schema_version", 1)
        existing.setdefault("app", "deepstar-v14-3d")
        existing.setdefault("app_build", APP_BUILD)
        existing.setdefault("job_id", self.identifier)
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.manifest_path, existing)
        return existing

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

