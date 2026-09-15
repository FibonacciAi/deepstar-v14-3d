from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import APP_BUILD, Settings


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
EXPORT_SCHEMA_VERSION = 1


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

    def export_package(self) -> Path:
        """Create a portable scene package without copying the source image.

        The package deliberately contains only the generated scene, its
        provenance receipt, and source-independent instructions/metadata. It
        is written beside the scene and replaced atomically so a browser
        download never observes a partial archive.
        """
        manifest = self.manifest()
        scene = self.output_dir / "scene.ply"
        if manifest.get("status") != "complete" or not scene.is_file():
            raise ValueError("A completed scene is required before exporting.")

        package = self.output_dir / "scene-package.zip"
        temporary = self.output_dir / f".scene-package-{uuid.uuid4().hex}.zip"
        metadata = {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "job_id": self.identifier,
            "app": manifest.get("app", "deepstar-v14-3d"),
            "app_build": manifest.get("app_build"),
            "scene": manifest.get("scene", {}),
            "backend": manifest.get("backend", {}),
            "privacy": {
                "source_image_included": False,
                "prompt_text_included": False,
                "model_weights_included": False,
                "derived_vision_guidance_included": False,
                "local_only": True,
            },
        }
        guidance: list[tuple[Path, str]] = []
        for filename in ("subject-mask.pgm", "vision-analysis.json"):
            sidecar = self.output_dir / filename
            if sidecar.is_file():
                guidance.append((sidecar, f"guidance/{filename}"))
        metadata["privacy"]["derived_vision_guidance_included"] = bool(guidance)
        metadata["guidance_entries"] = [archive_name for _, archive_name in guidance]
        readme = _export_readme(manifest)
        try:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                archive.write(scene, "scene.ply")
                archive.writestr(
                    "manifest.json",
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                )
                archive.writestr(
                    "metadata.json",
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                )
                archive.writestr("README.txt", readme)
                for sidecar, archive_name in guidance:
                    archive.write(sidecar, archive_name)
            temporary.replace(package)
        finally:
            temporary.unlink(missing_ok=True)
        return package


def _export_readme(manifest: dict[str, Any]) -> str:
    backend = manifest.get("backend", {})
    identifier = backend.get("identifier", "unknown")
    research = bool(backend.get("research_only"))
    rights = "confirmed in the Deepstar v14 3D job receipt"
    license_note = (
        "This scene was produced by the Apple SHARP research backend; review "
        "THIRD_PARTY.md and Apple's research license before sharing or using it."
        if research
        else "This scene was produced by the local Apple Vision spatial portrait backend."
    )
    return f"""Deepstar v14 3D scene package
===============================

This archive contains a generated PLY scene and a provenance receipt. It does
not contain the original source image, prompt text, credentials, or model
weights. The workspace is local-only; sharing this archive is an explicit user
action.

Backend: {identifier}
Rights: {rights}

Files
-----
scene.ply     Generated scene. Open it in MeshLab, Blender (with a PLY importer),
              or another PLY/3D Gaussian-compatible viewer.
manifest.json Full job receipt, including source and scene fingerprints.
metadata.json Export metadata and privacy boundaries without source pixels.
README.txt    This usage and rights note.
guidance/     Optional Apple Vision subject mask and face-analysis metadata.

Use
---
1. Extract the archive to a folder you control.
2. Open scene.ply in your preferred 3D tool.
3. Keep manifest.json with the scene when moving it so provenance stays attached.

{license_note}
"""
