from __future__ import annotations

import hashlib
from pathlib import Path

from .backends import AppleSharpBackend, DepthCardBackend
from .config import Settings
from .errors import RightsConfirmationRequired
from .imagegen import generate_with_command
from .workspace import Job, sha256_file


def create_scene(
    settings: Settings,
    *,
    image: Path | None = None,
    prompt: str | None = None,
    backend: str = "depth-card",
    confirm_rights: bool = False,
    allow_concurrent_training: bool = False,
    device: str = "default",
) -> Job:
    if not confirm_rights:
        raise RightsConfirmationRequired(
            "Confirm that you have permission to use the image and likenesses."
        )
    if (image is None) == (prompt is None):
        raise ValueError("Provide exactly one of image or prompt.")

    job = Job.create(settings)
    try:
        if image is not None:
            if not image.is_file():
                raise FileNotFoundError(image)
            source = job.copy_source(image)
            source_kind = "explicit-image"
            prompt_hash = None
            generator = None
        else:
            generated = job.source_dir / "generated.png"
            source = generate_with_command(settings.image_command, prompt or "", generated)
            source_kind = "configured-image-generator"
            prompt_hash = hashlib.sha256((prompt or "").encode()).hexdigest()
            generator = "external-command"

        job.write_manifest(
            {
                "status": "generating-3d",
                "rights_confirmed": True,
                "source": {
                    "kind": source_kind,
                    "filename": source.name,
                    "sha256": sha256_file(source),
                    "prompt_sha256": prompt_hash,
                    "generator": generator,
                },
                "requested_backend": backend,
            }
        )
        log = job.log_dir / f"{backend}.log"
        if backend == "depth-card":
            result = DepthCardBackend(settings).run(source, job.output_dir, log)
        elif backend == "apple-sharp":
            result = AppleSharpBackend(settings, device=device).run(
                source,
                job.output_dir,
                log,
                allow_concurrent_training=allow_concurrent_training,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        job.write_manifest(
            {
                "status": "complete",
                "backend": {
                    "identifier": result.identifier,
                    "research_only": result.research_only,
                    "commercial_use_approved": False if result.research_only else None,
                    "details": result.metadata,
                },
                "scene": {
                    "filename": result.scene.name,
                    "sha256": sha256_file(result.scene),
                    "format": "ply",
                },
            }
        )
        return job
    except Exception as error:
        job.write_manifest(
            {
                "status": "failed",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        raise
