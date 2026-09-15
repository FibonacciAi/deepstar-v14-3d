from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from . import __version__
from .backends import active_training_processes
from .config import DEFAULT_PORT, Settings
from .errors import Deepstar3DError
from .pipeline import create_scene


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="deepstar3d")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Inspect isolated runtime readiness.")

    create = commands.add_parser("create", help="Create one local 3D scene.")
    source = create.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--prompt")
    create.add_argument(
        "--backend",
        choices=("depth-card", "apple-sharp", "vision-sharp-hybrid"),
        default="depth-card",
    )
    create.add_argument("--device", choices=("default", "cpu", "mps", "cuda"), default="default")
    create.add_argument("--confirm-rights", action="store_true")
    create.add_argument("--allow-concurrent-training", action="store_true")

    serve = commands.add_parser("serve", help="Run the local studio UI.")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--open", action="store_true", dest="open_browser")
    return root


def doctor(settings: Settings) -> int:
    training = active_training_processes()
    payload = {
        "app": "Deepstar v14 3D",
        "version": __version__,
        "python": platform.python_version(),
        "workspace": str(settings.home),
        "cache": str(settings.cache),
        "depth_card": {"available": platform.system() == "Darwin"},
        "apple_sharp": {
            "executable": str(settings.sharp_executable),
            "installed": settings.sharp_executable.is_file(),
            "checkpoint": (
                str(settings.sharp_checkpoint) if settings.sharp_checkpoint else None
            ),
            "checkpoint_present": bool(
                settings.sharp_checkpoint and settings.sharp_checkpoint.is_file()
            ),
            "research_license_accepted": settings.accept_apple_research_license,
        },
        "vision_sharp_hybrid": {
            "identifier": "deepstar-vision-guided-apple-sharp-research",
            "available": (
                settings.sharp_executable.is_file()
                and settings.accept_apple_research_license
            ),
            "requires": ["apple-vision", "apple-sharp", "research-license-acceptance"],
            "ply_postprocessed": False,
        },
        "training_guard": {
            "active": bool(training),
            "matches": training,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    settings = Settings.from_environment()
    try:
        if arguments.command == "doctor":
            return doctor(settings)
        if arguments.command == "create":
            job = create_scene(
                settings,
                image=arguments.image,
                prompt=arguments.prompt,
                backend=arguments.backend,
                confirm_rights=arguments.confirm_rights,
                allow_concurrent_training=arguments.allow_concurrent_training,
                device=arguments.device,
            )
            print(
                json.dumps(
                    {
                        "job": job.identifier,
                        "path": str(job.root),
                        "scene": str(job.output_dir / "scene.ply"),
                        "package": str(job.output_dir / "scene-package.zip"),
                    },
                    indent=2,
                )
            )
            return 0
        if arguments.command == "serve":
            from .server import serve

            serve(settings, arguments.port, arguments.open_browser)
            return 0
    except (Deepstar3DError, FileNotFoundError, ValueError) as error:
        print(f"deepstar3d: {error}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
