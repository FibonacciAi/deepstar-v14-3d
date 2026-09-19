from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "Deepstar v14 3D"
APP_BUILD = "14.4.3"
DEFAULT_PORT = 47143
SHARP_CHECKPOINT_NAME = "sharp_2572gikvuh.pt"


def _first_existing_file(candidates: tuple[Path, ...], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return fallback


@dataclass(frozen=True)
class Settings:
    home: Path
    cache: Path
    sharp_executable: Path
    accept_apple_research_license: bool
    image_command: str | None
    sharp_checkpoint: Path | None = None

    @classmethod
    def from_environment(cls) -> "Settings":
        repo = Path(__file__).resolve().parents[2]
        user_home = Path.home()
        home = Path(
            os.environ.get(
                "DEEPSTAR3D_HOME",
                Path.home() / "Library" / "Application Support" / APP_NAME,
            )
        ).expanduser()
        cache = Path(
            os.environ.get(
                "DEEPSTAR3D_CACHE",
                Path.home() / "Library" / "Caches" / APP_NAME,
            )
        ).expanduser()
        repo_sharp = repo / ".venv-sharp" / "bin" / "sharp"
        external_sharp = user_home / ".spectra" / "miniconda" / "envs" / "sharp" / "bin" / "sharp"
        configured_sharp = os.environ.get("DEEPSTAR3D_SHARP")
        sharp = (
            Path(configured_sharp).expanduser()
            if configured_sharp
            else _first_existing_file((repo_sharp, external_sharp), repo_sharp)
        )
        configured_checkpoint = os.environ.get("DEEPSTAR3D_SHARP_CHECKPOINT")
        checkpoint_candidates = (
            cache / "torch" / "hub" / "checkpoints" / SHARP_CHECKPOINT_NAME,
            user_home / ".cache" / "torch" / "hub" / "checkpoints" / SHARP_CHECKPOINT_NAME,
        )
        if configured_checkpoint:
            sharp_checkpoint = Path(configured_checkpoint).expanduser()
        else:
            sharp_checkpoint = next(
                (candidate for candidate in checkpoint_candidates if candidate.is_file()),
                None,
            )
        return cls(
            home=home,
            cache=cache,
            sharp_executable=sharp,
            accept_apple_research_license=(
                os.environ.get("DEEPSTAR3D_ACCEPT_APPLE_RESEARCH_LICENSE") == "1"
            ),
            image_command=os.environ.get("DEEPSTAR3D_IMAGE_COMMAND"),
            sharp_checkpoint=sharp_checkpoint,
        )

    def prepare(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.home / "jobs").mkdir(parents=True, exist_ok=True)
        (self.cache / "torch").mkdir(parents=True, exist_ok=True)
