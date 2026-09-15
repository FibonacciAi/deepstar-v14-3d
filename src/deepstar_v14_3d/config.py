from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "Deepstar v14 3D"
APP_BUILD = "14.4.0"
DEFAULT_PORT = 47143


@dataclass(frozen=True)
class Settings:
    home: Path
    cache: Path
    sharp_executable: Path
    accept_apple_research_license: bool
    image_command: str | None

    @classmethod
    def from_environment(cls) -> "Settings":
        repo = Path(__file__).resolve().parents[2]
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
        sharp = Path(
            os.environ.get(
                "DEEPSTAR3D_SHARP",
                repo / ".venv-sharp" / "bin" / "sharp",
            )
        ).expanduser()
        return cls(
            home=home,
            cache=cache,
            sharp_executable=sharp,
            accept_apple_research_license=(
                os.environ.get("DEEPSTAR3D_ACCEPT_APPLE_RESEARCH_LICENSE") == "1"
            ),
            image_command=os.environ.get("DEEPSTAR3D_IMAGE_COMMAND"),
        )

    def prepare(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.home / "jobs").mkdir(parents=True, exist_ok=True)
        (self.cache / "torch").mkdir(parents=True, exist_ok=True)
