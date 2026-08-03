from __future__ import annotations

import math
import os
import shlex
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .errors import (
    AppleResearchLicenseRequired,
    BackendUnavailable,
    TrainingActive,
)


TRAINING_PATTERNS = (
    "deepstar_models.train",
    "deepstar-train",
    "deepstar_models.foundry",
    "deepstar-foundry",
    "render_mpfb_faces",
    "torchrun",
    "accelerate launch",
)


def active_training_processes() -> list[str]:
    pgrep = shutil.which("pgrep")
    if not pgrep:
        return []
    result = subprocess.run(
        [pgrep, "-fl", "deepstar|torchrun|accelerate"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        return []
    own_pid = str(os.getpid())
    matches = []
    for line in result.stdout.splitlines():
        if line.split(maxsplit=1)[0] == own_pid:
            continue
        lowered = line.lower()
        direct = any(pattern in lowered for pattern in TRAINING_PATTERNS)
        source_entry = "deepstar-models" in lowered and "train.py" in lowered
        if direct or source_entry:
            matches.append(line)
    return matches


@dataclass(frozen=True)
class BackendResult:
    scene: Path
    identifier: str
    research_only: bool


class AppleSharpBackend:
    identifier = "apple-ml-sharp-research"

    def __init__(self, settings: Settings, device: str = "default") -> None:
        self.settings = settings
        self.device = device

    def run(
        self,
        source: Path,
        output_dir: Path,
        log_path: Path,
        allow_concurrent_training: bool = False,
    ) -> BackendResult:
        if not self.settings.accept_apple_research_license:
            raise AppleResearchLicenseRequired(
                "Apple SHARP weights are research-only. Read THIRD_PARTY.md and set "
                "DEEPSTAR3D_ACCEPT_APPLE_RESEARCH_LICENSE=1 to accept for research use."
            )
        if not self.settings.sharp_executable.is_file():
            raise BackendUnavailable(
                "Apple SHARP is not installed in this repo. Run "
                "scripts/install_apple_sharp_research.sh --accept-research-license."
            )
        training = active_training_processes()
        if training and not allow_concurrent_training:
            raise TrainingActive("Deepstar training is active; 3D inference was not started.")

        output_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            str(self.settings.sharp_executable),
            "predict",
            "-i",
            str(source),
            "-o",
            str(output_dir),
            "--device",
            self.device,
            "--no-render",
        ]
        nice = shutil.which("nice")
        if nice:
            argv = [nice, "-n", "10", *argv]
        environment = os.environ.copy()
        environment["TORCH_HOME"] = str(self.settings.cache / "torch")
        environment["XDG_CACHE_HOME"] = str(self.settings.cache / "xdg")
        environment["HF_HOME"] = str(self.settings.cache / "huggingface")
        with log_path.open("w") as log:
            log.write("argv=" + shlex.join(argv) + "\n")
            log.flush()
            result = subprocess.run(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        if result.returncode != 0:
            raise BackendUnavailable(
                f"Apple SHARP exited with {result.returncode}; see {log_path}."
            )
        predicted = output_dir / f"{source.stem}.ply"
        if not predicted.is_file():
            candidates = list(output_dir.glob("*.ply"))
            if len(candidates) != 1:
                raise BackendUnavailable("Apple SHARP completed without one .ply output.")
            predicted = candidates[0]
        scene = output_dir / "scene.ply"
        if predicted != scene:
            predicted.replace(scene)
        return BackendResult(scene, self.identifier, True)


class DepthCardBackend:
    identifier = "deepstar-depth-card-v1"

    def run(self, source: Path, output_dir: Path, log_path: Path) -> BackendResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        bitmap = output_dir / ".source.bmp"
        sips = shutil.which("sips")
        if not sips:
            raise BackendUnavailable("The depth-card backend requires macOS sips.")
        result = subprocess.run(
            [sips, "-s", "format", "bmp", str(source), "--out", str(bitmap)],
            capture_output=True,
            text=True,
            check=False,
        )
        log_path.write_text(result.stdout + result.stderr)
        if result.returncode != 0 or not bitmap.is_file():
            raise BackendUnavailable("macOS could not decode the selected image.")
        width, height, pixels = _read_bmp(bitmap)
        bitmap.unlink(missing_ok=True)
        scene = output_dir / "scene.ply"
        _write_depth_card(scene, width, height, pixels)
        return BackendResult(scene, self.identifier, False)


def _read_bmp(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise BackendUnavailable("Unsupported bitmap from image decoder.")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_size = struct.unpack_from("<I", data, 14)[0]
    if dib_size < 40:
        raise BackendUnavailable("Unsupported bitmap header.")
    width, signed_height = struct.unpack_from("<ii", data, 18)
    planes, bits = struct.unpack_from("<HH", data, 26)
    compression = struct.unpack_from("<I", data, 30)[0]
    if width <= 0 or signed_height == 0 or planes != 1 or bits not in (24, 32) or compression != 0:
        raise BackendUnavailable("Depth-card supports uncompressed 24/32-bit bitmap input.")
    height = abs(signed_height)
    bottom_up = signed_height > 0
    bytes_per_pixel = bits // 8
    row_stride = ((width * bits + 31) // 32) * 4
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        source_y = height - 1 - y if bottom_up else y
        row = pixel_offset + source_y * row_stride
        for x in range(width):
            index = row + x * bytes_per_pixel
            blue, green, red = data[index : index + 3]
            pixels.append((red, green, blue))
    return width, height, pixels


def _write_depth_card(
    path: Path,
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
) -> None:
    max_side = 192
    step = max(1, math.ceil(max(width, height) / max_side))
    samples = [(x, y) for y in range(0, height, step) for x in range(0, width, step)]
    aspect = width / height
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(samples)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("comment Deepstar depth-card preview; not a learned reconstruction\n")
        stream.write("end_header\n")
        for x, y in samples:
            red, green, blue = pixels[y * width + x]
            luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
            nx = ((x + 0.5) / width - 0.5) * 2.0 * aspect
            ny = -(((y + 0.5) / height - 0.5) * 2.0)
            radius = min(1.0, math.sqrt((nx / max(aspect, 0.01)) ** 2 + ny**2))
            z = 0.18 * (1.0 - luminance) + 0.08 * (1.0 - radius)
            stream.write(f"{nx:.6f} {ny:.6f} {z:.6f} {red} {green} {blue}\n")
