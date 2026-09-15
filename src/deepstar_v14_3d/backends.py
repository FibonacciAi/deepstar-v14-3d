from __future__ import annotations

import json
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
    metadata: dict[str, object] | None = None


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
    identifier = "deepstar-vision-volume-v2"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

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
        analysis = _run_vision_analysis(self.settings, bitmap, output_dir, log_path)
        bitmap.unlink(missing_ok=True)
        scene = output_dir / "scene.ply"
        geometry = _write_spatial_portrait(scene, width, height, pixels, analysis)
        metadata: dict[str, object] = {
            "geometry": geometry,
            "vision_engine": analysis.engine if analysis else None,
            "face_count": len(analysis.faces) if analysis else 0,
            "subject_masked": analysis is not None,
        }
        return BackendResult(scene, self.identifier, False, metadata)


@dataclass(frozen=True)
class VisionAnalysis:
    engine: str
    mask_width: int
    mask_height: int
    mask: bytes
    faces: list[dict[str, float]]


def _run_vision_analysis(
    settings: Settings,
    bitmap: Path,
    output_dir: Path,
    log_path: Path,
) -> VisionAnalysis | None:
    source = Path(__file__).with_name("tools") / "DeepstarVisionGeometry.swift"
    xcrun = shutil.which("xcrun")
    if not source.is_file() or not xcrun:
        return None

    binary_dir = settings.cache / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)
    binary = binary_dir / "DeepstarVisionGeometry-v1"
    module_cache = settings.cache / "swift-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    compile_environment = os.environ.copy()
    compile_environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    compile_environment["SWIFT_MODULECACHE_PATH"] = str(module_cache)
    mask_path = output_dir / "subject-mask.pgm"
    analysis_path = output_dir / "vision-analysis.json"

    with log_path.open("a") as log:
        if not binary.is_file() or binary.stat().st_mtime < source.stat().st_mtime:
            temporary = binary.with_suffix(".tmp")
            compile_result = subprocess.run(
                [
                    xcrun,
                    "swiftc",
                    str(source),
                    "-O",
                    "-framework",
                    "Vision",
                    "-framework",
                    "CoreImage",
                    "-framework",
                    "CoreVideo",
                    "-o",
                    str(temporary),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=compile_environment,
                check=False,
            )
            if compile_result.returncode != 0 or not temporary.is_file():
                temporary.unlink(missing_ok=True)
                log.write("Apple Vision helper compilation failed; using geometric fallback.\n")
                return None
            temporary.replace(binary)

        run_result = subprocess.run(
            [str(binary), str(bitmap), str(mask_path), str(analysis_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
        if run_result.returncode != 0 or not mask_path.is_file() or not analysis_path.is_file():
            log.write("Apple Vision analysis failed; using geometric fallback.\n")
            return None

    try:
        mask_width, mask_height, mask = _read_pgm(mask_path)
        payload = json.loads(analysis_path.read_text())
        if int(payload.get("subjectPixelCount", 0)) < 32:
            return None
        faces = [
            {
                "x": float(face["x"]),
                "y": float(face["y"]),
                "width": float(face["width"]),
                "height": float(face["height"]),
                "confidence": float(face["confidence"]),
            }
            for face in payload.get("faces", [])
        ]
        return VisionAnalysis(
            engine=str(payload.get("engine", "apple-vision")),
            mask_width=mask_width,
            mask_height=mask_height,
            mask=mask,
            faces=faces,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


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
    supported_layout = (bits == 24 and compression == 0) or (
        bits == 32 and compression in (0, 3)
    )
    if width <= 0 or signed_height == 0 or planes != 1 or not supported_layout:
        raise BackendUnavailable("macOS produced an unsupported bitmap layout.")
    height = abs(signed_height)
    bottom_up = signed_height > 0
    bytes_per_pixel = bits // 8
    row_stride = ((width * bits + 31) // 32) * 4
    channel_masks: tuple[int, int, int] | None = None
    if bits == 32 and compression == 3:
        mask_offset = 14 + 40 if dib_size >= 52 else 14 + dib_size
        if mask_offset + 12 > len(data):
            raise BackendUnavailable("macOS produced a truncated alpha bitmap.")
        channel_masks = struct.unpack_from("<III", data, mask_offset)
        if any(mask == 0 for mask in channel_masks):
            raise BackendUnavailable("macOS produced an invalid alpha bitmap.")
    pixels: list[tuple[int, int, int]] = []
    for y in range(height):
        source_y = height - 1 - y if bottom_up else y
        row = pixel_offset + source_y * row_stride
        if row < 0 or row + width * bytes_per_pixel > len(data):
            raise BackendUnavailable("macOS produced a truncated bitmap.")
        for x in range(width):
            index = row + x * bytes_per_pixel
            if channel_masks:
                packed = struct.unpack_from("<I", data, index)[0]
                red, green, blue = (
                    _scaled_mask_channel(packed, mask) for mask in channel_masks
                )
            else:
                blue, green, red = data[index : index + 3]
            pixels.append((red, green, blue))
    return width, height, pixels


def _scaled_mask_channel(pixel: int, mask: int) -> int:
    least_bit = mask & -mask
    shift = least_bit.bit_length() - 1
    maximum = mask >> shift
    return round(((pixel & mask) >> shift) * 255 / maximum)


def _read_pgm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"P5"):
        raise ValueError("Unsupported Apple Vision mask")
    index = 2
    tokens: list[bytes] = []
    while len(tokens) < 3:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(data[start:index])
    width, height, maximum = (int(token) for token in tokens)
    if width <= 0 or height <= 0 or maximum != 255:
        raise ValueError("Invalid Apple Vision mask")
    while index < len(data) and data[index] in b" \t\r\n":
        index += 1
    mask = data[index : index + width * height]
    if len(mask) != width * height:
        raise ValueError("Truncated Apple Vision mask")
    return width, height, mask


def _mask_value(analysis: VisionAnalysis | None, u: float, v: float) -> int:
    if analysis is None:
        return 255
    x = min(analysis.mask_width - 1, max(0, int(u * analysis.mask_width)))
    y = min(analysis.mask_height - 1, max(0, int(v * analysis.mask_height)))
    return analysis.mask[y * analysis.mask_width + x]


def _gaussian(x: float, y: float, cx: float, cy: float, sx: float, sy: float) -> float:
    return math.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2) * 0.5)


def _portrait_depth(
    u: float,
    v: float,
    luminance: float,
    face: dict[str, float] | None,
) -> float:
    horizontal = (u - 0.5) * 2.0
    body = 0.06 + 0.10 * math.sqrt(max(0.0, 1.0 - min(1.0, horizontal * horizontal)))
    body *= 0.65 + 0.35 * min(1.0, max(0.0, (v - 0.28) / 0.72))
    if face is None:
        radial = math.sqrt(horizontal * horizontal + ((v - 0.48) * 1.35) ** 2)
        return body + 0.20 * max(0.0, 1.0 - radial) + 0.035 * (0.5 - luminance)

    face_cx = face["x"] + face["width"] * 0.5
    face_top = 1.0 - face["y"] - face["height"]
    face_cy = face_top + face["height"] * 0.5

    head_width = face["width"] * 1.48
    head_height = face["height"] * 1.62
    head_cy = face_cy - face["height"] * 0.08
    hx = (u - face_cx) / max(head_width * 0.5, 0.001)
    hy = (v - head_cy) / max(head_height * 0.5, 0.001)
    head_r2 = hx * hx + hy * hy
    head = 0.34 * math.sqrt(max(0.0, 1.0 - head_r2)) if head_r2 < 1.0 else 0.0

    fx = (u - face_cx) / max(face["width"] * 0.5, 0.001)
    fy = (v - face_cy) / max(face["height"] * 0.5, 0.001)
    nose = 0.105 * _gaussian(fx, fy, 0.0, 0.02, 0.22, 0.30)
    left_cheek = 0.045 * _gaussian(fx, fy, -0.45, 0.16, 0.28, 0.30)
    right_cheek = 0.045 * _gaussian(fx, fy, 0.45, 0.16, 0.28, 0.30)
    eye_sockets = -0.025 * (
        _gaussian(fx, fy, -0.34, -0.24, 0.20, 0.12)
        + _gaussian(fx, fy, 0.34, -0.24, 0.20, 0.12)
    )
    chin = 0.035 * _gaussian(fx, fy, 0.0, 0.72, 0.32, 0.20)
    detail = 0.025 * (0.5 - luminance) * max(0.0, 1.0 - min(1.0, head_r2))
    return body + head + nose + left_cheek + right_cheek + eye_sockets + chin + detail


def _write_spatial_portrait(
    path: Path,
    width: int,
    height: int,
    pixels: list[tuple[int, int, int]],
    analysis: VisionAnalysis | None,
) -> dict[str, object]:
    max_side = 256
    step = max(1, math.ceil(max(width, height) / max_side))
    aspect = width / height
    face = max(analysis.faces, key=lambda item: item["width"] * item["height"]) if analysis and analysis.faces else None
    front: dict[tuple[int, int], tuple[float, float, float, int, int, int, int]] = {}
    back_depth: dict[tuple[int, int], float] = {}
    x_values = list(range(0, width, step))
    y_values = list(range(0, height, step))

    for gy, y in enumerate(y_values):
        for gx, x in enumerate(x_values):
            u = (x + 0.5) / width
            v = (y + 0.5) / height
            confidence = _mask_value(analysis, u, v)
            if analysis is not None and confidence < 18:
                continue
            red, green, blue = pixels[y * width + x]
            luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
            nx = (u - 0.5) * 2.0 * aspect
            ny = -((v - 0.5) * 2.0)
            z = _portrait_depth(u, v, luminance, face)
            alpha = max(72, min(255, confidence)) if analysis else 255
            front[(gx, gy)] = (nx, ny, z, red, green, blue, alpha)
            back_depth[(gx, gy)] = -0.12 - 0.035 * (1.0 - confidence / 255.0)

    vertices = list(front.values())
    for (gx, gy), vertex in front.items():
        if (gx + gy) % 2 != 0:
            continue
        x, y, _, red, green, blue, alpha = vertex
        vertices.append(
            (
                x,
                y,
                back_depth[(gx, gy)],
                int(red * 0.38),
                int(green * 0.38),
                int(blue * 0.42),
                min(alpha, 210),
            )
        )

    boundary_count = 0
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for key, vertex in front.items():
        gx, gy = key
        if all((gx + dx, gy + dy) in front for dx, dy in neighbors):
            continue
        boundary_count += 1
        x, y, front_z, red, green, blue, alpha = vertex
        rear_z = back_depth[key]
        for layer in range(1, 6):
            amount = layer / 6.0
            z = front_z * (1.0 - amount) + rear_z * amount
            shade = 0.72 - amount * 0.30
            vertices.append(
                (x, y, z, int(red * shade), int(green * shade), int(blue * shade), alpha)
            )

    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property uchar alpha\n")
        stream.write("comment Deepstar Apple Vision spatial portrait; not a learned depth model\n")
        stream.write("end_header\n")
        for x, y, z, red, green, blue, alpha in vertices:
            stream.write(f"{x:.6f} {y:.6f} {z:.6f} {red} {green} {blue} {alpha}\n")
    return {
        "front_points": len(front),
        "boundary_points": boundary_count,
        "total_points": len(vertices),
        "portrait_face_used": face is not None,
    }
