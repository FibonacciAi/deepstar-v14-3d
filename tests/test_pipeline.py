from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from deepstar_v14_3d.backends import (
    AppleSharpBackend,
    AppleVisionSharpHybridBackend,
    BackendResult,
    VisionAnalysis,
    _read_bmp,
    _read_pgm,
    _write_spatial_portrait,
)
from deepstar_v14_3d.config import Settings
from deepstar_v14_3d.errors import AppleResearchLicenseRequired, RightsConfirmationRequired, TrainingActive
from deepstar_v14_3d.pipeline import create_scene


def write_bmp(path: Path, width: int = 4, height: int = 3) -> None:
    row = ((width * 3 + 3) // 4) * 4
    pixels = bytearray()
    for y in range(height):
        scan = bytearray()
        for x in range(width):
            scan.extend((40 + x * 20, 60 + y * 25, 120 + x * 10))
        scan.extend(b"\0" * (row - width * 3))
        pixels.extend(scan)
    offset = 54
    size = offset + len(pixels)
    header = b"BM" + struct.pack("<IHHI", size, 0, 0, offset)
    header += struct.pack("<IiiHHIIIIII", 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    path.write_bytes(header + pixels)


def write_alpha_bitfields_bmp(path: Path) -> None:
    width, height = 2, 1
    dib_size = 124
    pixel_offset = 14 + dib_size
    pixels = struct.pack("<II", 0xFF1E140A, 0x806496C8)
    header = b"BM" + struct.pack("<IHHI", pixel_offset + len(pixels), 0, 0, pixel_offset)
    dib = struct.pack(
        "<IiiHHIIIIII",
        dib_size,
        width,
        -height,
        1,
        32,
        3,
        len(pixels),
        2835,
        2835,
        0,
        0,
    )
    masks = struct.pack("<IIII", 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    dib += masks + b"\0" * (dib_size - len(dib) - len(masks))
    path.write_bytes(header + dib + pixels)


class PipelineTests(unittest.TestCase):
    def settings(self, root: Path, *, accepted: bool = False, sharp: Path | None = None) -> Settings:
        return Settings(root / "home", root / "cache", sharp or root / "missing-sharp", accepted, None)

    def test_rights_confirmation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.bmp"
            write_bmp(image)
            with self.assertRaises(RightsConfirmationRequired):
                create_scene(self.settings(root), image=image)

    def test_depth_card_pipeline_creates_isolated_scene_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.bmp"
            write_bmp(image)
            settings = self.settings(root)
            with patch("deepstar_v14_3d.backends.shutil.which", return_value="/usr/bin/sips"), patch(
                "deepstar_v14_3d.backends.subprocess.run"
            ) as run, patch("deepstar_v14_3d.backends._run_vision_analysis", return_value=None):
                def fake_sips(argv, **kwargs):
                    Path(argv[-1]).write_bytes(image.read_bytes())
                    return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
                run.side_effect = fake_sips
                job = create_scene(settings, image=image, backend="depth-card", confirm_rights=True)
            manifest = json.loads(job.manifest_path.read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["backend"]["identifier"], "deepstar-vision-volume-v2")
            self.assertEqual(manifest["export"]["filename"], "scene-package.zip")
            scene = job.output_dir / "scene.ply"
            self.assertTrue(scene.is_file())
            self.assertIn("property uchar alpha", scene.read_text().split("end_header", 1)[0])
            self.assertGreater(manifest["backend"]["details"]["geometry"]["total_points"], 0)
            self.assertTrue(str(job.root).startswith(str(settings.home)))
            package = job.output_dir / "scene-package.zip"
            self.assertTrue(package.is_file())
            with zipfile.ZipFile(package) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"scene.ply", "manifest.json", "metadata.json", "README.txt"},
                )
                packaged_manifest = json.loads(archive.read("manifest.json"))
                metadata = json.loads(archive.read("metadata.json"))
                self.assertEqual(packaged_manifest["job_id"], job.identifier)
                self.assertFalse(metadata["privacy"]["source_image_included"])
                self.assertNotIn("source/", archive.namelist())
                self.assertIn("Open it in MeshLab", archive.read("README.txt").decode())

    def test_reads_sips_alpha_bitfields_bitmap_from_transparent_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bitmap = Path(temporary) / "source.bmp"
            write_alpha_bitfields_bmp(bitmap)
            width, height, pixels = _read_bmp(bitmap)
            self.assertEqual((width, height), (2, 1))
            self.assertEqual(pixels, [(30, 20, 10), (100, 150, 200)])

    def test_apple_backend_requires_explicit_research_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(AppleResearchLicenseRequired):
                AppleSharpBackend(self.settings(root)).run(
                    root / "image.png", root / "out", root / "log"
                )

    def test_settings_reuse_existing_spectra_sharp_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sharp = root / ".spectra" / "miniconda" / "envs" / "sharp" / "bin" / "sharp"
            sharp.parent.mkdir(parents=True)
            sharp.write_text("#!/bin/sh\n")
            with patch.object(Path, "home", return_value=root), patch.dict(
                os.environ,
                {
                    "DEEPSTAR3D_HOME": str(root / "home"),
                    "DEEPSTAR3D_CACHE": str(root / "cache"),
                },
                clear=True,
            ):
                settings = Settings.from_environment()
            self.assertEqual(settings.sharp_executable, sharp)

    def test_apple_backend_reuses_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sharp = root / "sharp"
            sharp.write_text("#!/bin/sh\n")
            checkpoint = root / "sharp_2572gikvuh.pt"
            checkpoint.write_bytes(b"weights")
            settings = Settings(
                root / "home",
                root / "cache",
                sharp,
                True,
                None,
                checkpoint,
            )

            def fake_run(argv, **kwargs):
                output = Path(argv[argv.index("-o") + 1])
                output.joinpath("image.ply").write_text("ply\n")
                return type("Result", (), {"returncode": 0})()

            with patch(
                "deepstar_v14_3d.backends.active_training_processes", return_value=[]
            ), patch("deepstar_v14_3d.backends.shutil.which", return_value=None), patch(
                "deepstar_v14_3d.backends.subprocess.run", side_effect=fake_run
            ) as run:
                AppleSharpBackend(settings).run(
                    root / "image.png", root / "out", root / "log"
                )
            argv = run.call_args.args[0]
            self.assertEqual(argv[argv.index("-c") + 1], str(checkpoint))

    def test_vision_sharp_hybrid_requires_the_same_research_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(AppleResearchLicenseRequired):
                AppleVisionSharpHybridBackend(self.settings(root)).run(
                    root / "image.png", root / "out", root / "log"
                )

    def test_vision_sharp_hybrid_keeps_sharp_ply_unchanged_and_records_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source.write_bytes(b"original-image-bytes")
            output = root / "out"
            output.mkdir()
            scene = output / "scene.ply"
            original_ply = b"ply\nformat ascii 1.0\nend_header\n"
            scene.write_bytes(original_ply)
            analysis = VisionAnalysis(
                engine="test-vision",
                mask_width=2,
                mask_height=2,
                mask=b"\xff" * 4,
                faces=[{"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5, "confidence": 1.0}],
            )
            with patch.object(
                AppleSharpBackend,
                "run",
                return_value=BackendResult(scene, "apple-ml-sharp-research", True),
            ) as sharp_run, patch(
                "deepstar_v14_3d.backends._run_vision_analysis",
                return_value=analysis,
            ):
                result = AppleVisionSharpHybridBackend(self.settings(root, accepted=True)).run(
                    source, output, root / "hybrid.log"
                )
            self.assertEqual(sharp_run.call_args.args[0], source)
            self.assertEqual(scene.read_bytes(), original_ply)
            self.assertEqual(result.identifier, "deepstar-vision-guided-apple-sharp-research")
            self.assertFalse(result.metadata["sharp"]["ply_postprocessed"])
            self.assertEqual(result.metadata["vision"]["face_count"], 1)

    def test_vision_mask_creates_subject_volume_instead_of_background_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_path = root / "mask.pgm"
            mask_values = bytes(
                [0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0]
            )
            mask_path.write_bytes(b"P5\n4 4\n255\n" + mask_values)
            width, height, mask = _read_pgm(mask_path)
            analysis = VisionAnalysis(
                engine="test-vision",
                mask_width=width,
                mask_height=height,
                mask=mask,
                faces=[{"x": 0.25, "y": 0.25, "width": 0.5, "height": 0.5, "confidence": 1.0}],
            )
            scene = root / "scene.ply"
            geometry = _write_spatial_portrait(
                scene,
                4,
                4,
                [(180, 120, 90)] * 16,
                analysis,
            )
            self.assertLess(geometry["front_points"], 16)
            self.assertGreater(geometry["total_points"], geometry["front_points"])
            self.assertTrue(geometry["portrait_face_used"])
            self.assertIn("property uchar alpha", scene.read_text())

    def test_apple_backend_respects_training_guard_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sharp = root / "sharp"
            sharp.write_text("#!/bin/sh\nexit 0\n")
            sharp.chmod(0o755)
            with patch(
                "deepstar_v14_3d.backends.active_training_processes",
                return_value=["12 python -m deepstar_models.train"],
            ), self.assertRaises(TrainingActive):
                AppleSharpBackend(self.settings(root, accepted=True, sharp=sharp)).run(
                    root / "image.png", root / "out", root / "log"
                )


if __name__ == "__main__":
    unittest.main()
