from __future__ import annotations

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepstar_v14_3d.backends import AppleSharpBackend
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
            ) as run:
                def fake_sips(argv, **kwargs):
                    Path(argv[-1]).write_bytes(image.read_bytes())
                    return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
                run.side_effect = fake_sips
                job = create_scene(settings, image=image, backend="depth-card", confirm_rights=True)
            manifest = json.loads(job.manifest_path.read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["backend"]["identifier"], "deepstar-depth-card-v1")
            self.assertTrue((job.output_dir / "scene.ply").is_file())
            self.assertTrue(str(job.root).startswith(str(settings.home)))

    def test_apple_backend_requires_explicit_research_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(AppleResearchLicenseRequired):
                AppleSharpBackend(self.settings(root)).run(
                    root / "image.png", root / "out", root / "log"
                )

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

