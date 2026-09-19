from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from deepstar_v14_3d.config import Settings
from deepstar_v14_3d.server import WEB_ROOT, StudioHandler, backend_status
from deepstar_v14_3d.workspace import Job


class StatusTests(unittest.TestCase):
    def test_status_reports_hybrid_as_gated_until_sharp_and_license_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(
                root / "home", root / "cache", root / "missing-sharp", False, None
            )
            hybrid = backend_status(settings)["vision-sharp-hybrid"]
            self.assertFalse(hybrid["available"])
            self.assertTrue(hybrid["researchOnly"])
            self.assertFalse(hybrid["plyPostprocessed"])


class WebBundleTests(unittest.TestCase):
    def test_sharp_preview_uses_vendored_gaussian_renderer(self) -> None:
        index = (WEB_ROOT / "index.html").read_text()
        app = (WEB_ROOT / "app.js").read_text()
        engine = WEB_ROOT / "vendor" / "playcanvas-2.22.2.min.js"
        license_file = WEB_ROOT / "vendor" / "PLAYCANVAS-LICENSE.txt"
        self.assertIn('/vendor/playcanvas-2.22.2.min.js', index)
        self.assertIn("new GaussianSplatViewer", app)
        self.assertIn("data.numSplats", app)
        self.assertGreater(engine.stat().st_size, 1_000_000)
        self.assertIn("PlayCanvas Ltd.", license_file.read_text())
        self.assertIn("Permission is hereby granted", license_file.read_text())


class ExportEndpointTests(unittest.TestCase):
    def test_export_artifact_is_downloadable_and_job_root_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = Settings(root / "home", root / "cache", root / "missing-sharp", False, None)
            job = Job.create(settings)
            scene = job.output_dir / "scene.ply"
            scene.write_text(
                "ply\nformat ascii 1.0\nelement vertex 1\n"
                "property float x\nproperty float y\nproperty float z\n"
                "end_header\n0 0 0\n"
            )
            job.write_manifest(
                {
                    "status": "complete",
                    "rights_confirmed": True,
                    "backend": {"identifier": "test-backend", "research_only": False},
                    "scene": {"filename": "scene.ply", "format": "ply"},
                }
            )
            (job.output_dir / "subject-mask.pgm").write_bytes(b"P5\n1 1\n255\n\xff")
            (job.output_dir / "vision-analysis.json").write_text(
                '{"engine":"test-vision","faces":[]}\n'
            )
            package = job.export_package()
            self.assertEqual(package.name, "scene-package.zip")
            with zipfile.ZipFile(package) as archive:
                self.assertIn("scene.ply", archive.namelist())
                self.assertIn("guidance/subject-mask.pgm", archive.namelist())
                self.assertIn("guidance/vision-analysis.json", archive.namelist())
                metadata = json.loads(archive.read("metadata.json"))
                self.assertTrue(metadata["privacy"]["derived_vision_guidance_included"])

            handler = StudioHandler.__new__(StudioHandler)
            handler.server = SimpleNamespace(settings=settings)
            self.assertEqual(handler._job_root(job.identifier), job.root)
            self.assertEqual(handler._export_path(job.identifier), package.resolve())
            for identifier in ("../", f"{job.identifier}/../other", "job%2F.."):
                with self.assertRaises(ValueError):
                    handler._export_path(identifier)


if __name__ == "__main__":
    unittest.main()
