from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepstar_v14_3d.config import Settings
from deepstar_v14_3d.server import backend_status


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


if __name__ == "__main__":
    unittest.main()
