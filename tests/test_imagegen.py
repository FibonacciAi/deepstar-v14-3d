from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deepstar_v14_3d.errors import ImageGenerationFailed
from deepstar_v14_3d.imagegen import generate_with_command


class ImageGeneratorTests(unittest.TestCase):
    def test_requires_prompt_and_output_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ImageGenerationFailed):
                generate_with_command("generator --prompt {prompt}", "scene", Path(temporary) / "x.png")

    def test_invokes_without_shell_and_requires_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "generated.png"
            def fake_run(argv, **kwargs):
                self.assertNotIn("shell", kwargs)
                self.assertIn("quiet stars", argv)
                output.write_bytes(b"image")
                return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            with patch("deepstar_v14_3d.imagegen.subprocess.run", side_effect=fake_run):
                result = generate_with_command(
                    "generator --prompt {prompt} --output {output}", "quiet stars", output
                )
            self.assertEqual(result, output)


if __name__ == "__main__":
    unittest.main()

