from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .errors import BackendUnavailable, ImageGenerationFailed


def generate_with_command(command_template: str | None, prompt: str, output: Path) -> Path:
    if not command_template:
        raise BackendUnavailable(
            "Prompt generation is not configured. Set DEEPSTAR3D_IMAGE_COMMAND."
        )
    try:
        tokens = shlex.split(command_template)
    except ValueError as error:
        raise ImageGenerationFailed(f"Invalid image command: {error}") from error
    if not tokens or not any("{prompt}" in token for token in tokens):
        raise ImageGenerationFailed("Image command must contain {prompt}.")
    if not any("{output}" in token for token in tokens):
        raise ImageGenerationFailed("Image command must contain {output}.")
    argv = [token.replace("{prompt}", prompt).replace("{output}", str(output)) for token in tokens]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=900, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise ImageGenerationFailed(f"Image generator failed ({result.returncode}): {detail}")
    if not output.is_file() or output.stat().st_size == 0:
        raise ImageGenerationFailed("Image generator did not create the requested output file.")
    return output

