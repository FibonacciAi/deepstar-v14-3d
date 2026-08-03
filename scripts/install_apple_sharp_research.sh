#!/bin/sh
set -eu

if [ "${1:-}" != "--accept-research-license" ]; then
  echo "Stopped: read THIRD_PARTY.md, then rerun with --accept-research-license." >&2
  exit 2
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")
VENDOR_DIR="$REPO_DIR/vendor/apple-ml-sharp"
VENV_DIR="$REPO_DIR/.venv-sharp"
PIN="1eaa046834b81852261262b41b0919f5c1efdd2e"
PYTHON_BIN="${DEEPSTAR3D_SHARP_PYTHON:-/opt/homebrew/bin/python3.13}"

if [ -e "$VENDOR_DIR" ] || [ -e "$VENV_DIR" ]; then
  echo "Stopped: the isolated Apple SHARP installation already exists." >&2
  exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Stopped: Apple SHARP's recommended Python 3.13 was not found at $PYTHON_BIN." >&2
  echo "Set DEEPSTAR3D_SHARP_PYTHON to an isolated Python 3.13 executable." >&2
  exit 2
fi

mkdir -p "$REPO_DIR/vendor"
git clone https://github.com/apple/ml-sharp.git "$VENDOR_DIR"
git -C "$VENDOR_DIR" checkout --detach "$PIN"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$VENDOR_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m pip install --no-deps -e "$VENDOR_DIR"

echo "Installed the research adapter in this repository only."
echo "Review $VENDOR_DIR/LICENSE_MODEL before use."
echo "Set DEEPSTAR3D_ACCEPT_APPLE_RESEARCH_LICENSE=1 to enable runtime use."
