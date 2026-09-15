from __future__ import annotations

import base64
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .backends import active_training_processes
from .config import Settings
from .pipeline import create_scene


MAX_REQUEST = 32 * 1024 * 1024
WEB_ROOT = Path(__file__).with_name("web")


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], settings: Settings):
        self.settings = settings
        super().__init__(address, StudioHandler)


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, format: str, *args: object) -> None:
        print("deepstar3d-ui:", format % args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            training = active_training_processes()
            self._json(
                {
                    "appleSharpInstalled": self.server.settings.sharp_executable.is_file(),
                    "appleLicenseAccepted": self.server.settings.accept_apple_research_license,
                    "imageGeneratorConfigured": bool(self.server.settings.image_command),
                    "trainingActive": bool(training),
                    "trainingMatches": training,
                }
            )
            return
        if path.startswith("/api/jobs/"):
            identifier = path.removeprefix("/api/jobs/").strip("/")
            try:
                manifest = self._job_root(identifier).joinpath("manifest.json")
                self._json(json.loads(manifest.read_text()))
            except (ValueError, FileNotFoundError, json.JSONDecodeError):
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if path.startswith("/artifacts/"):
            parts = unquote(path).removeprefix("/artifacts/").split("/", 1)
            if len(parts) != 2:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                root = self._job_root(parts[0])
                requested = (root / parts[1]).resolve()
                if root.resolve() not in requested.parents or not requested.is_file():
                    raise ValueError
                self._file(requested, download=requested.suffix == ".ply")
            except ValueError:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        asset = "index.html" if path == "/" else path.lstrip("/")
        requested = (WEB_ROOT / asset).resolve()
        if WEB_ROOT.resolve() not in requested.parents or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._file(requested)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/jobs":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST:
                raise ValueError("Request is empty or too large.")
            payload = json.loads(self.rfile.read(length))
            if payload.get("confirmRights") is not True:
                raise ValueError("Rights confirmation is required.")
            image_path: Path | None = None
            prompt: str | None = None
            temporary: Path | None = None
            if payload.get("imageData"):
                header, encoded = payload["imageData"].split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValueError("Only image uploads are accepted.")
                suffix = _safe_suffix(payload.get("imageName", "image.png"))
                temporary = self.server.settings.home / f".upload-{threading.get_ident()}{suffix}"
                temporary.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_bytes(base64.b64decode(encoded, validate=True))
                image_path = temporary
            elif str(payload.get("prompt", "")).strip():
                prompt = str(payload["prompt"]).strip()
            else:
                raise ValueError("Choose an image or enter a prompt.")
            try:
                job = create_scene(
                    self.server.settings,
                    image=image_path,
                    prompt=prompt,
                    backend=str(payload.get("backend", "depth-card")),
                    confirm_rights=True,
                    allow_concurrent_training=False,
                    device=str(payload.get("device", "default")),
                )
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
            self._json(
                {
                    "jobId": job.identifier,
                    "manifest": job.manifest(),
                    "sceneUrl": f"/artifacts/{job.identifier}/outputs/scene.ply",
                    "manifestUrl": f"/artifacts/{job.identifier}/manifest.json",
                },
                status=HTTPStatus.CREATED,
            )
        except Exception as error:
            self._json(
                {"error": type(error).__name__, "message": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def _job_root(self, identifier: str) -> Path:
        if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in identifier):
            raise ValueError
        root = self.server.settings.home / "jobs" / identifier
        if not root.is_dir():
            raise ValueError
        return root

    def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, download: bool = False) -> None:
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)


def _safe_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return suffix if suffix in {".png", ".jpg", ".jpeg", ".heic", ".webp", ".tif", ".tiff"} else ".png"


def serve(settings: Settings, port: int, open_browser: bool) -> None:
    settings.prepare()
    server = StudioServer(("127.0.0.1", port), settings)
    url = f"http://127.0.0.1:{port}"
    print(f"Deepstar v14 3D is running at {url}")
    print("Press Control-C to stop. Existing Deepstar apps and training data are untouched.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
